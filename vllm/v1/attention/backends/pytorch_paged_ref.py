# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch paged attention reference, opt-in via VLLM_PYTORCH_PAGED_ATTN=1.

Replaces flash_attn_varlen_func in FlashAttentionImpl.forward for validation
of KV cache quantization work: numerics were verified against the kernel at
bf16 (see the qwen-int8-kv-cache-quantization repo, paged_attention_ref.py).
Requires enforce_eager=True; slow by design (materializes full score
matrices). Supports the plain decoder path only: causal, no alibi, no
sliding window, no soft cap, no sinks.

pytorch_paged_attention_int8 simulates an INT8 attention kernel: Q is
dynamically quantized per-channel at entry; the paged cache holds int8
codes with static per-channel scales [num_kv_heads, head_size]. Since
PyTorch has no int8 matmul ops, the int8 x int8 products are simulated as
code.float() * scale dequantization followed by fp32 contractions --
mirroring the arithmetic a real int8 kernel performs, with the fp32
accumulate kept for the softmax. Output is cast back to the model dtype
on the final write.
"""

import os

import torch

PYTORCH_PAGED_ATTN_ENABLED = os.getenv("VLLM_PYTORCH_PAGED_ATTN", "0") == "1"
PYTORCH_PAGED_ATTN_INT8_ENABLED = (
    os.getenv("VLLM_PYTORCH_PAGED_ATTN_INT8", "0") == "1"
)

_scales_file: dict | None = None
_scales_cache: dict = {}


def int8_quant_per_channel(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel INT8 quantization of [num_tokens, num_heads,
    head_size]: one scale per (head, channel), shared across tokens.

    Returns (codes int8, scale fp32 [num_heads, head_size])."""
    scale = x.float().abs().amax(dim=0).clamp_min(1e-8) / 127.0
    codes = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return codes, scale


def get_layer_scales(
    layer_name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads static per-channel (k_scale, v_scale) for a layer from the
    calibration file at $VLLM_KV_INT8_SCALES ({layer_idx: {"k_scale":
    [num_kv_heads, head_size], "v_scale": ...}})."""
    from vllm.model_executor.models.utils import extract_layer_index

    global _scales_file
    idx = extract_layer_index(layer_name)
    key = (idx, str(device))
    if key not in _scales_cache:
        if _scales_file is None:
            _scales_file = torch.load(
                os.environ["VLLM_KV_INT8_SCALES"], map_location="cpu"
            )
        entry = _scales_file[idx]
        _scales_cache[key] = (
            entry["k_scale"].float().to(device),
            entry["v_scale"].float().to(device),
        )
    return _scales_cache[key]


def int8_cache_views(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reinterprets the engine's bf16 [num_blocks, num_kv_heads, block_size,
    2 * head_size] cache storage as int8 code views [num_blocks, block_size,
    num_kv_heads, head_size] for K and V. Only the front half of each row's
    bytes is used; memory footprint is unchanged (capacity gains need the
    CacheDType-level integration)."""
    i8 = kv_cache.view(torch.int8)[..., : 2 * head_size]
    key_cache, value_cache = i8.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


def int8_quantize_and_scatter(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantizes new K/V tokens with static per-channel scales and scatters
    the int8 codes into the paged cache views (the write-side counterpart of
    reshape_and_cache_flash)."""
    block_size = key_cache.shape[1]
    num_tokens = slot_mapping.shape[0]
    slots = slot_mapping.long()
    valid = slots >= 0
    slots = slots[valid]
    k = key[:num_tokens][valid].float()
    v = value[:num_tokens][valid].float()
    k_codes = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_codes = (v / v_scale).round().clamp(-128, 127).to(torch.int8)
    key_cache[slots // block_size, slots % block_size] = k_codes
    value_cache[slots // block_size, slots % block_size] = v_codes


def _paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    _, block_size, num_kv_heads, head_size = key_cache.shape
    num_q_heads = query.shape[1]
    group_size = num_q_heads // num_kv_heads

    if k_scale is not None:
        # expand per-kv-head scales to query heads once (GQA)
        k_scale_q = k_scale.repeat_interleave(group_size, dim=0)
        v_scale_q = v_scale.repeat_interleave(group_size, dim=0)

    query_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    kv_lens = seqused_k.tolist()

    q_start = 0
    for seq_idx, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        if q_len == 0:
            continue
        q = query[q_start : q_start + q_len]

        num_blocks_used = (kv_len + block_size - 1) // block_size
        physical_blocks = block_table[seq_idx, :num_blocks_used].long()
        k = key_cache[physical_blocks].reshape(-1, num_kv_heads, head_size)
        v = value_cache[physical_blocks].reshape(-1, num_kv_heads, head_size)
        k, v = k[:kv_len], v[:kv_len]

        if group_size > 1:
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)

        q_f = q.float()
        if k_scale is not None:
            # fold K's per-channel scale into Q instead of dequantizing K:
            # sum_d (q_d*s_kd)*code_d == sum_d q_d*(code_d*s_kd)
            q_f = q_f * k_scale_q

        # p_ = q * k, fp32
        scores = torch.einsum("qhd,khd->hqk", q_f, k.float()) * scale

        q_pos = torch.arange(q_len, device=query.device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=query.device).unsqueeze(0)
        visible = kv_pos <= (kv_len - q_len) + q_pos
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

        # p = softmax(p_), fp32
        probs = torch.softmax(scores, dim=-1)
        if v_scale is None:
            probs = probs.to(v.dtype)
            out = torch.einsum("hqk,khd->qhd", probs, v)
        else:
            # V's per-channel scale is not on the summed (token) axis, so it
            # factors out of the PV contraction and is applied once at the end
            out = torch.einsum("hqk,khd->qhd", probs, v.float()) * v_scale_q
        output[q_start : q_start + q_len] = out
        q_start += q_len


def pytorch_paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> None:
    """Computes causal paged attention with plain PyTorch ops.

    Args:
        output: [num_tokens, num_q_heads, head_size], written in place.
        query: [num_tokens, num_q_heads, head_size], varlen-packed.
        key_cache: [num_blocks, block_size, num_kv_heads, head_size].
        value_cache: same layout as key_cache.
        cu_seqlens_q: [num_seqs + 1] cumulative query lengths.
        seqused_k: [num_seqs] kv length per sequence.
        block_table: [num_seqs, max_blocks_per_seq] physical block ids.
        scale: softmax temperature (1/sqrt(head_size)).
    """
    _paged_attention(
        output,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        None,
        None,
    )


def pytorch_paged_attention_int8(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> None:
    """Causal paged attention simulating an INT8 kernel end to end.

    Q is dynamically quantized per-channel here (q_int8, q_scale =
    int8_quant(q)) and dequantized with its own scale; key_cache/value_cache
    hold int8 codes with static per-channel fp32 scales [num_kv_heads,
    head_size]. K's scale is folded into Q before the QK contraction and
    V's scale is applied after the PV contraction, so the cached codes are
    never elementwise-dequantized. Softmax stays fp32; output is cast to
    the model dtype on the final write.
    """
    assert key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8
    assert k_scale.shape == key_cache.shape[-2:], k_scale.shape
    assert v_scale.shape == value_cache.shape[-2:], v_scale.shape
    q_codes, q_scale = int8_quant_per_channel(query)
    q_deq = q_codes.float() * q_scale
    _paged_attention(
        output,
        q_deq,
        key_cache,
        value_cache,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        k_scale.float(),
        v_scale.float(),
    )
