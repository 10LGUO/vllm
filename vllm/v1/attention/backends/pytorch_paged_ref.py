# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch paged attention reference, opt-in via VLLM_PYTORCH_PAGED_ATTN=1.

Replaces flash_attn_varlen_func in FlashAttentionImpl.forward for validation
of KV cache quantization work: numerics were verified against the kernel at
bf16 (see the qwen-int8-kv-cache-quantization repo, paged_attention_ref.py).
Requires enforce_eager=True; slow by design (materializes full score
matrices). Supports the plain decoder path only: causal, no alibi, no
sliding window, no soft cap, no sinks.

pytorch_paged_attention_int8 is the INT8 variant: the paged cache holds int8
codes with static per-channel scales [num_kv_heads, head_size]; codes are
dequantized to the query dtype after the paged gather, mirroring a
dequant-then-attend deployment.
"""

import os

import torch

PYTORCH_PAGED_ATTN_ENABLED = os.getenv("VLLM_PYTORCH_PAGED_ATTN", "0") == "1"


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

        if k_scale is not None:
            k = (k.float() * k_scale).to(query.dtype)
            v = (v.float() * v_scale).to(query.dtype)

        if group_size > 1:
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale

        q_pos = torch.arange(q_len, device=query.device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=query.device).unsqueeze(0)
        visible = kv_pos <= (kv_len - q_len) + q_pos
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        output[q_start : q_start + q_len] = torch.einsum("hqk,khd->qhd", probs, v)
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
    """Causal paged attention over an INT8-quantized KV cache.

    key_cache/value_cache hold int8 codes in the same paged layout as the
    bf16 path; k_scale/v_scale are static per-channel fp32 scales of shape
    [num_kv_heads, head_size] (code * scale ~= original value). Gathered
    codes are dequantized to query.dtype, then attention proceeds
    identically to pytorch_paged_attention.
    """
    assert key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8
    assert k_scale.shape == key_cache.shape[-2:], k_scale.shape
    assert v_scale.shape == value_cache.shape[-2:], v_scale.shape
    _paged_attention(
        output,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        k_scale.float(),
        v_scale.float(),
    )
