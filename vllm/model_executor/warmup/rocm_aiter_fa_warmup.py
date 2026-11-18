# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup run of ROCM_AITER_FA's extend path."""

import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def rocm_aiter_fa_extend_warmup(worker: "Worker") -> None:
    """Run ROCM_AITER_FA's extend path for each attention layer configuration.

    Only a query chunk after cached tokens (a prefix-cache hit or a later prefill
    chunk) reaches the extend path, and no dummy run has cached tokens. Its
    context gather and the AITER varlen variants it calls would otherwise load,
    or for variants missing from the AITER wheel JIT-build, on the first such
    request.
    """
    runner = worker.model_runner
    layers = worker.vllm_config.compilation_config.static_forward_context
    seen = set()
    groups = []
    for i, kv_groups in enumerate(runner.attn_groups):
        for group in kv_groups:
            if not issubclass(group.backend, AiterFlashAttentionBackend):
                continue
            names = []
            for name in group.layer_names:
                layer = layers[name]
                impl = layer.impl
                config = (
                    impl.num_heads,
                    impl.num_kv_heads,
                    impl.head_size,
                    impl.sliding_window,
                    impl.logits_soft_cap,
                    impl.sinks is None,
                    impl.alibi_slopes is None,
                    impl.kv_cache_dtype,
                    tuple(layer.kv_cache.shape),
                )
                if (
                    config not in seen
                    and impl.kv_sharing_target_layer_name is None
                    and layer.kv_cache.numel() > 0
                ):
                    seen.add(config)
                    names.append(name)
            if names:
                groups.append((i, group, names))
    if not groups:
        return
    start = time.perf_counter()
    device = worker.device
    dtype = worker.model_config.dtype
    block_tables = runner.block_tables.get_dummy_block_tables(1)
    with torch.inference_mode():
        for i, group, names in groups:
            # One request over 16 cached tokens, which the dummy block table puts
            # in the null block. Two query lengths above the decode threshold, one
            # a multiple of 16 and one not, cover Triton's integer specialization.
            threshold = group.get_metadata_builder().reorder_batch_threshold
            assert threshold is not None
            aligned = cdiv(threshold + 1, 16) * 16
            for num_query in (aligned, aligned + 1):
                seq_len = 16 + num_query
                query_start_loc = torch.tensor([0, num_query], dtype=torch.int32)
                seq_lens = torch.tensor([seq_len], dtype=torch.int32)
                metadata = group.build_metadata(
                    CommonAttentionMetadata(
                        query_start_loc=query_start_loc.to(device),
                        query_start_loc_cpu=query_start_loc,
                        seq_lens=seq_lens.to(device),
                        seq_lens_cpu_upper_bound=seq_lens,
                        num_reqs=1,
                        num_actual_tokens=num_query,
                        max_query_len=num_query,
                        max_seq_len=seq_len,
                        block_table_tensor=block_tables[i],
                        slot_mapping=torch.zeros(
                            num_query, dtype=torch.int64, device=device
                        ),
                    )
                )
                for name in names:
                    layer = layers[name]
                    impl = layer.impl
                    query = torch.zeros(
                        num_query,
                        impl.num_heads,
                        impl.head_size,
                        dtype=dtype,
                        device=device,
                    )
                    key = query.new_zeros(num_query, impl.num_kv_heads, impl.head_size)
                    impl.forward(
                        layer,
                        query,
                        key,
                        torch.zeros_like(key),
                        layer.kv_cache,
                        metadata,
                        torch.empty_like(query),
                    )
    torch.accelerator.synchronize()
    logger.info(
        "Warmed up the ROCM_AITER_FA extend path for %d attention layer "
        "configurations in %.1fs.",
        len(seen),
        time.perf_counter() - start,
    )
