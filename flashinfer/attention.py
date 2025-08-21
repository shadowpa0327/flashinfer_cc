"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import Optional, Tuple, Union

import torch

from .jit import gen_batch_attention_module
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
)

from .triton.batch_attn_utils import augment_head_major_triton


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    def __init__(
        self,
        kv_layout: str = "NHD",
        device: str = "cuda",
    ):
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout

        self.float_workspace_buffer = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor, # (num_layers, kv_heads, total_kv_indices) or (kv_heads, total_kv_indices)
        kv_len_arr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        layer_idx: torch.Tensor, # a 0-D tensors, buffer for deciding which layers of per-head kv indices to use
        causal: bool = False,
        is_per_head_indices: bool = False,
        sm_scale: float = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        use_profiler: bool = False,
        add_layer_idx_by_one_after_run: bool = False,
    ) -> None:
        # get jit module
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            use_profiler,  # different compiler path
        )
        self.module = get_holistic_attention_module(*get_module_args)

        qo_indptr_host = qo_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_indptr_host = kv_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_len_arr_host = kv_len_arr.to(torch.device("cpu"), non_blocking=True)
        torch.cuda.synchronize()

        batch_size = kv_len_arr.shape[0]
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler

        # No addtional buf allocated for CUDA graph tensor
        # Allocate outside FlashInfer
        self._kv_indices = kv_indices

        #NOTE(brian1009): For assisting kv_indices loading.
        #IMPORTANT!!!!!!!!!!!!!!
        #If kv_indices is a 2D tensor, the user should make sure that the layer_idx should not exceed self._kv_indices.shape[0].
        #If the layer_idx exceeds self._kv_indices.shape[0], the behavior is undefined.
        #If kv_indices is a 1D tensor, the user should make sure that the layer_idx should not exceed self._kv_indices.shape[0].
        #Then make sure that the layer_idx is always 0.
        self._layer_idx = layer_idx
        # If set, the self._layer_idx will be in-place added by one after each call of run()
        self._add_layer_idx_by_one_after_run = add_layer_idx_by_one_after_run
        self._is_per_head_indices = is_per_head_indices

        self._plan_info = self.module.plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        profiler_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        
        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            # lse shape: [batch_size, num_qo_heads]
            lse = torch.empty(q.shape[0], q.shape[1], device=q.device)

        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)

        # profiler_buffer is optional
        profiler_args = (profiler_buffer,) if self._use_profiler else ()

        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            self._layer_idx,
            self._is_per_head_indices,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self._sm_scale,
            *profiler_args,
        )

        if self._add_layer_idx_by_one_after_run:
            # inplace adding one.
            self._layer_idx.add_(1)

        return out, lse



class BatchAttentionWithPerHeadSelectPagedKVCacheWrapper:
    """
    FlashInfer wrapper for BatchAttention with per-layer-per-head KV selection.
    
    This wrapper extends FlashInfer's BatchAttention to enable different KV token 
    selection patterns for each KV head during attention operations.
    """
    
    def __init__(self, kv_layout: str = "NHD", device: str = "cuda"):
        """
        Initialize the per-head KV cache batch attention wrapper.
        
        Args:
            kv_layout: Layout for KV cache ("NHD" or "HND")
            device: Device to run on
        """
        self.device = torch.device(device)
        self.wrapper = BatchAttention(kv_layout=kv_layout)
    
    def plan(
        self,
        qo_indptr: torch.Tensor,                          # (batch_size + 1,) query token boundaries
        kv_indptr: torch.Tensor,                          # (batch_size + 1,) KV token boundaries
        all_layers_per_head_kv_indices: torch.Tensor,     # (num_layers, num_kv_heads, total_kv_indices) 
        seq_lens: torch.Tensor,                           # (batch_size,) sequence lengths
        num_qo_heads: int,
        num_kv_heads: int, 
        head_dim_qk: int,
        head_dim_vo: int,
        layer_idx: torch.Tensor,                          # a 0-D tensors, buffer for deciding which layers of per-head kv indices to use
        page_block_size: int = 1,
        causal: bool = True,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        use_triton: bool = True,
        add_layer_idx_by_one_after_run: bool = True,
    ):
        """
        Plan batch attention execution with per-KV-head token selection.
        
        Transform structure:
        - Original: batch_size requests with variable query tokens
        - Virtual: (batch_size × num_kv_heads) virtual batches
        - Each group of num_kv_heads virtual batches shares the same queries
        
        Args:
            qo_indptr: Query/output token boundaries for each request (batch_size + 1,)
            kv_indptr: KV token boundaries shared across all heads (batch_size + 1,)
            per_head_kv_indices: Indices for each head (num_kv_heads, total_kv_indices)
            seq_lens: Sequence lengths for each request (batch_size,)
            causal: Whether to use causal attention mask
            use_triton: Whether to use Triton kernel for performance
        """
        batch_size = qo_indptr.shape[0] - 1
        gqa_group_size = num_qo_heads // num_kv_heads
        
        # =====================================================================================
        # PHASE 1: Transform qo_indptr for Virtual Batches
        # =====================================================================================
        # Same as prefill wrapper - replicate query boundaries for each head
        qo_lengths = qo_indptr[1:] - qo_indptr[:-1]  # (batch_size,)
        
        # Build virtual qo_indptr in head-major order
        qo_lengths_repeated = qo_lengths.repeat(num_kv_heads)
        virtual_qo_indptr = torch.zeros(batch_size * num_kv_heads + 1, device=self.device, dtype=torch.int32)
        virtual_qo_indptr[1:] = qo_lengths_repeated
        torch.cumsum(virtual_qo_indptr, dim=0, out=virtual_qo_indptr)
        
        # =====================================================================================
        # PHASE 2: Transform KV Indices (Head-Major Order)
        # =====================================================================================
        # Calculate output offsets for head-major organization
        kv_batch_lengths = kv_indptr[1:] - kv_indptr[:-1]  # (batch_size,)
        
        # Create output_offsets for head-major organization
        kv_batch_lengths_repeated = kv_batch_lengths.repeat(num_kv_heads)
        output_offsets = torch.zeros(batch_size * num_kv_heads + 1, device=self.device, dtype=torch.int32)
        output_offsets[1:] = kv_batch_lengths_repeated
        torch.cumsum(output_offsets, dim=0, out=output_offsets)
        
        # Apply augmentation: block_idx * num_kv_heads + head_idx
        if use_triton:
            # Triton kernel for head-major augmentation (handles 3D tensors)
            try:
                augmented_indices = augment_head_major_triton(
                    per_head_kv_indices=all_layers_per_head_kv_indices,
                    device=str(self.device)
                )
            except (ImportError, RuntimeError):
                print("[Warning] Triton kernel not available, falling back to PyTorch")
                # Fallback to PyTorch if Triton kernel not available
                use_triton = False
        
        if not use_triton:
            # Optimized PyTorch implementation - process all layers at once
            head_offsets = torch.arange(num_kv_heads, device=self.device, dtype=all_layers_per_head_kv_indices.dtype)
            # Broadcast head_offsets to match 3D tensor shape
            # per_head_kv_indices: (num_layers, num_kv_heads, total_kv_indices)
            # head_offsets: (num_kv_heads,) -> (1, num_kv_heads, 1)
            augmented_per_head = all_layers_per_head_kv_indices * num_kv_heads + head_offsets.unsqueeze(0).unsqueeze(2)
            
            # Flatten the last two dimensions for each layer
            # Result: (num_layers, num_kv_heads * total_kv_indices)
            augmented_indices = augmented_per_head.flatten(start_dim=1)
        
        # =====================================================================================
        # PHASE 3: Transform seq_lens for Virtual Batches
        # =====================================================================================
        # Replicate seq_lens for each head in head-major order
        # Original: [seq_len_0, seq_len_1, ...]
        # Virtual: [seq_len_0, seq_len_1, ..., seq_len_0, seq_len_1, ..., ...]
        #          ^--------head 0---------^  ^--------head 1---------^
        virtual_seq_lens = seq_lens.repeat(num_kv_heads)
        
        # =====================================================================================
        # PHASE 4: FlashInfer Planning
        # =====================================================================================
        # Plan underlying FlashInfer wrapper with transformed data
        self.wrapper.plan(
            virtual_qo_indptr,      # Transformed query boundaries
            output_offsets,         # Virtual KV batch boundaries (augmented_indptr)
            augmented_indices,      # Augmented and interleaved KV indices
            virtual_seq_lens,       # Replicated sequence lengths for virtual batches
            gqa_group_size,         # num_qo_heads per virtual batch
            1,                      # num_kv_heads=1 since we unrolled
            head_dim_qk,               # head dimension
            head_dim_vo,               # head dimension for values
            page_block_size,
            layer_idx=layer_idx,
            causal=causal,
            q_data_type=q_data_type, 
            kv_data_type=kv_data_type,
            add_layer_idx_by_one_after_run=add_layer_idx_by_one_after_run,
        )
    
    def run(
        self,
        query: torch.Tensor,  # (total_query_tokens, num_qo_heads, head_dim)
        paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],  # KV cache tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run batch attention with per-KV-head token selection.
        
        Args:
            query: Query tensor (total_query_tokens, num_qo_heads, head_dim)
            paged_kv_cache: Tuple of (key, value) tensors
            
        Returns:
            output: Attention output (total_query_tokens, num_qo_heads, head_dim)
            lse: Log-sum-exp values
        """
        # NOTE(brian1009): defer import following flashinfer/sparse.py
        import einops

        key, value = paged_kv_cache
        total_query_tokens, num_qo_heads, head_dim = query.shape
        total_blocks, page_block_size, num_kv_heads, _ = key.shape
        gqa_group_size = num_qo_heads // num_kv_heads
        
        # =====================================================================================
        # PHASE 1: Reshape KV Cache (Same as Prefill)
        # =====================================================================================
        # Use interleaved layout for KV cache
        key_reshaped = key.view(total_blocks * num_kv_heads, page_block_size, 1, head_dim)
        value_reshaped = value.view(total_blocks * num_kv_heads, page_block_size, 1, head_dim)
        
        # =====================================================================================
        # PHASE 2: Query Transformation to Head-Major Layout
        # =====================================================================================
        # Same transformation as prefill wrapper
        
        # Using einops for clearer dimension rearrangement
        query_for_flashinfer = einops.rearrange(
            query,
            'tokens (kv_heads group_size) head_dim -> (kv_heads tokens) group_size head_dim',
            tokens=total_query_tokens,
            kv_heads=num_kv_heads,
            group_size=gqa_group_size
        ).contiguous()
        
        # =====================================================================================
        # PHASE 3: Run Attention
        # =====================================================================================
        # BatchAttention.run() returns (output, lse) directly
        output, lse = self.wrapper.run(query_for_flashinfer, (key_reshaped, value_reshaped))
        
        # =====================================================================================
        # PHASE 4: Reshape Output Back
        # =====================================================================================
        # Reverse transformation using einops
        output_final = einops.rearrange(
            output,
            '(kv_heads tokens) group_size head_dim -> tokens (kv_heads group_size) head_dim',
            kv_heads=num_kv_heads,
            tokens=total_query_tokens,
            group_size=gqa_group_size
        ).contiguous()
        
        # Similar for LSE
        lse_final = einops.rearrange(
            lse,
            '(kv_heads tokens) group_size -> tokens (kv_heads group_size)',
            kv_heads=num_kv_heads,
            tokens=total_query_tokens,
            group_size=gqa_group_size
        ).contiguous()
        
        return output_final, lse_final
