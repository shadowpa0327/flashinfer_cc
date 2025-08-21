"""
Pytest for testing native BatchAttention with is_per_head_indices=True against golden reference
"""

import torch
import numpy as np
import pytest
import flashinfer

def generate_test_data(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    device, dtype, seed, sparsity_ratio=0.5,
    min_seq_len=100, max_seq_len=500,
    min_q_len=1, max_q_len=50, num_layers=1
):
    """Generate test data for per-head KV selection tests"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate sequence lengths with configurable ranges
    seq_lens = torch.randint(min_seq_len, max_seq_len, (batch_size,), device=device, dtype=torch.int32)
    q_lens = torch.randint(min_q_len, max_q_len, (batch_size,), device=device, dtype=torch.int32)
    
    # Build indptr arrays
    qo_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(q_lens, dim=0)], dim=0).int()
    total_query_tokens = qo_indptr[-1].item()
    
    # Determine tokens to select per batch
    selected_tokens_per_batch = []
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx].item()
        num_selected = max(1, int(seq_len * sparsity_ratio))
        selected_tokens_per_batch.append(num_selected)
    
    kv_indptr = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(selected_tokens_per_batch, device=device), dim=0)
    ], dim=0).int()
    
    # Generate per-head KV indices with shuffling
    all_layers_per_head_kv_indices = []
    for layer_idx in range(num_layers):
        per_head_kv_indices = []
        for head_idx in range(num_kv_heads):
            head_indices = []
            for batch_idx in range(batch_size):
                seq_len = seq_lens[batch_idx].item()
                num_selected = selected_tokens_per_batch[batch_idx]
                
                # Select different random tokens for each head
                selected_local = torch.randperm(seq_len, device=device)[:num_selected].sort()[0]
                
                # Global indices
                global_offset = sum(seq_lens[:batch_idx].tolist()) if batch_idx > 0 else 0
                selected_global = selected_local + global_offset
                
                head_indices.append(selected_global)
            
            head_indices = torch.cat(head_indices)
            per_head_kv_indices.append(head_indices)
        
        per_head_kv_indices = torch.stack(per_head_kv_indices)
        all_layers_per_head_kv_indices.append(per_head_kv_indices)
    
    all_layers_per_head_kv_indices = torch.stack(all_layers_per_head_kv_indices)
    
    # Generate tensors
    query = torch.rand(total_query_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    
    # Use original (non-sparsified) total tokens for KV cache size
    original_total_kv_tokens = seq_lens.sum().item()
    page_block_size = 1
    key = torch.rand(original_total_kv_tokens, page_block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.rand(original_total_kv_tokens, page_block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    # Compute last_page_len
    last_page_len = (seq_lens - 1) % page_block_size + 1
    last_page_len = last_page_len.unsqueeze(1).expand(batch_size, num_kv_heads)
    
    sparsified_seq_lens = torch.tensor(selected_tokens_per_batch, device=device, dtype=torch.int32)
    
    return {
        'query': query,
        'key': key,
        'value': value,
        'qo_indptr': qo_indptr,
        'kv_indptr': kv_indptr,
        'all_layers_per_head_kv_indices': all_layers_per_head_kv_indices,
        'seq_lens': seq_lens,
        'sparsified_seq_lens': sparsified_seq_lens,
        'last_page_len': last_page_len,
        'page_block_size': page_block_size,
    }


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [
    (4, 32),  # 1:8 GQA ratio
    (8, 32),  # 1:4 GQA ratio
])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("max_seq_len,max_q_len", [
    (512, 32),    # Default
    (2048, 128),  # Medium
])
def test_golden_reference_native_with_preallocated_buffer(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    causal, dtype, max_seq_len, max_q_len, device="cuda"
):
    """Test native BatchAttention with pre-allocated kv_indices buffer against golden reference"""
    seed = 42
    
    # Set min lengths relative to max lengths
    min_seq_len = max(100, max_seq_len // 10)
    min_q_len = max(1, max_q_len // 10)
    
    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        min_q_len=min_q_len, max_q_len=max_q_len
    )
    
    # Pre-allocate kv_indices buffer with maximum possible size
    max_kv_len = batch_size * max_seq_len  # Conservative upper bound
    num_layers = data['all_layers_per_head_kv_indices'].shape[0]
    kv_indices_buffer = torch.zeros((num_layers, num_kv_heads, max_kv_len), dtype=torch.int32, device=device)
    
    # Copy actual data into the pre-allocated buffer
    actual_kv_indices = data['all_layers_per_head_kv_indices'].int()
    actual_kv_len = actual_kv_indices.shape[2]  # Get the actual length
    kv_indices_buffer[:, :, :actual_kv_len].copy_(actual_kv_indices)
    
    # Run native BatchAttention with is_per_head_indices=True using pre-allocated buffer
    batch_attention = flashinfer.BatchAttention(kv_layout="NHD", device=device)
    layer_idx_buffer = torch.tensor([0], device=device)
    batch_attention.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        kv_indices=kv_indices_buffer, # Pre-allocated buffer (num_layers, num_kv_heads, max_kv_len)
        kv_len_arr=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=data['page_block_size'],
        layer_idx=layer_idx_buffer, # a 0-D tensors
        causal=causal,
        is_per_head_indices=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        add_layer_idx_by_one_after_run=True
    )
    output_native, lse_native = batch_attention.run(
        data['query'], (data['key'], data['value'])
    )
    # check whether layer_idx_buffer is updated
    assert layer_idx_buffer.item() == 1, "layer_idx_buffer should be updated to 1"
    torch.cuda.synchronize()
    
    # Compute golden reference
    gqa_group_size = num_qo_heads // num_kv_heads
    golden_outputs = []
    golden_lses = []

    for kv_head_idx in range(num_kv_heads):
        # Extract queries for this KV head group
        query_start = kv_head_idx * gqa_group_size
        query_end = (kv_head_idx + 1) * gqa_group_size
        query_per_head = data['query'][:, query_start:query_end, :].contiguous()
        
        # Extract KV cache for this head
        head_kv_indices = data['all_layers_per_head_kv_indices'][0, kv_head_idx] # assuming that num_layers = 1
        key_per_head = data['key'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        value_per_head = data['value'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        
        # Run standard FlashInfer BatchAttention
        wrapper = flashinfer.BatchAttention(kv_layout="NHD")
        wrapper.plan(
            data['qo_indptr'],
            data['kv_indptr'],
            head_kv_indices.int(),
            data['sparsified_seq_lens'],
            gqa_group_size,
            1,
            head_dim,
            head_dim,
            data['page_block_size'],
            layer_idx=torch.tensor(0, device=device), # a 0-D tensors
            causal=causal,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        
        output_per_head, lse_per_head = wrapper.run(query_per_head, (key_per_head, value_per_head))
        torch.cuda.synchronize()
        
        golden_outputs.append(output_per_head)
        golden_lses.append(lse_per_head)
    
    # Concatenate results
    golden_output = torch.cat(golden_outputs, dim=1)
    golden_lse = torch.cat(golden_lses, dim=1)
    
    # Assert with appropriate tolerances
    torch.testing.assert_close(output_native, golden_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_native, golden_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 64])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [
    (4, 32),  # 1:8 GQA ratio
    (8, 32),  # 1:4 GQA ratio
])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("max_seq_len,max_q_len", [
    (512, 32),    # Default
    (2048, 128),  # Medium
    (8192, 256),  # Large
])
def test_golden_reference_native(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    causal, dtype, max_seq_len, max_q_len, device="cuda"
):
    """Test native BatchAttention with is_per_head_indices=True against golden reference"""
    seed = 42
    
    # Set min lengths relative to max lengths
    min_seq_len = max(100, max_seq_len // 10)
    min_q_len = max(1, max_q_len // 10)
    
    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        min_q_len=min_q_len, max_q_len=max_q_len
    )
    
    # Run native BatchAttention with is_per_head_indices=True
    batch_attention = flashinfer.BatchAttention(kv_layout="NHD", device=device)
    layer_idx_buffer = torch.tensor([0], device=device)
    batch_attention.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        kv_len_arr=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=data['page_block_size'],
        layer_idx=layer_idx_buffer, # a 0-D tensors
        causal=causal,
        is_per_head_indices=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        add_layer_idx_by_one_after_run=True
    )
    output_native, lse_native = batch_attention.run(
        data['query'], (data['key'], data['value'])
    )
    # check whether layer_idx_buffer is updated
    assert layer_idx_buffer.item() == 1, "layer_idx_buffer should be updated to 1"
    torch.cuda.synchronize()
    
    # Compute golden reference
    gqa_group_size = num_qo_heads // num_kv_heads
    golden_outputs = []
    golden_lses = []

    for kv_head_idx in range(num_kv_heads):
        # Extract queries for this KV head group
        query_start = kv_head_idx * gqa_group_size
        query_end = (kv_head_idx + 1) * gqa_group_size
        query_per_head = data['query'][:, query_start:query_end, :].contiguous()
        
        # Extract KV cache for this head
        head_kv_indices = data['all_layers_per_head_kv_indices'][0, kv_head_idx] # assuming that num_layers = 1
        key_per_head = data['key'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        value_per_head = data['value'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        
        # Run standard FlashInfer BatchAttention
        wrapper = flashinfer.BatchAttention(kv_layout="NHD")
        wrapper.plan(
            data['qo_indptr'],
            data['kv_indptr'],
            head_kv_indices.int(),
            data['sparsified_seq_lens'],
            gqa_group_size,
            1,
            head_dim,
            head_dim,
            data['page_block_size'],
            layer_idx=torch.tensor(0, device=device), # a 0-D tensors
            causal=causal,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        
        output_per_head, lse_per_head = wrapper.run(query_per_head, (key_per_head, value_per_head))
        torch.cuda.synchronize()
        
        golden_outputs.append(output_per_head)
        golden_lses.append(lse_per_head)
    
    # Concatenate results
    golden_output = torch.cat(golden_outputs, dim=1)
    golden_lse = torch.cat(golden_lses, dim=1)
    
    # Assert with appropriate tolerances
    torch.testing.assert_close(output_native, golden_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_native, golden_lse, rtol=1e-3, atol=1e-3)

import pytest
import torch

@pytest.mark.parametrize("device", ["cuda"])
def test_cuda_graph_compatibility(device):
    # -------------------------  test hyper-params  ------------------------- #
    seed = 42
    gen = torch.Generator(device=device).manual_seed(seed)

    batch_size   = 4
    num_kv_heads = 4
    num_qo_heads = 32
    head_dim     = 128
    causal       = True
    dtype        = torch.bfloat16
    min_seq_len  = 100
    max_seq_len  = 500
    min_q_len    = 1
    max_q_len    = 50

    # Helper makes a new dataset **with identical shapes** as the first one
    # (CUDA Graphs require fixed shapes / addresses).
    def make_data_like(first_data=None, *, seed_offset=0):
        local_seed = seed + seed_offset
        return generate_test_data(
            batch_size, num_kv_heads, num_qo_heads, head_dim,
            device, dtype, local_seed,
            min_seq_len=min_seq_len, max_seq_len=max_seq_len,
            min_q_len=min_q_len, max_q_len=max_q_len, num_layers=1
        )

    # First dataset defines the shapes (static for capture).
    data0 = make_data_like()

    # -------------------------  baseline (eager) ------------------------- #
    batch_attention = flashinfer.BatchAttention(kv_layout="NHD", device=device)
    layer_idx_buffer = torch.tensor([0], device=device)

    batch_attention.plan(
        qo_indptr=data0["qo_indptr"],
        kv_indptr=data0["kv_indptr"],
        kv_indices=data0["all_layers_per_head_kv_indices"].int(),
        kv_len_arr=data0["sparsified_seq_lens"],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=data0["page_block_size"],
        layer_idx=layer_idx_buffer,     # 0-D tensor
        causal=causal,
        is_per_head_indices=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        add_layer_idx_by_one_after_run=False,
    )

    out_ref0, lse_ref0 = batch_attention.run(data0["query"], (data0["key"], data0["value"]))
    assert layer_idx_buffer.item() == 0, "layer_idx_buffer should remain 0 when add_layer_idx_by_one_after_run=False"
    torch.cuda.synchronize()

    # -------------------------  graph setup ------------------------- #
    batch_attention_cg = flashinfer.BatchAttention(kv_layout="NHD", device=device)
    layer_idx_buffer_cg = torch.tensor([0], device=device)

    # --- Allocate persistent input buffers (STATIC addresses for capture) ---
    q_buf = torch.empty_like(data0["query"])
    k_buf = torch.empty_like(data0["key"])
    v_buf = torch.empty_like(data0["value"])
    kv_indices_buf = torch.empty((1, num_kv_heads, max_seq_len), dtype=torch.int32, device=device)
    
    batch_attention_cg.plan(
        qo_indptr=data0["qo_indptr"],
        kv_indptr=data0["kv_indptr"],
        kv_indices=kv_indices_buf,
        kv_len_arr=data0["sparsified_seq_lens"],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=data0["page_block_size"],
        layer_idx=layer_idx_buffer_cg,   # 0-D tensor
        causal=causal,
        is_per_head_indices=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        add_layer_idx_by_one_after_run=False,
    )

    # Seed buffers with initial data0 to make the first captured run valid.
    q_buf.copy_(data0["query"])
    k_buf.copy_(data0["key"])
    v_buf.copy_(data0["value"])

    # --- Warm-up to stabilize kernels/allocator before capture ---
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            kv_indices_buf[:, :, :data0["all_layers_per_head_kv_indices"].shape[2]].copy_(data0["all_layers_per_head_kv_indices"].int())
            batch_attention_cg.run(q_buf, (k_buf, v_buf))
    torch.cuda.current_stream().wait_stream(warm_stream)

    # --- One uncaptured eager run on default stream (reduces capture-time allocs) ---
    _o_tmp, _l_tmp = batch_attention_cg.run(q_buf, (k_buf, v_buf))
    torch.cuda.synchronize()

    # --- Output placeholders that will RECEIVE copies during capture ---
    cap_output = torch.empty_like(out_ref0)
    cap_lse    = torch.empty_like(lse_ref0)

    capture_graph = torch.cuda.CUDAGraph()

    # Important: do not allocate new tensors under this context;
    # use only pre-allocated buffers to avoid allocator calls.
    with torch.cuda.graph(capture_graph):
        layer_idx_buffer_cg.zero_()  # allowed; same tensor, in-place op
        kv_indices_buf[:, :, :data0["all_layers_per_head_kv_indices"].shape[2]].copy_(data0["all_layers_per_head_kv_indices"].int())
        o, l = batch_attention_cg.run(q_buf, (k_buf, v_buf))
        cap_output.copy_(o)
        cap_lse.copy_(l)

    torch.cuda.synchronize()

    # -------------------------  validate first replay ------------------------- #
    capture_graph.replay()
    torch.cuda.synchronize()

    assert layer_idx_buffer_cg.item() == 0, "layer_idx_buffer_cg should remain 0 after graph replay"
    torch.testing.assert_close(cap_output, out_ref0, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cap_lse,    lse_ref0, rtol=1e-3, atol=1e-3)

    # -------------------------  multiple replays with new contents ------------------------- #
    # Reuse the same plan/metadata; only change Q/K/V contents to new random values.
    def _randn_like_with_gen(t: torch.Tensor, *, gen: torch.Generator) -> torch.Tensor:
        # Generate in float32 for numerical stability, then cast to t.dtype.
        out = torch.empty_like(t, dtype=torch.float32)
        out.normal_(mean=0.0, std=1.0, generator=gen)  # generator supported here
        if t.dtype in (torch.float16, torch.bfloat16):
            out = out.to(t.dtype)
        else:
            out = out.to(dtype=t.dtype)
        return out

    for i in range(1, 3):
        # Make new inputs with the exact same shapes as the captured buffers.
        gen_i = torch.Generator(device=device).manual_seed(seed + i)
        q_new = _randn_like_with_gen(q_buf, gen=gen_i)
        k_new = _randn_like_with_gen(k_buf, gen=gen_i)
        v_new = _randn_like_with_gen(v_buf, gen=gen_i)

        # Eager reference with the *same plan/metadata* as capture
        out_ref_i, lse_ref_i = batch_attention.run(q_new, (k_new, v_new))
        torch.cuda.synchronize()

        # Refill static input buffers (no reallocations, shapes identical)
        q_buf.copy_(q_new)
        k_buf.copy_(k_new)
        v_buf.copy_(v_new)

        layer_idx_buffer_cg.zero_()

        capture_graph.replay()
        torch.cuda.synchronize()

        assert layer_idx_buffer_cg.item() == 0
        torch.testing.assert_close(cap_output, out_ref_i, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cap_lse,    lse_ref_i, rtol=1e-3, atol=1e-3)



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 