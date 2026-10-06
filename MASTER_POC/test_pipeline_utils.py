#!/usr/bin/env python3
"""
Quick test script for pipeline_utils modules.
"""

import numpy as np

def test_dtypes():
    """Test dtypes module."""
    print("=" * 50)
    print("Testing dtypes module...")
    print("=" * 50)
    
    from pipeline_utils.dtypes import (
        get_numpy_dtype, 
        safe_cast, 
        ONNX_DTYPE_MAP, 
        supports_bfloat16,
        get_dtype_info
    )
    
    print(f"  ONNX_DTYPE_MAP entries: {len(ONNX_DTYPE_MAP)}")
    print(f"  bfloat16 supported: {supports_bfloat16()}")
    
    # Test type conversions
    test_types = [
        "tensor(float16)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(bfloat16)",
        "tensor(bool)",
    ]
    
    for t in test_types:
        try:
            dtype = get_numpy_dtype(t)
            print(f"  {t} -> {dtype}")
        except Exception as e:
            print(f"  {t} -> ERROR: {e}")
    
    # Test safe_cast
    arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    casted = safe_cast(arr, np.float16, warn_on_precision_loss=False)
    print(f"  safe_cast float32->float16: {arr.dtype} -> {casted.dtype}")
    
    # Test dtype info
    info = get_dtype_info(np.float16)
    print(f"  dtype info for float16: itemsize={info['itemsize']}, is_float={info['is_float']}")
    
    print("dtypes module: OK\n")
    return True


def test_kv_cache():
    """Test kv_cache module."""
    print("=" * 50)
    print("Testing kv_cache module...")
    print("=" * 50)
    
    from pipeline_utils.kv_cache import KVCacheManager, KVCacheBackend, KVCachePolicy
    from pipeline_utils.io_binding import DeviceBuffer
    
    # Create cache manager
    cache = KVCacheManager(
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        dtype=np.float16,
        batch_size=1
    )
    
    print(f"  Created: {cache}")
    print(f"  Initial sequence length: {cache.get_sequence_length()}")
    
    # Test inject into registry
    registry = {}
    cache.inject_into_registry(registry)
    print(f"  Injected {len(registry)} tensors into registry")
    
    # Simulate adding a token (update with fake present tensors)
    fake_present = np.random.randn(1, 8, 5, 128).astype(np.float16)  # 5 tokens cached
    cache.update_from_present(0, fake_present, fake_present)
    print(f"  After update layer 0: seq_len = {cache.get_sequence_length()}")
    assert not cache.static
    assert cache.current_seq_len == 5
    assert int(cache.sequence_length_scalar()) == 5
    
    # Test memory usage
    cache.reset()
    for i in range(24):
        fake_key = np.random.randn(1, 8, 10, 128).astype(np.float16)
        fake_value = np.random.randn(1, 8, 10, 128).astype(np.float16)
        cache.update_from_present(i, fake_key, fake_value)
    
    mem = cache.get_memory_usage()
    print(f"  Memory after 10 tokens cached: {mem['total_mb']:.2f} MB")
    
    # Test reset
    cache.reset()
    print(f"  After reset: seq_len = {cache.get_sequence_length()}")
    assert cache.backend is KVCacheBackend.CPU
    assert cache.policy is KVCachePolicy.DENSE_CONTIGUOUS
    assert isinstance(cache.get_past_tensors(0)[0], np.ndarray)
    assert cache.get_memory_usage()["storage"] == "numpy"
    assert cache.get_memory_usage()["static"] is False

    present = np.arange(1 * 2 * 4 * 8, dtype=np.float16).reshape(1, 2, 4, 8)

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, present_seq=8):
            self.present_seq = present_seq

        def get_inputs(self):
            return [
                _Tensor("past_key_values.0.key", [1, 2, 8, 8]),
                _Tensor("past_key_values.0.value", [1, 2, 8, 8]),
                _Tensor("past_key_values.1.key", [1, 2, 8, 8]),
                _Tensor("past_key_values.1.value", [1, 2, 8, 8]),
                _Tensor("attention_mask", [1, 8]),
                _Tensor("sequence_length", []),
            ]

        def get_outputs(self):
            return [
                _Tensor("present.0.key", [1, 2, self.present_seq, 8]),
                _Tensor("present.0.value", [1, 2, self.present_seq, 8]),
                _Tensor("present.1.key", [1, 2, self.present_seq, 8]),
                _Tensor("present.1.value", [1, 2, self.present_seq, 8]),
            ]

    class _Binding:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}

        def bind_ortvalue_input(self, name, value):
            self.inputs[name] = value

        def bind_ortvalue_output(self, name, value):
            self.outputs[name] = value

    class _EmptySession:
        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    static_cpu = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float16,
        batch_size=1,
        backend=KVCacheBackend.CPU,
        max_seq_len=8,
    )
    static_key, static_value = static_cpu.get_past_tensors(0)
    assert static_cpu.static
    assert tuple(static_key.shape) == (1, 2, 8, 8)
    assert static_cpu.current_seq_len == 0
    static_cpu.update_from_present(0, present, present.copy())
    static_cpu.update_from_present(1, present, present.copy())
    assert static_cpu.get_past_tensors(0)[0] is static_key
    assert static_cpu.get_past_tensors(0)[1] is static_value
    assert static_cpu.current_seq_len == 4
    assert np.array_equal(static_key[:, :, :4, :], present)
    token = np.full((1, 2, 1, 8), 7, dtype=np.float16)
    static_cpu.update_from_present(0, token, token.copy())
    static_cpu.update_from_present(1, token, token.copy())
    assert static_cpu.current_seq_len == 5
    assert np.array_equal(static_key[:, :, :4, :], present)
    assert np.array_equal(static_key[:, :, 4, :], token[:, :, 0, :])
    longer = np.zeros((1, 2, 6, 8), dtype=np.float16)
    longer[:, :, :5, :] = static_key[:, :, :5, :]
    longer[:, :, 5, :] = 3
    static_cpu.update_from_present(0, longer, longer.copy())
    assert static_cpu.current_seq_len == 6
    assert np.array_equal(static_key[:, :, :5, :], longer[:, :, :5, :])
    assert np.array_equal(static_key[:, :, 5, :], np.full((1, 2, 8), 3, dtype=np.float16))
    scalar = static_cpu.sequence_length_scalar()
    assert scalar.shape == ()
    assert scalar.dtype == np.int64
    assert int(scalar) == 6
    static_mask = static_cpu.attention_mask()
    assert static_mask.shape == (1, 8)
    assert int(static_mask[0, 5]) == 1
    assert int(static_mask[0, 6]) == 0
    assert static_cpu.get_memory_usage()["static"] is True
    assert static_cpu.get_memory_usage()["current_seq_len"] == 6
    static_cpu.reset()
    assert static_cpu.get_past_tensors(0)[0] is static_key
    assert static_cpu.current_seq_len == 0
    print("  static CPU KV: preallocated max_seq_len, append at current_seq_len")
    for backend in (
        KVCacheBackend.VITISAI,
        KVCacheBackend.RYZENAI,
        KVCacheBackend.ROCM,
        KVCacheBackend.CUDA,
        KVCacheBackend.PINNED_HOST,
        KVCacheBackend.HYBRID_NPU_IGPU,
    ):
        resident = KVCacheManager(
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=np.float16,
            batch_size=1,
            backend=backend,
            max_seq_len=8,
        )
        key, value = resident.get_past_tensors(0)
        assert isinstance(key, DeviceBuffer)
        assert isinstance(value, DeviceBuffer)
        assert key.ort_value is not None
        assert tuple(key.shape) == (1, 2, 8, 8)
        assert resident.get_sequence_length() == 0
        before = int(key.ort_value.data_ptr())
        resident.update_from_present(0, present, present.copy())
        resident.update_from_present(1, present, present.copy())
        assert resident.get_sequence_length() == 4
        assert int(key.ort_value.data_ptr()) == before
        written = key.to_numpy()[:, :, :4, :]
        assert np.array_equal(written, present)
        assert resident.get_past_tensors(0)[0] is key
        token = np.full((1, 2, 1, 8), 7, dtype=np.float16)
        resident.update_from_present(0, token, token.copy())
        resident.update_from_present(1, token, token.copy())
        assert resident.current_seq_len == 5
        assert int(key.ort_value.data_ptr()) == before
        assert np.array_equal(key.to_numpy()[:, :, :4, :], present)
        assert np.array_equal(key.to_numpy()[:, :, 4, :], token[:, :, 0, :])
        appended = resident.sequence_length_scalar()
        assert appended.shape == ()
        assert appended.dtype == np.int64
        assert int(appended) == 5
        again_key, again_value = resident.get_past_tensors(0)
        resident.update_from_present(0, again_key, again_value, valid_seq_len=3)
        assert resident.get_sequence_length() == 3
        assert int(again_key.ort_value.data_ptr()) == before
        ort_key, ort_value = resident.get_ort_values(0)
        device_key, device_value = resident.get_device_tensors(0)
        assert device_key is ort_key
        assert device_value is ort_value
        assert int(ort_key.data_ptr()) == before
        assert ort_value is again_value.ort_value
        mask = resident.attention_mask()
        assert mask.shape[-1] == 8
        assert int(mask[0, 2]) == 1
        assert int(mask[0, 3]) == 0
        bound_io = _Binding()
        resident.bind_cache_to_session(_Session(), bound_io, 0)
        assert bound_io.inputs["past_key_values.0.key"] is device_key
        assert bound_io.outputs["present.0.key"] is device_key
        assert bound_io.outputs["present.0.value"] is device_value
        assert int(np.asarray(bound_io.inputs["sequence_length"].numpy()).reshape(-1)[0]) == 3
        assert int(device_key.data_ptr()) == before
        resident.reset()
        assert resident.get_sequence_length() == 0
        assert int(resident.get_past_tensors(0)[0].ort_value.data_ptr()) == before
        try:
            resident.update_from_present(0, np.zeros((1, 2, 9, 8), np.float16), np.zeros((1, 2, 9, 8), np.float16))
            raise AssertionError(f"{backend.value} accepted a sequence past max_seq_len")
        except ValueError as exc:
            assert "max_seq_len" in str(exc)
        assert resident.get_memory_usage()["storage"] == "device_buffer"
        assert resident.get_memory_usage()["backend"] == backend.value
    try:
        KVCacheManager(1, 1, 1, backend="disk")
        raise AssertionError("unknown KV backend was accepted")
    except ValueError as exc:
        assert "disk" in str(exc)

    from pipeline_utils.io_binding import IOBindingManager, MemoryMode

    host = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float16,
        batch_size=1,
    )
    assert host.device_resident is False
    try:
        host.get_device_tensors(0)
        raise AssertionError("a CPU dictionary returned device tensors")
    except RuntimeError as exc:
        assert "allocate_device_cache" in str(exc)
    host.update_from_present(0, present, present.copy())
    host.update_from_present(1, present, present.copy())
    manager = IOBindingManager(
        device_type="cpu",
        memory_mode=MemoryMode.NPU_STATIC,
        max_seq_len=8,
    )
    adopted = host.allocate_device_cache(manager, max_seq_len=8)
    assert adopted is manager
    assert host.device_resident is True
    assert host.current_seq_len == 4
    assert host.get_memory_usage()["storage"] == "device_buffer"
    key_ort, value_ort = host.get_device_tensors(0)
    assert key_ort is manager.get_buffer("kv.layer0.key").ort_value
    assert value_ort is manager.get_buffer("kv.layer0.value").ort_value
    assert np.array_equal(key_ort.numpy()[:, :, :4, :], present)
    again = host.get_device_tensors(0)
    assert again[0] is key_ort and again[1] is value_ort
    host.allocate_device_cache(manager, max_seq_len=8)
    assert host.get_device_tensors(0)[0] is key_ort
    assert host.current_seq_len == 4
    binding = _Binding()
    bound = host.bind_cache_to_session(_Session(), binding, 0)
    assert bound["past_key"] == "past_key_values.0.key"
    assert bound["present_key"] == "present.0.key"
    assert binding.inputs["past_key_values.0.key"] is key_ort
    assert binding.outputs["present.0.key"] is key_ort
    assert binding.inputs["past_key_values.0.value"] is value_ort
    assert binding.outputs["present.0.value"] is value_ort
    assert "past_key_values.1.key" not in binding.inputs
    mask_ort = manager.get_buffer("kv.attention_mask").ort_value
    assert binding.inputs["attention_mask"] is mask_ort
    assert int(mask_ort.numpy()[0, 3]) == 1
    assert int(mask_ort.numpy()[0, 4]) == 0
    scalar_ort = binding.inputs["sequence_length"]
    assert int(np.asarray(scalar_ort.numpy()).reshape(-1)[0]) == 4
    assert scalar_ort.numpy().size == 1
    binding_again = _Binding()
    host.bind_cache_to_session(_Session(), binding_again, 0)
    assert binding_again.inputs["sequence_length"] is scalar_ort
    assert binding_again.inputs["attention_mask"] is mask_ort
    layer_one = _Binding()
    host.bind_cache_to_session(_Session(), layer_one, 1)
    assert layer_one.inputs["past_key_values.1.key"] is host.get_device_tensors(1)[0]
    assert layer_one.outputs["present.1.value"] is host.get_device_tensors(1)[1]
    assert "past_key_values.0.key" not in layer_one.inputs
    try:
        host.bind_cache_to_session(_Session(present_seq=1), _Binding(), 0)
        raise AssertionError("a 1-token present tensor was bound to the static cache")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    try:
        host.bind_cache_to_session(_EmptySession(), _Binding(), 0)
        raise AssertionError("a session without KV tensors was bound")
    except ValueError as exc:
        assert "layer 0" in str(exc)
    try:
        host.allocate_device_cache(manager, max_seq_len=16)
        raise AssertionError("a different max_seq_len replaced the resident pool")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    assert host.get_device_tensors(0)[0] is key_ort
    try:
        host.allocate_device_cache(object(), 8)
        raise AssertionError("a non-manager was accepted")
    except TypeError as exc:
        assert "IOBindingManager" in str(exc)
    host.reset()
    assert host.current_seq_len == 0
    assert host.get_device_tensors(0)[0] is key_ort
    print("  device KV: vitisai, ryzenai, rocm, cuda, pinned host, hybrid NPU/iGPU")
    print("  zero-copy bind: allocate_device_cache, get_device_tensors, bind_cache_to_session")

    def seq_tokens(seq, value_at=None):
        data = np.empty((1, 2, seq, 4), dtype=np.float16)
        for index in range(seq):
            data[:, :, index, :] = index if value_at is None else value_at
        return data

    def values_at(storage, count):
        host = storage.to_numpy() if hasattr(storage, "to_numpy") else storage
        return [int(host[0, 0, index, 0]) for index in range(count)]

    sliding = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        max_seq_len=8,
        policy=KVCachePolicy.SLIDING_WINDOW,
        window_size=4,
    )
    sliding_key = sliding.get_past_tensors(0)[0]
    sliding.update_from_present(0, seq_tokens(6), seq_tokens(6).copy())
    assert sliding.current_seq_len == 4
    assert sliding.seen_tokens == 6
    assert values_at(sliding_key, 4) == [2, 3, 4, 5]
    sliding.update_from_present(0, seq_tokens(1, value_at=6), seq_tokens(1, value_at=6))
    assert sliding.get_past_tensors(0)[0] is sliding_key
    assert tuple(sliding_key.shape) == (1, 2, 8, 4)
    assert sliding.current_seq_len == 4
    assert sliding.seen_tokens == 7
    assert values_at(sliding_key, 4) == [3, 4, 5, 6]
    assert int(sliding.attention_mask()[0, 3]) == 1
    assert int(sliding.attention_mask()[0, 4]) == 0

    window_only = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        policy="sliding",
        window_size=4,
    )
    assert window_only.max_seq_len == 4
    assert window_only.static
    window_key = window_only.get_past_tensors(0)[0]
    window_only.update_from_present(0, seq_tokens(6), seq_tokens(6).copy())
    assert window_only.get_past_tensors(0)[0] is window_key
    assert window_only.current_seq_len == 4

    device_window = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=8,
        policy=KVCachePolicy.SLIDING_WINDOW,
        window_size=4,
    )
    device_key = device_window.get_past_tensors(0)[0]
    device_ptr = int(device_key.ort_value.data_ptr())
    device_window.update_from_present(0, seq_tokens(6), seq_tokens(6).copy())
    device_window.update_from_present(0, seq_tokens(1, value_at=6), seq_tokens(1, value_at=6))
    assert int(device_key.ort_value.data_ptr()) == device_ptr
    assert device_window.current_seq_len == 4
    assert values_at(device_key, 4) == [3, 4, 5, 6]

    paged = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        max_seq_len=8,
        policy=KVCachePolicy.PAGED_BLOCKS,
        block_size=4,
    )
    paged_key = paged.get_past_tensors(0)[0]
    paged.update_from_present(0, seq_tokens(8), seq_tokens(8).copy())
    assert paged.current_seq_len == 8
    assert np.array_equal(paged.block_tables(0)[0], np.array([0, 1], dtype=np.int32))
    assert values_at(paged_key, 8) == list(range(8))
    paged.release_blocks(0, [0])
    assert paged.current_seq_len == 4
    assert int(paged.block_tables(0)[0, 0]) == -1
    assert int(paged.block_tables(0)[0, 1]) == 1
    released = paged.attention_mask()
    assert int(released[0, 0]) == 0
    assert int(released[0, 4]) == 1
    assert paged.get_past_tensors(0)[0] is paged_key
    paged.update_from_present(0, seq_tokens(4, value_at=9), seq_tokens(4, value_at=9))
    assert paged.current_seq_len == 8
    assert paged.seen_tokens == 12
    table = paged.block_tables(0)[0]
    assert int(table[0]) == -1
    assert int(table[1]) == 1
    assert int(table[2]) == 0
    assert values_at(paged_key, 4) == [9, 9, 9, 9]
    assert values_at(paged_key, 8)[4:] == [4, 5, 6, 7]
    try:
        paged.update_from_present(0, seq_tokens(1, value_at=3), seq_tokens(1, value_at=3))
        raise AssertionError("a full paged cache accepted another token")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    assert paged.get_past_tensors(0)[0] is paged_key

    hybrid = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        max_seq_len=16,
        policy=KVCachePolicy.HYBRID_SPARSE,
        window_size=4,
        sink_size=2,
        block_size=4,
        retrieval_blocks=1,
    )
    hybrid.pin_blocks(0, [1])
    hybrid.update_from_present(0, seq_tokens(10), seq_tokens(10).copy())
    assert hybrid.layer_length(0) == 8
    assert values_at(hybrid.get_past_tensors(0)[0], 8) == [0, 1, 4, 5, 6, 7, 8, 9]
    assert hybrid.seen_tokens == 10
    hybrid.update_from_present(0, seq_tokens(1, value_at=10), seq_tokens(1, value_at=10))
    assert values_at(hybrid.get_past_tensors(0)[0], 8) == [0, 1, 4, 5, 7, 8, 9, 10]
    assert hybrid.seen_tokens == 11
    assert hybrid.get_past_tensors(0)[0].shape[2] == 16

    mixed = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        max_seq_len=8,
        layer_policies=[KVCachePolicy.SLIDING_WINDOW, KVCachePolicy.DENSE_CONTIGUOUS],
        window_size=3,
    )
    mixed.update_from_present(0, seq_tokens(6), seq_tokens(6).copy())
    mixed.update_from_present(1, seq_tokens(6), seq_tokens(6).copy())
    assert mixed.policy_for(0) is KVCachePolicy.SLIDING_WINDOW
    assert mixed.policy_for(1) is KVCachePolicy.DENSE_CONTIGUOUS
    assert mixed.layer_length(0) == 3
    assert mixed.layer_length(1) == 6
    assert values_at(mixed.get_past_tensors(0)[0], 3) == [3, 4, 5]
    assert values_at(mixed.get_past_tensors(1)[0], 6) == [0, 1, 2, 3, 4, 5]

    try:
        KVCacheManager(1, 1, 1, policy="tape")
        raise AssertionError("unknown KV policy was accepted")
    except ValueError as exc:
        assert "tape" in str(exc)
    try:
        KVCacheManager(1, 1, 1, policy=KVCachePolicy.SLIDING_WINDOW)
        raise AssertionError("sliding window without window_size was accepted")
    except ValueError as exc:
        assert "window_size" in str(exc)
    print("  policies: sliding window, paged blocks, hybrid sparse")

    assert cache.kv_quantization is None
    assert cache.cache_dtype == np.dtype(np.float16)
    assert cache.key_scale is None and cache.value_scale is None
    try:
        cache.quantize_kv()
        raise AssertionError("quantize_kv ran without a scheme")
    except ValueError as exc:
        assert "fp16" in str(exc) and "fp4" in str(exc)
    written = np.linspace(-2, 3, 1 * 8 * 5 * 128, dtype=np.float32).astype(np.float16)
    written = written.reshape(1, 8, 5, 128)
    cache.update_from_present(0, written, written.copy())
    plain_key, plain_value = cache.dequantize_kv(0)
    assert plain_key.shape == written.shape
    assert np.array_equal(plain_key, written)
    assert np.array_equal(plain_value, written)

    fp16_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="half",
        max_seq_len=4,
    )
    fp16_src = np.linspace(-4, 4, 32, dtype=np.float32).reshape(1, 1, 4, 8)
    fp16_cache.update_from_present(0, fp16_src, fp16_src)
    fp16_cache.quantize_kv(0)
    assert fp16_cache.kv_quantization == "fp16"
    assert fp16_cache.cache_dtype == np.dtype(np.float16)
    assert fp16_cache.key_scale is None
    fp16_key, _ = fp16_cache.dequantize_kv(0)
    assert np.allclose(fp16_key, fp16_src.astype(np.float16).astype(np.float32))

    int8_cache = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="int8",
        max_seq_len=8,
    )
    int8_src = (np.linspace(-3, 5, 1 * 2 * 4 * 8, dtype=np.float32)).reshape(1, 2, 4, 8)
    int8_cache.update_from_present(0, int8_src, -int8_src)
    int8_cache.quantize_kv(0)
    assert int8_cache.cache_dtype == np.dtype(np.int8)
    assert float(int8_cache.key_scale[0]) > 0
    assert np.isnan(int8_cache.key_scale[1])
    int8_key, int8_value = int8_cache.dequantize_kv(0)
    int8_scale = float(int8_cache.key_scale[0])
    assert np.max(np.abs(int8_key - int8_src)) <= 0.5 * int8_scale + 1e-4
    assert np.max(np.abs(int8_value + int8_src)) <= 0.5 * float(int8_cache.value_scale[0]) + 1e-4
    int8_mem = int8_cache.get_memory_usage()
    assert int8_mem["quantized_layers"] == 1
    assert int8_mem["compressed_bytes"] < int8_src.nbytes * 2
    assert int8_mem["kv_quantization"] == "int8"
    int8_cache.reset()
    assert np.isnan(int8_cache.key_scale[0])
    assert int8_cache.kv_quantization == "int8"

    fresh = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="int8",
    )
    fresh.update_from_present(0, int8_src, -int8_src)
    fresh.quantize_kv(0)
    assert np.isfinite(fresh.key_scale[0])
    longer = np.concatenate([int8_src, int8_src[:, :, :1, :] + 1], axis=2)
    fresh.update_from_present(0, longer, -longer)
    assert np.isnan(fresh.key_scale[0])
    assert np.allclose(fresh.dequantize_kv(0)[0], longer)

    int4_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="int4",
        max_seq_len=4,
    )
    int4_src = np.linspace(-4, 4, 24, dtype=np.float32).reshape(1, 1, 3, 8)
    int4_cache.update_from_present(0, int4_src, int4_src)
    int4_cache.quantize_kv()
    assert int4_cache._compressed[0]["key"].dtype == np.dtype(np.uint8)
    assert int4_cache._compressed[0]["key"].shape[-1] == 4
    int4_key, _ = int4_cache.dequantize_kv(0)
    assert int4_key.shape == int4_src.shape
    assert np.max(np.abs(int4_key - int4_src)) <= 0.5 * float(int4_cache.key_scale[0]) + 1e-4

    odd_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=5,
        dtype=np.float32,
        kv_quantization="int4",
        max_seq_len=2,
    )
    odd_src = np.linspace(-2, 2, 10, dtype=np.float32).reshape(1, 1, 2, 5)
    odd_cache.update_from_present(0, odd_src, odd_src)
    odd_cache.quantize_kv(0)
    assert odd_cache._compressed[0]["key"].shape[-1] == 3
    odd_key, _ = odd_cache.dequantize_kv(0)
    assert odd_key.shape[-1] == 5
    assert np.max(np.abs(odd_key - odd_src)) <= 0.5 * float(odd_cache.key_scale[0]) + 1e-4

    fp8_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="e4m3",
        max_seq_len=2,
    )
    fp8_src = np.array([0, 1, -1, 2, 4, 16, 32, 448], dtype=np.float32).reshape(1, 1, 1, 8)
    fp8_cache.update_from_present(0, fp8_src, fp8_src)
    fp8_cache.quantize_kv(0)
    assert fp8_cache.kv_quantization == "fp8"
    assert fp8_cache.cache_dtype == np.dtype(np.uint8)
    fp8_key, _ = fp8_cache.dequantize_kv(0)
    assert np.all(np.isfinite(fp8_key))
    assert np.allclose(fp8_key, fp8_src, rtol=0, atol=0)

    fp4_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="float4",
        max_seq_len=2,
    )
    fp4_src = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32).reshape(1, 1, 1, 8)
    fp4_cache.update_from_present(0, fp4_src, -fp4_src)
    fp4_cache.quantize_kv(0)
    assert fp4_cache.kv_quantization == "fp4"
    assert fp4_cache._compressed[0]["key"].shape[-1] == 4
    fp4_key, fp4_value = fp4_cache.dequantize_kv(0)
    assert np.allclose(fp4_key, fp4_src)
    assert np.allclose(fp4_value, -fp4_src)
    fp4_span = np.linspace(-6, 6, 16, dtype=np.float32).reshape(1, 1, 2, 8)
    fp4_bounded = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="fp4",
        max_seq_len=4,
    )
    fp4_bounded.update_from_present(0, fp4_span, fp4_span)
    fp4_bounded.quantize_kv(0)
    fp4_round, _ = fp4_bounded.dequantize_kv(0)
    assert np.max(np.abs(fp4_round - fp4_span)) <= 1.0 + 1e-5

    bf16_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=np.float32,
        kv_quantization="bfloat16",
        max_seq_len=4,
    )
    assert bf16_cache.kv_quantization == "bf16"
    assert bf16_cache.cache_dtype == np.dtype(np.uint16) or str(bf16_cache.cache_dtype) == "bfloat16"
    assert bf16_cache.key_scale is None
    bf16_src = np.arange(16, dtype=np.float32).reshape(1, 1, 2, 8)
    bf16_cache.update_from_present(0, bf16_src, bf16_src)
    bf16_cache.quantize_kv(0)
    bf16_key, _ = bf16_cache.dequantize_kv(0)
    assert np.array_equal(bf16_key, bf16_src)

    try:
        KVCacheManager(1, 1, 4, kv_quantization="int3")
        raise AssertionError("unknown kv quantization was accepted")
    except ValueError as exc:
        assert "int3" in str(exc)
    try:
        KVCacheManager(1, 1, 4, dtype=np.float16, cache_dtype=np.int8)
        raise AssertionError("cache_dtype without a scheme was accepted")
    except ValueError as exc:
        assert "kv_quantization" in str(exc)

    covered = KVCacheManager(
        num_layers=3,
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float16,
        kv_quantization="int8",
        max_seq_len=4,
    )
    token = np.ones((1, 1, 2, 4), dtype=np.float16)
    for layer in range(3):
        covered.update_from_present(layer, token * (layer + 1), token)
    covered.quantize_kv()
    assert covered.get_memory_usage()["quantized_layers"] == 3
    assert np.all(np.isfinite(covered.key_scale))
    assert np.all(np.isfinite(covered.value_scale))

    device_quant = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=8,
        kv_quantization="int8",
    )
    device_present = np.linspace(-1, 1, 1 * 2 * 4 * 8, dtype=np.float32).astype(np.float16)
    device_present = device_present.reshape(1, 2, 4, 8)
    device_quant.update_from_present(0, device_present, device_present)
    device_ptr = int(device_quant.get_device_tensors(0)[0].data_ptr())
    device_quant.quantize_kv(0)
    assert int(device_quant.get_device_tensors(0)[0].data_ptr()) == device_ptr
    device_key, device_value = device_quant.dequantize_kv(0)
    assert device_key.shape[2] == device_quant.layer_length(0)
    assert device_value.shape == device_key.shape
    print("  quantization: fp16, bf16, int8, fp8, int4, fp4")

    from pipeline_utils.kv_cache import (
        CacheEntry,
        DenseKVCache,
        RecurrentStateCache,
        SparseKVCache,
        HybridKVCacheManager,
    )

    dense = DenseKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float32,
        max_seq_len=8,
    )
    assert isinstance(dense, KVCacheManager)
    assert dense.cache_type == "dense_kv"
    dense_present = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    dense.update_from_present(0, dense_present, dense_present + 1)
    dense_entry = dense.entry(0)
    assert isinstance(dense_entry, CacheEntry)
    assert dense_entry.layer_idx == 0
    assert dense_entry.cache_type == "dense_kv"
    assert dense_entry.tensors["key"] is dense.get_past_tensors(0)[0]
    assert dense_entry.metadata["seq_len"] == 3
    assert len(dense.entries()) == 2

    recurrent = RecurrentStateCache(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dtype=np.float32,
        state_dim=4,
        conv_dim=3,
        conv_kernel=4,
    )
    recurrent_state = recurrent.get_state(0)
    assert recurrent_state.shape == (1, 2, 4, 4)
    assert np.all(recurrent_state == 0)
    recurrent.recurrent_step(0, decay=0.5, update=np.ones_like(recurrent_state))
    assert recurrent.get_state(0) is recurrent_state
    assert np.allclose(recurrent_state, 1)
    recurrent.recurrent_step(0, decay=0.5, update=np.ones_like(recurrent_state))
    assert np.allclose(recurrent_state, 1.5)
    frames = [
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
        np.array([4, 5, 6], dtype=np.float32),
    ]
    for frame in frames:
        recurrent.push_conv(0, frame)
    conv_state = recurrent.get_conv_state(0)
    assert conv_state.shape == (1, 3, 3)
    assert np.allclose(conv_state[0, :, 0], frames[1])
    assert np.allclose(conv_state[0, :, 1], frames[2])
    assert np.allclose(conv_state[0, :, 2], frames[3])
    recurrent_entry = recurrent.entry(0)
    assert recurrent_entry.cache_type == "recurrent_state"
    assert recurrent_entry.tensors["state"] is recurrent_state
    assert recurrent_entry.tensors["conv_state"] is conv_state

    sparse = SparseKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float32,
        max_blocks=4,
        block_size=2,
    )
    sparse_entry = sparse.entry(0)
    assert sparse_entry.cache_type == "sparse_kv"
    assert sparse_entry.tensors["key"].shape == (1, 2, 4, 2, 4)
    sparse_block = np.arange(1 * 2 * 1 * 2 * 4, dtype=np.float32).reshape(1, 2, 1, 2, 4)
    sparse.write_blocks(0, sparse_block, sparse_block + 1, block_ids=np.array([2]))
    assert sparse.entry(0).tensors["key"] is sparse_entry.tensors["key"]
    assert np.allclose(sparse.entry(0).tensors["key"][:, :, 2], sparse_block[:, :, 0])
    assert np.all(sparse.block_indices(0)[0, :2] == -1)
    assert int(sparse.block_indices(0)[0, 2]) == 2
    sparse.set_indices(0, np.array([5, 8, 9]))
    assert list(sparse.token_indices(0)[0, :3]) == [5, 8, 9]
    assert int(sparse.token_indices(0)[0, 3]) == -1
    sparse_flat = np.ones((1, 2, 3, 4), dtype=np.float32)
    sparse.update(0, sparse_flat, sparse_flat, np.array([1, 2, 4]))
    flat_view = sparse.entry(0).tensors["key"].reshape(1, 2, -1, 4)
    assert np.allclose(flat_view[:, :, :3], sparse_flat)
    assert list(sparse.token_indices(0)[0, :3]) == [1, 2, 4]

    hybrid = HybridKVCacheManager(
        layer_types=["dense", "gated_deltanet", "sparse", "moe"],
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float32,
        max_seq_len=8,
        block_size=2,
        conv_kernel=3,
        conv_dim=4,
        num_experts=8,
        top_k=2,
        history=3,
        draft=2,
        verify=["dense", "dense"],
    )
    assert hybrid.num_layers == 4
    assert [hybrid.cache_type_for(i) for i in range(4)] == [
        "dense_kv",
        "recurrent_state",
        "sparse_kv",
        "router",
    ]
    hybrid_tokens = np.arange(1 * 2 * 2 * 4, dtype=np.float32).reshape(1, 2, 2, 4)
    hybrid.update(0, key=hybrid_tokens, value=hybrid_tokens + 1)
    assert hybrid.entry(0).layer_idx == 0
    assert hybrid.entry(0).metadata["seq_len"] == 2
    assert hybrid.entry(0).tensors["key"] is hybrid.cache_for(0).get_past_tensors(0)[0]
    recurrent_update = np.ones((1, 2, 4, 4), dtype=np.float32)
    hybrid.update(1, decay=0.0, update=recurrent_update)
    assert np.allclose(hybrid.entry(1).tensors["state"], recurrent_update)
    hybrid.update(
        2,
        key=sparse_block,
        value=sparse_block,
        block_ids=np.array([1]),
        logical_ids=np.array([7]),
    )
    assert np.allclose(hybrid.entry(2).tensors["key"][:, :, 1], sparse_block[:, :, 0])
    assert int(hybrid.entry(2).tensors["block_indices"][0, 1]) == 7
    hybrid.update(3, expert_ids=np.array([1, 4]), weights=np.array([0.7, 0.3]))
    hybrid.update(3, expert_ids=np.array([2, 5]))
    hybrid.update(3, expert_ids=np.array([3, 6]))
    hybrid.update(3, expert_ids=np.array([0, 7]))
    recent_ids, recent_weights = hybrid.cache_for(3).recent(0)
    assert recent_ids.shape == (1, 3, 2)
    assert list(recent_ids[0, 0]) == [2, 5]
    assert list(recent_ids[0, -1]) == [0, 7]
    assert hybrid.entry(3).metadata["filled"] == 3
    assert np.allclose(recent_weights[0, 0], [0.5, 0.5])
    draft_token = np.ones((1, 2, 1, 4), dtype=np.float32)
    hybrid.draft_cache.update_from_present(0, draft_token, draft_token)
    assert hybrid.draft_cache.get_sequence_length() == 1
    assert hybrid.verify_cache.cache_for(0).get_sequence_length() == 0
    assert hybrid.role_entry("draft").cache_type == "draft"
    assert hybrid.role_entry("verification").metadata["num_layers"] == 2
    verify_entries = hybrid.role_entries("verify")
    assert len(verify_entries) == 2
    assert verify_entries[0].cache_type == "dense_kv"
    assert verify_entries[0].metadata["role"] == "verify"
    hybrid_memory = hybrid.get_memory_usage()
    assert hybrid_memory["total_bytes"] > 0
    assert hybrid_memory["cache_types"] == [
        "dense_kv",
        "recurrent_state",
        "sparse_kv",
        "router",
    ]
    assert len(hybrid_memory["per_layer_bytes"]) == 4
    assert hybrid_memory["draft_bytes"] > 0
    assert hybrid_memory["verify_bytes"] > 0
    hybrid.reset()
    assert hybrid.entry(0).metadata["seq_len"] == 0
    assert np.all(hybrid.entry(1).tensors["state"] == 0)
    assert hybrid.entry(3).metadata["filled"] == 0
    assert hybrid.draft_cache.get_sequence_length() == 0
    assert hybrid.verify_cache.cache_for(0).get_sequence_length() == 0

    plain_entry = cache.entry(0)
    assert plain_entry.cache_type == "dense_kv"
    assert plain_entry.tensors["key"] is cache.get_past_tensors(0)[0]

    device_hybrid = HybridKVCacheManager(
        layer_types=["dense"],
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=4,
    )
    assert device_hybrid.cache_for(0).backend is KVCacheBackend.VITISAI
    assert device_hybrid.cache_for(0).uses_device_memory

    try:
        HybridKVCacheManager(layer_types=["tape"], num_kv_heads=1, head_dim=1)
        raise AssertionError("unknown cache type was accepted")
    except ValueError as exc:
        assert "tape" in str(exc)
    try:
        RecurrentStateCache(1, 1, 4, dtype=np.float32).update_state(
            0, np.zeros((1, 1, 2, 2), dtype=np.float32)
        )
        raise AssertionError("mismatched recurrent state was accepted")
    except ValueError as exc:
        assert "does not match" in str(exc)
    try:
        sparse.write_blocks(0, sparse_block, sparse_block, block_ids=np.array([9]))
        raise AssertionError("block id past max_blocks was accepted")
    except ValueError as exc:
        assert "max_blocks" in str(exc)
    try:
        hybrid.update(3, expert_ids=np.array([8, 1]))
        raise AssertionError("expert id outside the router was accepted")
    except ValueError as exc:
        assert "expert" in str(exc)
    print("  hybrid caches: dense, recurrent, sparse, router, draft, verify")

    import threading

    locked = KVCacheManager(num_layers=4, num_kv_heads=2, head_dim=4, dtype=np.float32)
    assert type(locked._lock) is type(threading.RLock())
    present_registry = {}
    for layer in range(4):
        token = np.full((1, 2, 2, 4), layer + 1, dtype=np.float32)
        present_registry[f"present.{layer}.key"] = token
        present_registry[f"present.{layer}.value"] = token + 1
    with locked._lock:
        extracted = locked.extract_from_registry(present_registry)
    assert extracted == 4
    assert locked.get_sequence_length() == 2
    copied = {}
    locked.inject_into_registry(copied)
    assert copied["past_key_values.3.key"].shape[2] == 2
    assert np.allclose(copied["past_key_values.3.value"], present_registry["present.3.value"])

    errors = []

    def _writer(mark):
        try:
            local = {}
            token = np.full((1, 2, 1, 4), mark, dtype=np.float32)
            for layer in range(4):
                local[f"present.{layer}.key"] = token
                local[f"present.{layer}.value"] = token
            for _ in range(20):
                locked.extract_from_registry(local)
                snapshot = {}
                locked.inject_into_registry(snapshot)
                lengths = [
                    int(snapshot[f"past_key_values.{layer}.key"].shape[2])
                    for layer in range(4)
                ]
                if len(set(lengths)) != 1 or lengths[0] not in (0, 1):
                    raise AssertionError(f"torn cache lengths {lengths}")
                if lengths[0] == 1:
                    values = [
                        float(snapshot[f"past_key_values.{layer}.key"].reshape(-1)[0])
                        for layer in range(4)
                    ]
                    if len(set(values)) != 1:
                        raise AssertionError(f"torn cache values {values}")
        except Exception as exc:
            errors.append(exc)

    def _resetter():
        try:
            for _ in range(20):
                locked.reset()
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=_writer, args=(i + 1,)) for i in range(4)]
    workers.append(threading.Thread(target=_resetter))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert errors == []
    locked.reset()
    assert locked.get_sequence_length() == 0
    print("  lock: reentrant RLock around reset, update, inject, and extract")

    print("kv_cache module: OK\n")
    return True


def test_config_loader():
    """Test config_loader module."""
    print("=" * 50)
    print("Testing config_loader module...")
    print("=" * 50)
    
    from pipeline_utils.config_loader import ModelConfig, load_model_config
    
    # Test ModelConfig dataclass
    config = ModelConfig(
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=4096
    )
    
    print(f"  Created ModelConfig: layers={config.num_layers}, kv_heads={config.num_kv_heads}")
    
    # Validate
    config.validate()
    print("  Validation passed")
    
    # Test to_dict
    d = config.to_dict()
    print(f"  to_dict keys: {list(d.keys())}")
    
    print("config_loader module: OK\n")
    return True


def test_imports():
    """Test main package imports."""
    print("=" * 50)
    print("Testing main package imports...")
    print("=" * 50)
    
    from pipeline_utils import (
        KVCacheManager,
        get_numpy_dtype,
        safe_cast,
        load_model_config,
        ModelConfig,
        ONNX_DTYPE_MAP,
        supports_bfloat16,
    )
    
    print("  All imports successful")
    print("imports: OK\n")
    return True


def test_session_manager():
    """Test session_manager module."""
    print("=" * 50)
    print("Testing session_manager module...")
    print("=" * 50)
    
    from pipeline_utils.session_manager import (
        SessionManager,
        SessionConfig,
        SessionInfo,
        ExecutionPolicy,
        ExecutionPlan,
        ExecutionStep,
        infer_component_phase,
        provider_profile_for,
        filter_session_providers,
        get_available_providers,
        get_provider_for_device,
        provider_is_npu,
        validate_npu_execution_policy,
        NPU_SEQUENTIAL_SPLIT_ERROR,
        SUPPORTED_PROVIDERS,
        ProviderCacheSpec,
    )
    import os
    import tempfile
    import warnings
    
    # Test available providers
    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    
    # Test provider for device
    auto_providers = get_provider_for_device("auto")
    print(f"  Auto providers: {auto_providers}")
    
    cpu_providers = get_provider_for_device("cpu")
    print(f"  CPU providers: {cpu_providers}")
    
    # Test SessionConfig
    config = SessionConfig(
        providers=['CPUExecutionProvider'],
        optimization_level="DISABLE",
        enable_profiling=False
    )
    
    opts = config.create_session_options()
    print(f"  SessionConfig created, optimization level: {config.optimization_level}")
    
    # Test SessionInfo dataclass
    info = SessionInfo(
        name="test_component",
        file_path="/fake/path/test.onnx",
        file_size_bytes=1024 * 1024,
        is_loaded=False
    )
    info_dict = info.to_dict()
    print(f"  SessionInfo: {info.name}, {info_dict['file_size_mb']} MB")
    
    # Test SUPPORTED_PROVIDERS constant
    print(f"  Supported provider count: {len(SUPPORTED_PROVIDERS)}")

    assert provider_is_npu("VitisAIExecutionProvider")
    assert provider_is_npu("RyzenAIExecutionProvider")
    assert not provider_is_npu("DmlExecutionProvider")
    assert not provider_is_npu("CPUExecutionProvider")
    split_layers = ["embedding", "layer_0", "layer_1", "lm_head"]
    monolithic = ["model"]

    def _expect_npu_split_rejected(providers, policy, components):
        try:
            validate_npu_execution_policy(policy, providers, components)
        except RuntimeError as exc:
            assert str(exc) == NPU_SEQUENTIAL_SPLIT_ERROR
            return
        raise AssertionError(f"{providers} + {policy} was accepted")

    _expect_npu_split_rejected(
        ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["RyzenAIExecutionProvider"],
        ExecutionPolicy.SPLIT_GPU_PERSISTENT,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["VitisAIExecutionProvider"],
        ExecutionPolicy.MONOLITHIC_ONLY,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["RyzenAIExecutionProvider"],
        "split_cpu_memory_optimized",
        monolithic,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.MONOLITHIC_ONLY,
        ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        monolithic,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ["VitisAIExecutionProvider", "DmlExecutionProvider"],
        split_layers,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.HYBRID_NPU_MOE_ROUTING,
        ["RyzenAIExecutionProvider", "ROCmExecutionProvider"],
        split_layers,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
        ["CPUExecutionProvider"],
        split_layers,
    )

    split_info = {
        "components": {
            "layer_0": {"file": "layer_0.onnx", "input_names": [], "output_names": []},
            "layer_1": {"file": "layer_1.onnx", "input_names": [], "output_names": []},
        }
    }
    if "VitisAIExecutionProvider" in get_available_providers():
        try:
            SessionManager(
                ".",
                SessionConfig(providers=["VitisAIExecutionProvider"]),
                model_info=split_info,
            )
            raise AssertionError("installed VitisAI loaded a layer split")
        except RuntimeError as exc:
            assert NPU_SEQUENTIAL_SPLIT_ERROR in str(exc)
    else:
        with warnings.catch_warnings(record=True) as dropped_warnings:
            warnings.simplefilter("always")
            portable = SessionManager(
                ".",
                SessionConfig(providers=["VitisAIExecutionProvider"]),
                model_info=split_info,
            )
        assert portable.config.providers == ["CPUExecutionProvider"]
        assert portable.provider_profile.device_type == "cpu"
        assert any(
            "VitisAIExecutionProvider" in str(item.message)
            for item in dropped_warnings
        )
    try:
        SessionManager(
            ".",
            SessionConfig(providers=["VitisAIExecutionProvider"]),
            model_info=split_info,
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
        raise AssertionError("VitisAI SessionManager loaded a layer split")
    except RuntimeError as exc:
        assert NPU_SEQUENTIAL_SPLIT_ERROR in str(exc)
    phase_manager = SessionManager(
        ".",
        SessionConfig(
            providers=["RyzenAIExecutionProvider", "DmlExecutionProvider"],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info=split_info,
        available_providers=[
            "RyzenAIExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert len(phase_manager) == 2
    assert phase_manager.config.providers[0] == "RyzenAIExecutionProvider"
    assert phase_manager.provider_profile.device_type == "ryzenai"
    cpu_manager = SessionManager(
        ".",
        SessionConfig(
            providers=["CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
        ),
        model_info=split_info,
    )
    assert "layer_0" in cpu_manager
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_npu_execution_policy(
            ExecutionPolicy.HYBRID_NPU_MOE_ROUTING,
            ["CPUExecutionProvider"],
            split_layers,
        )
    assert caught and issubclass(caught[0].category, RuntimeWarning)
    print("  NPU sequential split rejected; phase-split and MoE routing allowed")

    import onnxruntime as ort
    cpu_profile = provider_profile_for("CPUExecutionProvider")
    assert cpu_profile.device_type == "cpu"
    assert cpu_profile.optimization_level == "DISABLE"
    assert cpu_profile.enable_graph_cache is False
    assert cpu_profile.provider_options["enable_cpu_mem_arena"] is True
    assert cpu_profile.provider_options["arena_extend_strategy"] == "kSameAsRequested"
    assert cpu_profile.provider_options["intra_op_num_threads"] >= 1
    cpu_options = SessionConfig(providers=["CPUExecutionProvider"]).create_session_options()
    assert cpu_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert cpu_options.enable_cpu_mem_arena is True
    assert cpu_options.intra_op_num_threads == cpu_profile.provider_options["intra_op_num_threads"]

    cuda_profile = provider_profile_for("CUDAExecutionProvider")
    assert cuda_profile.optimization_level == "EXTENDED"
    assert cuda_profile.provider_options["do_copy_in_default_stream"] == "1"
    assert cuda_profile.provider_options["enable_cuda_graph"] == "0"
    cuda_config = SessionConfig(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        device_id=2,
        memory_limit_mb=256,
    )
    assert cuda_config.resolved_optimization_level() == "EXTENDED"
    cuda_providers = cuda_config.ort_providers()
    assert cuda_providers[0][0] == "CUDAExecutionProvider"
    assert cuda_providers[0][1]["device_id"] == "2"
    assert cuda_providers[0][1]["gpu_mem_limit"] == str(256 * 1024 * 1024)
    assert cuda_providers[0][1]["arena_extend_strategy"] == "kSameAsRequested"
    assert cuda_providers[1] == "CPUExecutionProvider"
    pinned = SessionConfig(
        providers=["CUDAExecutionProvider"],
        optimization_level="DISABLE",
    )
    assert pinned.resolved_optimization_level() == "DISABLE"

    rocm_config = SessionConfig(
        providers=["ROCmExecutionProvider", "CPUExecutionProvider"],
        device_id=1,
        memory_limit_mb=128,
        execution_policy=ExecutionPolicy.SPLIT_GPU_PERSISTENT,
    )
    rocm_providers = rocm_config.ort_providers()
    assert rocm_providers[0][1]["device_id"] == "1"
    assert rocm_providers[0][1]["do_copy_in_default_stream"] == "1"
    assert rocm_providers[0][1]["arena_extend_strategy"] == "kSameAsRequested"
    assert rocm_providers[0][1]["enable_hip_graph"] == "0"
    assert rocm_providers[0][1]["gpu_mem_limit"] == str(128 * 1024 * 1024)
    assert rocm_providers[1] == "CPUExecutionProvider"

    dml_profile = provider_profile_for("DmlExecutionProvider")
    assert dml_profile.device_type == "directml"
    assert dml_profile.enable_graph_cache is True
    assert dml_profile.provider_options["disable_metacommands"] == "0"

    npu_profile = provider_profile_for("VitisAIExecutionProvider")
    assert npu_profile.enable_graph_cache is True
    assert npu_profile.enable_compile_cache is True
    assert npu_profile.provider_options["performance_mode"] == "balanced"
    assert npu_profile.optimization_level == "DISABLE"
    with warnings.catch_warnings(record=True) as npu_warnings:
        warnings.simplefilter("always")
        npu_options = SessionConfig(
            providers=["VitisAIExecutionProvider"]
        ).create_session_options()
    assert npu_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert any("static shapes" in str(item.message) for item in npu_warnings)
    mono_info = {"components": {"model": {"file": "model.onnx"}}}
    npu_manager = SessionManager(
        ".",
        SessionConfig(providers=["VitisAIExecutionProvider", "CPUExecutionProvider"]),
        model_info=mono_info,
        available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    )
    assert npu_manager.provider_profile.device_type == "vitisai"
    assert npu_manager.provider_profile.cache_dir.endswith(".ort_cache")
    npu_providers = npu_manager.config.ort_providers(
        cache_dir=npu_manager.provider_profile.cache_dir
    )
    assert npu_providers[0][0] == "VitisAIExecutionProvider"
    assert npu_providers[0][1]["target"] == "X1"
    assert npu_providers[0][1]["cacheDir"].endswith(".ort_cache")
    assert npu_providers[0][1]["cacheKey"] == "vitisai"
    assert "performance_mode" not in npu_providers[0][1]
    assert npu_providers[1] == "CPUExecutionProvider"

    hybrid = SessionConfig(
        providers=["RyzenAIExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        npu_compilation_mode="optimized",
    )
    assert hybrid.active_profile().device_type == "ryzenai"
    assert hybrid.active_profile().provider_options["performance_mode"] == "high_performance"
    hybrid_providers = hybrid.ort_providers(cache_dir="/tmp/ort-cache")
    assert hybrid_providers[0][0] == "RyzenAIExecutionProvider"
    assert hybrid_providers[0][1]["cacheDir"] == "/tmp/ort-cache"
    assert hybrid_providers[1][0] == "DmlExecutionProvider"
    assert hybrid_providers[1][1]["disable_metacommands"] == "0"
    assert hybrid_providers[2] == "CPUExecutionProvider"

    custom = provider_profile_for("CUDAExecutionProvider", device_id=3)
    forced = SessionConfig(providers=["CPUExecutionProvider"], profile=custom)
    assert forced.active_profile().device_type == "cuda"
    assert forced.active_profile().device_id == 3

    installed = get_available_providers()
    requested = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
        "VitisAIExecutionProvider",
    ]
    kept, dropped = filter_session_providers(requested, installed)
    expected = [provider for provider in requested if provider in installed]
    assert kept == (expected or ["CPUExecutionProvider"])
    assert all(provider not in installed for provider in dropped)
    assert "CPUExecutionProvider" in kept
    tuple_kept, tuple_dropped = filter_session_providers(
        [("CUDAExecutionProvider", {"device_id": "1"}), "VitisAIExecutionProvider"],
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    assert tuple_kept == [("CUDAExecutionProvider", {"device_id": "1"})]
    assert tuple_dropped == ["VitisAIExecutionProvider"]
    empty_kept, empty_dropped = filter_session_providers(
        ["VitisAIExecutionProvider"],
        ["AzureExecutionProvider"],
    )
    assert empty_kept == ["CPUExecutionProvider"]
    assert empty_dropped == ["VitisAIExecutionProvider"]
    azure_kept, azure_dropped = filter_session_providers(
        ["AzureExecutionProvider", "VitisAIExecutionProvider"],
        ["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    assert azure_kept == ["AzureExecutionProvider"]
    assert "AzureExecutionProvider" not in SUPPORTED_PROVIDERS
    assert azure_dropped == ["VitisAIExecutionProvider"]
    print("  provider profiles: cpu arena, rocm hip, vitisai cache, directml, cuda stream")
    print(f"  available providers kept: {kept}")

    def _plan_info(*names):
        return {
            "components": {
                name: {"file": f"{name}.onnx", "input_names": [], "output_names": []}
                for name in names
            }
        }

    order_manager = SessionManager(
        ".",
        SessionConfig(
            providers=["CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
        ),
        model_info=_plan_info("embedding", "norm", "lm_head", "other"),
    )
    assert order_manager.get_component_names() == ["other", "embedding", "norm", "lm_head"]
    order_manager.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep("norm", "CPUExecutionProvider", "cpu", "norm", [], ["hidden"], True),
        ExecutionStep(
            "embedding", "CPUExecutionProvider", "cpu", "embedding",
            ["norm"], ["tokens"], True,
        ),
        ExecutionStep(
            "lm_head", "CPUExecutionProvider", "cpu", "lm_head",
            ["embedding"], ["logits"], True,
        ),
    ]))
    assert order_manager.get_component_names() == ["norm", "embedding", "lm_head"]
    assert "other" not in order_manager.get_component_names()
    assert order_manager.placement("lm_head").output_buffers == ["logits"]
    assert order_manager.resolved_provider("norm") == "CPUExecutionProvider"

    installed = [
        "VitisAIExecutionProvider",
        "DmlExecutionProvider",
        "RyzenAIExecutionProvider",
        "ROCmExecutionProvider",
        "CPUExecutionProvider",
    ]
    phase_plan_manager = SessionManager(
        ".",
        SessionConfig(
            providers=["VitisAIExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info=_plan_info("embedding", "prefill", "decode", "norm", "lm_head"),
        available_providers=installed,
    )
    phase_plan_manager.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep("embedding", "CPUExecutionProvider", "cpu", "embedding", [], ["hidden"]),
        ExecutionStep(
            "prefill", "VitisAIExecutionProvider", "npu", "prefill",
            ["embedding"], ["kv"], True,
        ),
        ExecutionStep(
            "decode", "DmlExecutionProvider", "igpu", "decode",
            ["prefill"], ["hidden"], True,
        ),
        ExecutionStep("norm", "CPUExecutionProvider", "cpu", "norm", ["decode"], ["hidden"]),
        ExecutionStep(
            "lm_head", "DmlExecutionProvider", "igpu", "lm_head",
            ["norm"], ["logits"], True,
        ),
    ]))
    assert [step.component_name for step in phase_plan_manager.execution_order()] == [
        "embedding", "prefill", "decode", "norm", "lm_head",
    ]
    assert [step.component_name for step in phase_plan_manager.sessions_for_phase("prefill")] == ["prefill"]
    assert phase_plan_manager.sessions_for_phase("decode")[0].device_type == "igpu"
    assert phase_plan_manager.resolved_provider("prefill") == "VitisAIExecutionProvider"
    assert phase_plan_manager.get_session_info("prefill").phase == "prefill"
    assert phase_plan_manager.get_session_info("prefill").resident is True
    assert phase_plan_manager.get_session_info("decode").device_type == "igpu"

    moe_manager = SessionManager(
        ".",
        SessionConfig(
            providers=["RyzenAIExecutionProvider", "ROCmExecutionProvider"],
            execution_policy=ExecutionPolicy.HYBRID_NPU_MOE_ROUTING,
        ),
        model_info=_plan_info("router", "expert_0"),
        available_providers=installed,
    )
    moe_manager._sessions["expert_0"] = object()
    moe_manager._session_info["expert_0"].is_loaded = True
    moe_manager.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep(
            "router", "RyzenAIExecutionProvider", "npu", "router",
            [], ["route"], True,
        ),
        ExecutionStep(
            "expert_0", "ROCmExecutionProvider", "igpu", "expert",
            ["router"], ["hidden"], False,
        ),
    ]))
    assert "expert_0" not in moe_manager.get_loaded_sessions()
    assert moe_manager.get_session_info("expert_0").resident is False
    assert moe_manager.get_session_info("expert_0").is_loaded is False
    assert moe_manager.get_session_info("router").resident is True
    assert moe_manager.resolved_provider("expert_0") == "ROCmExecutionProvider"

    multi_npu = SessionManager(
        ".",
        SessionConfig(
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.MONOLITHIC_ONLY,
        ),
        model_info=_plan_info("prefill", "decode"),
        available_providers=installed,
    )
    try:
        multi_npu.load_from_plan(ExecutionPlan(steps=[
            ExecutionStep("prefill", "VitisAIExecutionProvider", "npu", "prefill", [], ["kv"]),
            ExecutionStep(
                "decode", "VitisAIExecutionProvider", "npu", "decode",
                ["prefill"], ["hidden"],
            ),
        ]))
        raise AssertionError("two NPU steps were accepted in monolithic mode")
    except RuntimeError as exc:
        assert NPU_SEQUENTIAL_SPLIT_ERROR in str(exc)
    assert multi_npu.execution_plan is None

    incomplete = SessionManager(
        ".",
        SessionConfig(
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info=_plan_info("prefill"),
        available_providers=installed,
    )
    try:
        incomplete.load_from_plan(ExecutionPlan(steps=[
            ExecutionStep("prefill", "VitisAIExecutionProvider", "npu", "prefill", [], ["kv"]),
        ]))
        raise AssertionError("phase split without decode was accepted")
    except ValueError as exc:
        assert "decode" in str(exc)

    try:
        ExecutionPlan(steps=[
            ExecutionStep("prefill", "VitisAIExecutionProvider", "igpu", "prefill"),
        ]).validate()
        raise AssertionError("NPU provider was accepted as an iGPU step")
    except ValueError as exc:
        assert "npu" in str(exc)

    try:
        ExecutionPlan(steps=[
            ExecutionStep("a", "CPUExecutionProvider", "cpu", "embedding", ["b"]),
            ExecutionStep("b", "CPUExecutionProvider", "cpu", "norm", ["a"]),
        ]).validate()
        raise AssertionError("a dependency cycle was accepted")
    except ValueError as exc:
        assert "cycle" in str(exc)

    fallback_manager = SessionManager(
        ".",
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info=_plan_info("model"),
        available_providers=["CPUExecutionProvider"],
    )
    with warnings.catch_warnings(record=True) as fallback_warnings:
        warnings.simplefilter("always")
        fallback_manager.load_from_plan(ExecutionPlan(steps=[
            ExecutionStep("model", "VitisAIExecutionProvider", "npu", "prefill", [], ["kv"]),
        ]))
    assert fallback_manager.resolved_provider("model") == "CPUExecutionProvider"
    assert any("model" in str(item.message) for item in fallback_warnings)
    try:
        fallback_manager.load_from_plan(ExecutionPlan(steps=[
            ExecutionStep("missing", "CPUExecutionProvider", "cpu", "norm"),
        ]))
        raise AssertionError("unknown component was scheduled")
    except KeyError:
        pass
    print("  execution plan: dependency order, phase split, MoE residency, monolithic NPU guard")

    from pipeline_utils.io_binding import HeterogeneousIOBindingManager, IOBindingManager

    sync_info = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "hidden_size": 16,
        "dtype": "float16",
        "components": {
            "prefill": {
                "file": "prefill.onnx",
                "input_names": ["hidden", "attention_mask"],
                "output_names": ["hidden"],
                "input_shapes": {
                    "hidden": [1, "seq", 16],
                    "attention_mask": ["batch", "seq"],
                },
                "output_shapes": {"hidden": [1, "seq", 16]},
                "input_dtypes": {"hidden": "float16", "attention_mask": "int64"},
                "output_dtypes": {"hidden": "float16"},
            },
            "decode": {
                "file": "decode.onnx",
                "input_names": ["hidden"],
                "output_names": ["logits"],
                "input_shapes": {"hidden": [1, "seq", 16]},
                "output_shapes": {"logits": [1, "seq", 32]},
            },
        },
    }
    sync_manager = SessionManager(
        ".",
        SessionConfig(
            providers=[
                "VitisAIExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info=sync_info,
        available_providers=[
            "VitisAIExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    sync_manager.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep(
            "prefill", "VitisAIExecutionProvider", "npu", "prefill",
            output_buffers=["hidden"], resident=True,
        ),
        ExecutionStep(
            "decode", "DmlExecutionProvider", "igpu", "decode",
            depends_on=["prefill"], output_buffers=["logits"], resident=True,
        ),
    ]))
    prefill_meta = sync_manager.get_io_metadata("prefill")
    assert prefill_meta.device_type == "npu"
    assert prefill_meta.inputs[0].name == "hidden"
    assert prefill_meta.inputs[0].dynamic_axes == (1,)
    assert prefill_meta.inputs[1].dynamic_axes == (0, 1)
    assert prefill_meta.inputs[1].dtype == "int64"
    bridge = HeterogeneousIOBindingManager(max_seq_len=32)
    allocation = sync_manager.allocate_static_buffers(bridge, max_seq_len=32, batch_size=1)
    assert allocation.kv_pool.max_seq_len == 32
    assert allocation.kv_pool.num_layers == 2
    assert allocation.kv_pool.kv_shape == (1, 4, 32, 8)
    assert "hidden" in allocation.bridges
    assert "logits" in allocation.resident_hidden
    assert tuple(allocation.resident_hidden["logits"].shape) == (1, 32, 32)
    shared_ptr = allocation.bridges["hidden"].data_ptr
    assert shared_ptr != 0
    prefill_io = sync_manager.prepare_io_binding("prefill", bridge, static_shapes=True)
    decode_io = sync_manager.prepare_io_binding("decode", bridge, static_shapes=True)
    assert tuple(prefill_io.buffers["hidden"].shape) == (1, 32, 16)
    assert tuple(prefill_io.buffers["attention_mask"].shape) == (1, 32)
    assert prefill_io.buffers["hidden"].data_ptr == shared_ptr
    assert decode_io.buffers["hidden"].data_ptr == shared_ptr
    again = sync_manager.prepare_io_binding("decode", bridge, static_shapes=True)
    assert again.buffers["hidden"].data_ptr == shared_ptr
    try:
        sync_manager.prepare_io_binding("prefill", bridge, static_shapes=False)
        raise AssertionError("dynamic sequence was allocated")
    except ValueError as exc:
        assert "dynamic" in str(exc).lower()
    try:
        sync_manager.get_io_metadata("missing")
        raise AssertionError("missing component returned metadata")
    except KeyError:
        pass
    bare = SessionManager(
        ".",
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info={"components": {"bare": {"file": "bare.onnx", "input_names": ["x"]}}},
        available_providers=["CPUExecutionProvider"],
    )
    try:
        bare.prepare_io_binding("bare", IOBindingManager("cpu"))
        raise AssertionError("missing shape was allocated")
    except ValueError as exc:
        assert "no shape" in str(exc)
    try:
        bare.allocate_static_buffers(IOBindingManager("cpu"), max_seq_len=8)
        raise AssertionError("KV allocation ran without architecture")
    except ValueError as exc:
        assert "num_key_value_heads" in str(exc)
    print("  io sync: static KV, resident hidden, shared NPU to iGPU pointer")

    import os
    import tempfile
    from pipeline_utils.session_manager import (
        COMPILED_GRAPH_FLOOR_MB,
        PROVIDER_WORKSPACE_FLOOR_MB,
        RuntimeMemoryEstimate,
    )

    memory_info = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "hidden_size": 16,
        "dtype": "float16",
        "max_seq_len": 32,
        "batch_size": 1,
        "components": {
            "embedding": {"file": "embedding.onnx"},
            "prefill": {
                "file": "prefill.onnx",
                "weights_bytes": 4 * 1024 * 1024,
                "input_names": ["hidden", "attention_mask"],
                "output_names": ["hidden"],
                "input_shapes": {
                    "hidden": [1, "seq", 16],
                    "attention_mask": ["batch", "seq"],
                },
                "output_shapes": {"hidden": [1, "seq", 16]},
                "input_dtypes": {"hidden": "float16", "attention_mask": "int64"},
                "output_dtypes": {"hidden": "float16"},
            },
            "decode": {
                "file": "decode.onnx",
                "input_names": ["hidden"],
                "output_names": ["logits"],
                "input_shapes": {"hidden": [1, "seq", 16]},
                "output_shapes": {"logits": [1, "seq", 32]},
            },
        },
    }
    memory_manager = SessionManager(
        ".",
        SessionConfig(
            providers=[
                "VitisAIExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info=memory_info,
        available_providers=[
            "VitisAIExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    memory_manager.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep("embedding", "CPUExecutionProvider", "cpu", "embedding"),
        ExecutionStep(
            "prefill", "VitisAIExecutionProvider", "npu", "prefill",
            depends_on=["embedding"], output_buffers=["hidden"], resident=True,
        ),
        ExecutionStep(
            "decode", "DmlExecutionProvider", "igpu", "decode",
            depends_on=["prefill"], output_buffers=["logits"], resident=True,
        ),
    ]))
    prefill_memory = memory_manager.estimate_runtime_memory("prefill")
    decode_memory = memory_manager.estimate_runtime_memory("decode")
    embed_memory = memory_manager.estimate_runtime_memory("embedding")
    assert isinstance(prefill_memory, RuntimeMemoryEstimate)
    assert prefill_memory.weights_mb == 4.0
    assert prefill_memory.compiled_graph_mb == COMPILED_GRAPH_FLOOR_MB["npu"]
    assert prefill_memory.provider_workspace_mb == PROVIDER_WORKSPACE_FLOOR_MB["npu"]
    assert prefill_memory.activation_mb > 0
    assert prefill_memory.kv_cache_mb > 0
    assert prefill_memory.total_mb == round(
        prefill_memory.weights_mb
        + prefill_memory.compiled_graph_mb
        + prefill_memory.activation_mb
        + prefill_memory.kv_cache_mb
        + prefill_memory.provider_workspace_mb,
        6,
    )
    assert memory_manager.estimate_runtime_memory_mb("prefill") == prefill_memory.total_mb
    assert prefill_memory.total_mb > memory_manager.get_session_info("prefill").to_dict()["file_size_mb"]
    assert decode_memory.kv_cache_mb == prefill_memory.kv_cache_mb
    assert embed_memory.kv_cache_mb == 0.0
    assert embed_memory.compiled_graph_mb == COMPILED_GRAPH_FLOOR_MB["cpu"]
    assert decode_memory.compiled_graph_mb == COMPILED_GRAPH_FLOOR_MB["directml"]
    assert decode_memory.compiled_graph_mb < prefill_memory.compiled_graph_mb
    model_mb = memory_manager.estimate_model_runtime_memory_mb()
    summed = prefill_memory.total_mb + decode_memory.total_mb + embed_memory.total_mb
    assert abs((summed - model_mb) - prefill_memory.kv_cache_mb) < 1e-5
    usage = memory_manager.get_memory_usage()
    assert usage["total_mb"] == 0
    assert usage["total_runtime_mb"] == model_mb
    assert usage["per_session_runtime_mb"]["prefill"] == prefill_memory.total_mb
    assert memory_manager.fits_npu_monolithic(64) is False
    assert memory_manager.fits_npu_monolithic(100000) is True
    try:
        memory_manager.fits_npu_monolithic()
        raise AssertionError("missing NPU budget was accepted")
    except ValueError as exc:
        assert "npu_memory_mb" in str(exc)
    try:
        memory_manager.estimate_runtime_memory_mb("missing")
        raise AssertionError("missing component was estimated")
    except KeyError:
        pass

    shard = SessionManager(
        ".",
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info={
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_seq_len": 32,
            "components": {
                "layer_0": {"file": "layer_0.onnx", "weights_mb": 1},
                "layer_1": {"file": "layer_1.onnx", "weights_mb": 1},
            },
        },
        available_providers=["CPUExecutionProvider"],
    )
    layer_0 = shard.estimate_runtime_memory("layer_0")
    layer_1 = shard.estimate_runtime_memory("layer_1")
    assert layer_0.kv_cache_mb == layer_1.kv_cache_mb
    assert layer_0.kv_cache_mb > 0
    shard_model = shard.estimate_model_runtime_memory_mb()
    shard_non_kv = (
        (layer_0.total_mb - layer_0.kv_cache_mb)
        + (layer_1.total_mb - layer_1.kv_cache_mb)
    )
    shard_kv = round(shard_model - shard_non_kv, 6)
    assert layer_0.kv_cache_mb < shard_kv < layer_0.kv_cache_mb + layer_1.kv_cache_mb

    with tempfile.TemporaryDirectory() as model_dir:
        external = os.path.join(model_dir, "prefill.onnx.data")
        with open(external, "wb") as handle:
            handle.write(b"\0" * (2 * 1024 * 1024))
        cache_root = os.path.join(model_dir, "cache", "prefill")
        os.makedirs(cache_root)
        with open(os.path.join(cache_root, "model.xclbin"), "wb") as handle:
            handle.write(b"\0" * (3 * 1024 * 1024))
        measured = SessionManager(
            model_dir,
            SessionConfig(
                providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                cache_dir=os.path.join(model_dir, "cache"),
            ),
            model_info={
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_seq_len": 8,
                "components": {
                    "prefill": {"file": "prefill.onnx"},
                },
            },
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
        measured_memory = measured.estimate_runtime_memory("prefill")
        assert measured_memory.weights_mb == 2.0
        assert measured_memory.compiled_graph_mb == 3.0
        assert measured_memory.compiled_graph_mb < COMPILED_GRAPH_FLOOR_MB["npu"]
    print("  runtime memory: weights, graph, activations, one KV pool, workspace")

    import os
    import tempfile
    import pipeline_utils.session_manager as session_manager_module

    placed_info = {
        "components": {
            "embedding": {
                "file": "embedding.onnx",
                "provider": "VitisAIExecutionProvider",
                "provider_options": {"cacheKey": "embedding"},
            },
            "decode": {
                "file": "decode.onnx",
                "provider": "DmlExecutionProvider",
            },
            "router": {
                "file": "router.onnx",
                "provider": "CPUExecutionProvider",
                "provider_options": {
                    "intra_op_num_threads": 2,
                    "enable_cpu_mem_arena": False,
                },
            },
            "expert_0": {
                "file": "expert_0.onnx",
                "provider": "VitisAIExecutionProvider",
                "provider_options": {"cacheKey": "expert-0"},
            },
            "lm_head": {
                "file": "lm_head.onnx",
                "provider": "DmlExecutionProvider",
                "provider_options": {"device_id": "1"},
            },
            "shared": {"file": "shared.onnx"},
        },
    }
    placed_providers = [
        "VitisAIExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    placed = SessionManager(
        ".",
        SessionConfig(
            providers=list(placed_providers),
            execution_policy=ExecutionPolicy.HYBRID_NPU_MOE_ROUTING,
            npu_compilation_mode="optimized",
        ),
        model_info=placed_info,
        available_providers=list(placed_providers),
    )
    embedding_assignment = placed.provider_assignment("embedding")
    decode_assignment = placed.provider_assignment("decode")
    router_assignment = placed.provider_assignment("router")
    expert_assignment = placed.provider_assignment("expert_0")
    lm_assignment = placed.provider_assignment("lm_head")
    shared_assignment = placed.provider_assignment("shared")
    assert embedding_assignment.providers == [
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert embedding_assignment.provider_options[0]["cacheKey"] == "embedding"
    assert embedding_assignment.provider_options[0]["target"] == "X1"
    assert "performance_mode" not in embedding_assignment.provider_options[0]
    assert embedding_assignment.provider_options[1] == {}
    assert decode_assignment.providers[0] == "DmlExecutionProvider"
    assert decode_assignment.provider_options[0]["disable_metacommands"] == "0"
    assert router_assignment.providers == ["CPUExecutionProvider"]
    assert router_assignment.provider_options == [{}]
    assert expert_assignment.provider_options[0]["cacheKey"] == "expert-0"
    assert lm_assignment.provider_options[0]["device_id"] == "1"
    assert shared_assignment.providers == placed_providers
    assert placed.config.providers == placed_providers
    router_options = placed.config.create_session_options(
        profile=router_assignment.profile,
        providers=router_assignment.providers,
    )
    decode_options = placed.config.create_session_options(
        profile=decode_assignment.profile,
        providers=decode_assignment.providers,
    )
    embedding_options = placed.config.create_session_options(
        profile=embedding_assignment.profile,
        providers=embedding_assignment.providers,
    )
    assert router_options.intra_op_num_threads == 2
    assert router_options.enable_cpu_mem_arena is False
    assert router_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert decode_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    assert embedding_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert embedding_assignment.profile.provider_options["performance_mode"] == "high_performance"
    try:
        placed.provider_assignment(
            "router",
            providers=["CPUExecutionProvider", "VitisAIExecutionProvider"],
            provider_options=[{"intra_op_num_threads": 1}],
        )
        raise AssertionError("mismatched provider options were accepted")
    except ValueError as exc:
        assert "same length" in str(exc)
    with warnings.catch_warnings(record=True) as dropped_warnings:
        warnings.simplefilter("always")
        cpu_only = SessionManager(
            ".",
            SessionConfig(providers=["CPUExecutionProvider"]),
            model_info={"components": {"embedding": {
                "file": "embedding.onnx",
                "provider": "VitisAIExecutionProvider",
            }}},
            available_providers=["CPUExecutionProvider"],
        )
        dropped_assignment = cpu_only.provider_assignment("embedding")
    assert dropped_assignment.providers == ["CPUExecutionProvider"]
    assert any("embedding" in str(item.message) for item in dropped_warnings)
    placed.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep(
            "router", "CPUExecutionProvider", "cpu", "router",
            provider_options={"intra_op_num_threads": 4},
        ),
        ExecutionStep(
            "expert_0", "VitisAIExecutionProvider", "npu", "expert",
            depends_on=["router"],
            provider_options={"cacheKey": "planned-expert"},
        ),
        ExecutionStep("embedding", "VitisAIExecutionProvider", "npu", "embedding"),
        ExecutionStep("decode", "DmlExecutionProvider", "igpu", "decode"),
        ExecutionStep("lm_head", "DmlExecutionProvider", "igpu", "lm_head"),
    ]))
    assert placed.provider_assignment("expert_0").provider_options[0]["cacheKey"] == "planned-expert"
    assert placed.provider_assignment("router").profile.provider_options["intra_op_num_threads"] == 4
    assert placed.resolved_provider("lm_head") == "DmlExecutionProvider"
    assert placed.resolved_provider("embedding") == "VitisAIExecutionProvider"

    captured = {}

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None, provider_options=None, **kwargs):
            captured["path"] = path
            captured["providers"] = list(providers or [])
            captured["provider_options"] = list(provider_options or [])
            captured["intra_op_num_threads"] = sess_options.intra_op_num_threads
            captured["arena"] = sess_options.enable_cpu_mem_arena

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

        def get_providers(self):
            return list(captured["providers"])

    with tempfile.TemporaryDirectory() as model_dir:
        router_path = os.path.join(model_dir, "router.onnx")
        with open(router_path, "wb") as handle:
            handle.write(b"onnx")
        loader = SessionManager(
            model_dir,
            SessionConfig(providers=["VitisAIExecutionProvider", "CPUExecutionProvider"]),
            model_info={"components": {"router": {
                "file": "router.onnx",
                "provider": "CPUExecutionProvider",
                "provider_options": {
                    "intra_op_num_threads": 3,
                    "enable_cpu_mem_arena": False,
                },
            }}},
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
        original = session_manager_module.ort.InferenceSession
        session_manager_module.ort.InferenceSession = _FakeSession
        try:
            loaded = loader._load_single_session(
                "router",
                providers=["CPUExecutionProvider"],
                provider_options=[{
                    "intra_op_num_threads": 3,
                    "enable_cpu_mem_arena": False,
                }],
            )
        finally:
            session_manager_module.ort.InferenceSession = original
    assert loaded is not None
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]
    assert all(isinstance(name, str) for name in captured["providers"])
    assert len(captured["providers"]) == len(captured["provider_options"])
    assert captured["intra_op_num_threads"] == 3
    assert captured["arena"] is False
    print("  placement: per-component provider and provider_options")

    import statistics
    import pipeline_utils.session_manager as session_manager_module

    class _CountingSession:
        def __init__(self):
            self.calls = 0

        def run(self, output_names, inputs):
            self.calls += 1
            return {"y": np.ones((1,), dtype=np.float32)}

        def get_inputs(self):
            return [type("Input", (), {"name": "x", "shape": [1, 4]})()]

    bench_manager = SessionManager(
        ".",
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info={"components": {
            "prefill": {"file": "prefill.onnx"},
            "decode": {"file": "decode.onnx"},
        }},
        available_providers=["CPUExecutionProvider"],
    )
    prefill_session = _CountingSession()
    decode_session = _CountingSession()
    bench_manager._sessions["prefill"] = prefill_session
    bench_manager._sessions["decode"] = decode_session
    feed = {"x": np.zeros((1, 4), dtype=np.float32)}
    clock_calls = {"n": 0}

    def _clock():
        clock_calls["n"] += 1
        tick = clock_calls["n"]
        if tick == 1:
            return 0.0
        if tick == 2:
            return 0.050
        return 0.100 + (tick - 3) * 0.001

    original_clock = session_manager_module.time.perf_counter
    session_manager_module.time.perf_counter = _clock
    try:
        cold_result = bench_manager.benchmark_session("prefill", feed)
        warm_result = bench_manager.benchmark_session("prefill", feed)
        steady_result = bench_manager.benchmark_session(
            "decode",
            feed,
            exclude_first_run=False,
            warmup_iterations=2,
            measured_iterations=5,
        )
    finally:
        session_manager_module.time.perf_counter = original_clock
    assert cold_result.exclude_first_run is True
    assert cold_result.warmup_iterations == 2
    assert cold_result.measured_iterations == 5
    assert cold_result.cold_run_excluded is True
    assert cold_result.first_run_ms == 50.0
    assert len(cold_result.latencies_ms) == 5
    assert cold_result.mean_ms == statistics.mean(cold_result.latencies_ms)
    assert cold_result.first_run_ms > max(cold_result.latencies_ms)
    assert prefill_session.calls == 15
    assert warm_result.cold_run_excluded is False
    assert warm_result.first_run_ms is None
    assert prefill_session.calls == 15
    assert decode_session.calls == 7
    assert steady_result.cold_run_excluded is False
    assert steady_result.first_run_ms is None
    bench_manager.unload_session("prefill")
    assert "prefill" not in bench_manager._warmed_sessions
    bench_manager._sessions["prefill"] = prefill_session
    clock_calls["n"] = 0
    before_calls = prefill_session.calls
    session_manager_module.time.perf_counter = _clock
    try:
        reloaded = bench_manager.benchmark_session("prefill", feed)
    finally:
        session_manager_module.time.perf_counter = original_clock
    assert reloaded.cold_run_excluded is True
    assert prefill_session.calls - before_calls == 8
    before_calls = prefill_session.calls
    try:
        bench_manager.warmup_session("prefill", {"x": np.zeros((1, 3), dtype=np.float32)})
        raise AssertionError("static shape mismatch was warmed")
    except ValueError as exc:
        assert "shape" in str(exc)
    assert prefill_session.calls == before_calls
    try:
        bench_manager.warmup_session("prefill", {"other": feed["x"]})
        raise AssertionError("missing input was warmed")
    except ValueError as exc:
        assert "missing" in str(exc)
    try:
        bench_manager.warmup_session("prefill", feed, iterations=-1)
        raise AssertionError("negative warmup was accepted")
    except ValueError:
        pass
    try:
        bench_manager.benchmark_session("prefill", feed, measured_iterations=0)
        raise AssertionError("empty measurement window was accepted")
    except ValueError:
        pass
    print("  warmup: cold run excluded, then 2 warmup and 5 measured")

    assert SessionConfig().enable_provider_cache is True
    assert SessionConfig().profile_dir is None
    with tempfile.TemporaryDirectory() as cache_root:
        compile_dir = os.path.join(cache_root, "compile")
        trace_dir = os.path.join(cache_root, "traces")
        cached = SessionConfig(
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            cache_dir=compile_dir,
            enable_provider_cache=True,
            enable_profiling=True,
            profile_dir=trace_dir,
        )
        spec = cached.artifact_cache()
        assert isinstance(spec, ProviderCacheSpec)
        assert spec.enabled is True
        assert spec.cache_dir == compile_dir
        assert spec.xclbin_dir == os.path.join(compile_dir, "xclbin")
        assert spec.vitisai_dir == os.path.join(compile_dir, "vitisai")
        assert spec.profile_dir == trace_dir
        assert spec.profiling is True
        assert spec.logical_options == {"cache_dir": compile_dir, "enable_cache": True}
        assert spec.provider_options["cacheDir"] == compile_dir
        assert spec.provider_options["cacheKey"] == "vitisai"
        assert "enable_cache" not in spec.provider_options
        forwarded = cached.ort_providers()
        assert forwarded[0][1]["cacheDir"] == compile_dir
        assert forwarded[0][1]["cacheKey"] == "vitisai"
        spec.ensure_directories()
        assert os.path.isdir(spec.xclbin_dir)
        assert os.path.isdir(spec.vitisai_dir)
        profiled = cached.create_session_options()
        assert profiled.enable_profiling is True
        assert profiled.profile_file_prefix == os.path.join(trace_dir, "ort_profile")
        assert os.path.isdir(trace_dir)

        ryzen = SessionConfig(
            providers=["RyzenAIExecutionProvider"],
            cache_dir=compile_dir,
            enable_provider_cache=True,
        )
        ryzen_options = ryzen.ort_providers()[0][1]
        assert ryzen_options["cacheDir"] == compile_dir
        assert ryzen_options["cacheKey"] == "ryzenai"
        assert ryzen.artifact_cache().xclbin_dir.endswith(os.path.join("compile", "xclbin"))

        disabled = SessionConfig(
            providers=["VitisAIExecutionProvider"],
            cache_dir=compile_dir,
            enable_provider_cache=False,
            enable_profiling=True,
        )
        quiet = disabled.artifact_cache()
        assert quiet.enabled is False
        assert quiet.cache_dir is None
        assert quiet.logical_options["enable_cache"] is False
        assert quiet.provider_options == {}
        assert "cacheDir" not in disabled.ort_providers()[0][1]
        assert "cacheKey" not in disabled.ort_providers()[0][1]
        fallback_trace = os.path.join(cache_root, "unused-cache", "profiles")
        disabled_options = disabled.create_session_options()
        assert disabled_options.enable_profiling is True
        assert disabled_options.profile_file_prefix == "onnxruntime_profile_"
        assert not os.path.isdir(fallback_trace)

        model_dir = os.path.join(cache_root, "model")
        os.makedirs(model_dir)
        caching_manager = SessionManager(
            model_dir,
            SessionConfig(
                providers=["VitisAIExecutionProvider"],
                enable_profiling=True,
            ),
            model_info={"components": {"model": {"file": "model.onnx"}}},
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
        managed = caching_manager.artifact_cache()
        assert managed.cache_dir == os.path.join(model_dir, ".ort_cache")
        assert managed.logical_options["enable_cache"] is True
        assert managed.profile_dir == os.path.join(model_dir, ".ort_cache", "profiles")
        caching_manager._prepare_profile_cache()
        assert os.path.isdir(managed.xclbin_dir)
        assert os.path.isdir(managed.profile_dir)
        assignment = caching_manager.provider_assignment(
            "model",
            providers=["VitisAIExecutionProvider"],
        )
        assert assignment.provider_options[0]["cacheDir"] == managed.cache_dir

        blocked = SessionManager(
            model_dir,
            SessionConfig(
                providers=["RyzenAIExecutionProvider"],
                cache_dir=compile_dir,
                enable_provider_cache=False,
            ),
            model_info={"components": {"model": {"file": "model.onnx"}}},
            available_providers=["RyzenAIExecutionProvider", "CPUExecutionProvider"],
        )
        assert blocked.provider_profile.cache_dir is None
        assert blocked.artifact_cache().provider_options == {}
        blocked_assignment = blocked.provider_assignment(
            "model",
            providers=["RyzenAIExecutionProvider"],
        )
        assert "cacheDir" not in blocked_assignment.provider_options[0]

    assert infer_component_phase("expert_0") == "expert"
    assert infer_component_phase("speculative_draft_0") == "speculative_draft"
    assert infer_component_phase("layer_0") is None
    assert ExecutionStep(
        "router", "CPUExecutionProvider", "cpu", "moe_router"
    ).phase == "router"
    bare = SessionInfo("bare", "bare.onnx")
    assert bare.phase is None
    assert bare.execution_group is None
    assert bare.device_preference is None
    assert bare.resident_policy == "auto"
    assert bare.resolved_resident_policy() == "on_demand"

    phase_components = {
        "prefill": {
            "file": "prefill.onnx",
            "phase": "prefill",
            "device_preference": "npu",
            "execution_group": "context",
            "resident_policy": "resident",
        },
        "decode": {
            "file": "decode.onnx",
            "phase": "decode",
            "device_preference": "igpu",
            "execution_group": "token",
        },
        "router": {
            "file": "router.onnx",
            "phase": "moe_router",
            "device_preference": "npu",
            "execution_group": "moe",
        },
        "expert_0": {
            "file": "expert_0.onnx",
            "phase": "expert_execution",
            "device_preference": "igpu",
            "execution_group": "moe",
            "resident_policy": "on_demand",
        },
        "expert_1": {
            "file": "expert_1.onnx",
            "phase": "expert",
            "device_preference": "igpu",
            "execution_group": "moe",
            "resident_policy": "evict",
        },
        "kv": {
            "file": "kv.onnx",
            "phase": "kv-cache",
            "device_preference": "npu",
            "execution_group": "context",
            "resident_policy": "resident",
        },
        "draft": {
            "file": "draft.onnx",
            "phase": "speculative draft",
            "device_preference": "igpu",
            "execution_group": "speculative",
        },
        "verify": {
            "file": "verify.onnx",
            "phase": "verification",
            "device_preference": "xdna",
            "execution_group": "speculative",
        },
    }
    phase_loader = SessionManager(
        ".",
        SessionConfig(
            providers=[
                "VitisAIExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
        ),
        model_info={"components": phase_components},
        available_providers=[
            "VitisAIExecutionProvider",
            "DmlExecutionProvider",
            "RyzenAIExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert phase_loader.get_session_info("router").phase == "router"
    assert phase_loader.get_session_info("expert_0").phase == "expert"
    assert phase_loader.get_session_info("kv").phase == "kv_cache"
    assert phase_loader.get_session_info("draft").phase == "speculative_draft"
    assert phase_loader.get_session_info("verify").device_preference == "npu"
    assert phase_loader.get_session_info("draft").resolved_resident_policy() == "on_demand"
    assert phase_loader.get_session_info("kv").resolved_resident_policy() == "resident"
    assert phase_loader.get_session_info("decode").execution_group == "token"

    prefill_loaded = phase_loader.load_phase("prefill")
    assert list(prefill_loaded) == ["prefill"]
    assert prefill_loaded["prefill"].scheduled_provider == "VitisAIExecutionProvider"
    assert prefill_loaded["prefill"].device_type == "npu"
    assert prefill_loaded["prefill"].resident is True
    decode_loaded = phase_loader.load_phase("decode")
    assert decode_loaded["decode"].scheduled_provider == "DmlExecutionProvider"
    assert decode_loaded["decode"].device_type == "igpu"
    assert phase_loader.get_session_info("prefill").resident is True

    phase_loader._sessions["draft"] = object()
    phase_loader._session_info["draft"].is_loaded = True
    phase_loader._sessions["kv"] = object()
    phase_loader._session_info["kv"].is_loaded = True
    verified = phase_loader.load_phase("verification")
    assert list(verified) == ["verify"]
    assert verified["verify"].device_preference == "npu"
    assert verified["verify"].scheduled_provider == "VitisAIExecutionProvider"
    assert "draft" not in phase_loader.get_loaded_sessions()
    assert "kv" in phase_loader.get_loaded_sessions()

    experts = phase_loader.load_phase("expert_execution")
    assert list(experts) == ["expert_0", "expert_1"]
    assert experts["expert_0"].scheduled_provider == "DmlExecutionProvider"
    assert experts["expert_0"].resident is False
    assert experts["expert_1"].resident is False
    assert "expert_1" not in phase_loader.get_loaded_sessions()
    assert phase_loader.get_session_info("kv").is_loaded is True
    routed = phase_loader.load_phase("moe_router")
    assert list(routed) == ["router"]
    assert routed["router"].phase == "router"
    try:
        phase_loader.load_phase("expert", execution_group="missing")
        raise AssertionError("an empty execution group was loaded")
    except ValueError as exc:
        assert "expert" in str(exc)
    try:
        SessionManager(
            ".",
            SessionConfig(providers=["CPUExecutionProvider"]),
            model_info={"components": {"a": {"file": "a.onnx", "phase": "sideband"}}},
        )
        raise AssertionError("an unknown phase was accepted")
    except ValueError as exc:
        assert "sideband" in str(exc)

    grouped = SessionManager(
        ".",
        SessionConfig(
            providers=["CPUExecutionProvider"],
            execution_policy=ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
        ),
        model_info={"components": {
            "prefill": {"file": "prefill.onnx"},
            "decode": {"file": "decode.onnx"},
        }},
    )
    grouped.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep(
            "prefill", "CPUExecutionProvider", "cpu", "prefill",
            execution_group="context", device_preference="cpu",
            resident_policy="resident",
        ),
        ExecutionStep(
            "decode", "CPUExecutionProvider", "cpu", "decode",
            depends_on=["prefill"], resident=False,
        ),
    ]))
    assert grouped.get_session_info("prefill").execution_group == "context"
    assert grouped.get_session_info("prefill").device_preference == "cpu"
    assert grouped.get_session_info("prefill").resident_policy == "resident"
    assert grouped.get_session_info("decode").resident_policy == "on_demand"
    assert grouped.sessions_for_phase("speculative_draft") == []
    print("  phases: prefill, decode, router, expert, kv cache, draft, verification")

    defaults = SessionConfig()
    assert defaults.device_id == 0
    assert defaults.provider_options is None
    assert defaults.use_available_providers_only is True
    assert defaults.cache_dir is None
    assert defaults.warmup is False
    assert defaults.static_shapes is True
    assert defaults.memory_policy == "auto"
    assert defaults.execution_policy == "auto"
    assert defaults.resolved_execution_policy() == ExecutionPolicy.MONOLITHIC_ONLY
    assert defaults.execution_policy == ExecutionPolicy.AUTO
    assert defaults.resolved_memory_policy() == "persistent"
    npu_defaults = SessionConfig(providers=["VitisAIExecutionProvider"])
    assert npu_defaults.resolved_memory_policy() == "npu_static"
    cuda_options = SessionConfig(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": "4"}, {}],
    )
    assert cuda_options.ort_providers()[0][1]["device_id"] == "4"
    try:
        SessionConfig(
            providers=["CPUExecutionProvider"],
            provider_options=[{}, {}],
        )
        raise AssertionError("mismatched provider_options were accepted")
    except ValueError as exc:
        assert "same length" in str(exc)
    try:
        SessionConfig(memory_policy="sticky")
        raise AssertionError("unknown memory policy was accepted")
    except ValueError as exc:
        assert "sticky" in str(exc)

    surface = SessionManager(
        ".",
        SessionConfig(
            providers=["CPUExecutionProvider"],
            execution_policy="auto",
            warmup=True,
            static_shapes=True,
            memory_policy="auto",
        ),
        model_info={"components": {
            "embedding": {"file": "embedding.onnx", "device_preference": "cpu"},
            "layer_0": {"file": "layer_0.onnx"},
            "lm_head": {"file": "lm_head.onnx", "phase": "lm_head"},
        }},
    )
    assert surface.filter_available_providers() == ["CPUExecutionProvider"]
    assert surface.get_execution_order() == ["embedding", "layer_0", "lm_head"]
    assert surface.get_session_info("embedding").execution_order == 0
    assert surface.get_session_info("lm_head").execution_order == 2
    assert surface.get_session_info("embedding").device_affinity == "cpu"
    assert surface.get_session_info("lm_head").phase == "lm_head"
    try:
        surface.load_component("missing")
        raise AssertionError("unknown component was loaded")
    except KeyError:
        pass
    assert surface.load_component("embedding") is None
    plan = surface.estimate_memory_plan()
    assert plan["memory_policy"] == "persistent"
    assert plan["execution_policy"] == "monolithic_only"
    assert plan["static_shapes"] is True
    assert plan["warmup"] is True
    assert plan["order"] == ["embedding", "layer_0", "lm_head"]
    assert plan["components"]["embedding"]["estimated_runtime_memory_mb"] > 0
    assert surface.get_session_info("embedding").estimated_runtime_memory_mb > 0
    assert surface.get_session_info("embedding").warmed_up is False
    assert surface.get_session_info("embedding").compiled is False

    class _WarmSession:
        def __init__(self):
            self.calls = 0
        def get_inputs(self):
            item = type("Input", (), {"name": "x", "shape": [1, 3]})()
            return [item]
        def run(self, _outputs, feed):
            self.calls += 1
            return [feed["x"]]

    warm_session = _WarmSession()
    surface._sessions["embedding"] = warm_session
    surface._session_info["embedding"].is_loaded = True
    feed = {"x": np.zeros((1, 3), dtype=np.float32)}
    warmed = surface.warmup_component("embedding", feed)
    assert warmed.iterations == 1
    assert warm_session.calls == 1
    assert surface.get_session_info("embedding").warmed_up is True
    assert surface.get_session_info("embedding").compiled is True
    surface.unload_session("embedding")
    assert surface.get_session_info("embedding").warmed_up is False
    assert surface.get_session_info("embedding").compiled is False

    from pipeline_utils.io_binding import IOBindingManager, MemoryMode
    binding = surface.attach_io_binding_manager(IOBindingManager(device_type="cpu"))
    assert binding.memory_mode == MemoryMode.PERSISTENT
    npu_surface = SessionManager(
        ".",
        SessionConfig(
            providers=["VitisAIExecutionProvider"],
            use_available_providers_only=False,
            memory_policy="auto",
        ),
        model_info={"components": {"model": {"file": "model.onnx", "phase": "prefill"}}},
        available_providers=["CPUExecutionProvider"],
    )
    assert npu_surface.filter_available_providers() == ["VitisAIExecutionProvider"]
    assert npu_surface.config.providers == ["VitisAIExecutionProvider"]
    npu_binding = npu_surface.attach_io_binding_manager(
        IOBindingManager(device_type="cpu")
    )
    assert npu_binding.memory_mode == MemoryMode.NPU_STATIC
    assert npu_surface.estimate_memory_plan()["memory_policy"] == "npu_static"
    print("  session surface: providers, phase load, warmup, memory plan, io attach")

    print("session_manager module: OK\n")
    return True


def test_io_binding():
    """Test io_binding module."""
    print("=" * 50)
    print("Testing io_binding module...")
    print("=" * 50)
    
    from pipeline_utils.io_binding import (
        IOBindingManager,
        DeviceBuffer,
        MemoryMode,
        MemoryPlacement,
        get_supported_devices,
        get_onnx_element_type,
        NUMPY_TO_ONNX_ELEMENT_TYPE,
        VALID_ORTVALUE_DEVICES,
        ort_memory_candidates,
        create_ortvalue_from_numpy,
    )
    import pipeline_utils.io_binding as io_binding_module
    import warnings
    
    # Test MemoryMode enum
    print(f"  MemoryMode.MINIMAL: {MemoryMode.MINIMAL.value}")
    print(f"  MemoryMode.PERSISTENT: {MemoryMode.PERSISTENT.value}")
    print(f"  MemoryMode.IO_BOUND: {MemoryMode.IO_BOUND.value}")
    print(f"  Description: {MemoryMode.get_description(MemoryMode.MINIMAL)}")
    print(f"  NPU_STATIC: {MemoryMode.get_description(MemoryMode.NPU_STATIC)}")
    
    # Test supported devices
    devices = get_supported_devices()
    print(f"  Supported devices: {devices}")
    assert devices[0] == "cpu"
    npu_devices = get_supported_devices([
        "CPUExecutionProvider",
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
    ])
    assert "vitisai" in npu_devices
    assert "ryzenai" in npu_devices
    assert npu_devices.index("vitisai") > 0
    assert npu_devices.index("ryzenai") > 0
    print(f"  NPU discovery: {npu_devices}")
    
    # Test dtype mapping
    onnx_type = get_onnx_element_type(np.float16)
    print(f"  np.float16 -> ONNX type: {onnx_type}")
    
    onnx_type_int = get_onnx_element_type(np.int64)
    print(f"  np.int64 -> ONNX type: {onnx_type_int}")
    
    # Test DeviceBuffer
    buffer = DeviceBuffer(
        name="test_buffer",
        shape=(1, 8, 512, 128),
        dtype=np.float16,
        device_type="cpu",
        device_id=0
    )
    print(f"  DeviceBuffer: {buffer.name}, size={buffer.size_mb:.2f} MB")
    print(f"  Buffer matches (1,8,512,128): {buffer.matches_shape((1, 8, 512, 128))}")
    
    # Test IOBindingManager
    manager = IOBindingManager(device_type="cpu", device_id=0)
    print(f"  IOBindingManager created: {manager}")
    
    # Allocate a buffer with data
    test_data = np.random.randn(1, 4, 8).astype(np.float32)
    allocated = manager.allocate_buffer("test", test_data.shape, np.float32, test_data)
    print(f"  Allocated buffer: {allocated.name}, is_allocated={allocated.is_allocated}")
    
    # Test memory usage
    mem = manager.get_memory_usage()
    print(f"  Memory usage: {mem['total_mb']:.4f} MB, buffers={mem['buffer_count']}")
    
    # Test get_or_allocate
    same_buf = manager.get_or_allocate("test", test_data.shape, np.float32)
    print(f"  get_or_allocate same shape: OK (reused)")
    
    # Test deallocate
    manager.deallocate_all()
    print(f"  After deallocate: {len(manager)} buffers")

    # AMD OrtValue devices stay in the validation set (no eager CPU rewrite).
    for required in ("rocm", "hip", "vitisai", "ryzenai"):
        assert required in VALID_ORTVALUE_DEVICES, required
    npu_chain = [candidate.device_type for candidate in ort_memory_candidates("vitisai")]
    print(f"  vitisai memory chain: {npu_chain}")
    assert npu_chain[0] == "vitisai"
    assert "npu" in npu_chain
    assert npu_chain.index("vitisai") < npu_chain.index("hip_pinned")
    assert npu_chain.index("hip_pinned") < npu_chain.index("rocm_pinned")
    assert npu_chain.index("rocm_pinned") < npu_chain.index("cuda_pinned")
    assert npu_chain.index("cuda_pinned") < npu_chain.index("cpu")
    gpu_chain = [candidate.device_type for candidate in ort_memory_candidates("rocm")]
    assert gpu_chain[0] == "rocm"
    assert gpu_chain[1] == "gpu"

    vitis = IOBindingManager(device_type="vitisai", device_id=0)
    assert vitis.ortvalue_device == "vitisai"
    assert vitis.memory_placement == MemoryPlacement.DEVICE
    assert vitis.memory_resolved is False

    # Direct device strings fail on this wheel; pinned strings fail next;
    # the manager must still allocate rather than crash.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        npu_data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        npu_buffer = vitis.allocate_buffer("npu_hidden", npu_data.shape, np.float32, npu_data)
        empty = vitis.allocate_buffer("empty_hidden", (2, 3), np.float32)
    assert npu_buffer.is_allocated
    assert np.array_equal(npu_buffer.to_numpy(), npu_data)
    assert empty.ort_value is not None and empty.ort_value.numpy().shape == (2, 3)
    assert empty.memory_placement == vitis.memory_placement
    assert vitis.memory_resolved is True
    assert vitis.memory_placement in (MemoryPlacement.PINNED_HOST, MemoryPlacement.CPU)
    assert any(item.category is RuntimeWarning for item in caught)
    print(f"  vitisai resolved to {vitis.ortvalue_device} ({vitis.memory_placement.value})")

    # When direct AMD allocation is rejected and cuda_pinned is accepted,
    # pinned host memory wins over pageable CPU.
    calls = []
    real_from_numpy = io_binding_module.ort.OrtValue.ortvalue_from_numpy

    def _fake_from_numpy(array, device_type="cpu", device_id=0, vendor_id=-1):
        calls.append(device_type)
        if device_type in ("cuda_pinned", "cpu"):
            return real_from_numpy(array, "cpu", device_id)
        raise Exception(f"Unsupported device type: {device_type}")

    io_binding_module.ort.OrtValue.ortvalue_from_numpy = _fake_from_numpy
    try:
        pinned_manager = IOBindingManager(device_type="ryzenai")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pinned_value = pinned_manager._create_ort_value(np.ones((2,), dtype=np.float32))
        assert calls[0] == "ryzenai"
        assert calls.index("hip_pinned") < calls.index("rocm_pinned") < calls.index("cuda_pinned")
        assert calls[-1] == "cuda_pinned"
        assert pinned_manager.ortvalue_device == "cuda_pinned"
        assert pinned_manager.memory_placement == MemoryPlacement.PINNED_HOST
        assert pinned_value.numpy().shape == (2,)
        print(f"  ryzenai pinned fallback order: {calls}")
    finally:
        io_binding_module.ort.OrtValue.ortvalue_from_numpy = real_from_numpy

    cpu_value = create_ortvalue_from_numpy(np.arange(3, dtype=np.float32), device_type="cpu")
    assert cpu_value.numpy().tolist() == [0.0, 1.0, 2.0]

    from pipeline_utils.io_binding import (
        HeterogeneousIOBindingManager,
        SharedAllocationKind,
        SharedBuffer,
        OsSharedAllocation,
        allocate_os_shared_memory,
    )

    class _FakeBinding:
        def __init__(self):
            self.events = []

        def bind_output(self, name, device_type="cpu", device_id=0, element_type=None, shape=None, buffer_ptr=None):
            self.events.append(("output-ptr", name, device_type, buffer_ptr, tuple(shape)))

        def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
            self.events.append(("input-ptr", name, device_type, buffer_ptr, tuple(shape)))

        def bind_ortvalue_output(self, name, value):
            self.events.append(("output-ort", name, value, int(value.data_ptr())))

        def bind_ortvalue_input(self, name, value):
            self.events.append(("input-ort", name, value, int(value.data_ptr())))

    bridge = HeterogeneousIOBindingManager(npu_device_type="vitisai", igpu_device_type="rocm")
    assert bridge.prefill.device_type == "vitisai"
    assert bridge.decode.device_type == "rocm"
    kv = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    with warnings.catch_warnings(record=True) as shared_warnings:
        warnings.simplefilter("always")
        shared = bridge.allocate_shared_buffer("past_key", kv.shape, np.float32, kv)
    assert shared.data_ptr == int(shared.ort_value.data_ptr())
    assert np.array_equal(shared.to_numpy(), kv)
    assert shared.allocation_kind in (
        SharedAllocationKind.UNIFIED,
        SharedAllocationKind.PINNED_HOST,
        SharedAllocationKind.CPU,
    )
    if shared.allocation_kind == SharedAllocationKind.CPU:
        assert any(item.category is RuntimeWarning for item in shared_warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        private = bridge.prefill.allocate_buffer("weight", (4,), np.float32, np.ones(4, np.float32))
    assert int(private.ort_value.data_ptr()) != shared.data_ptr

    prefill_io = _FakeBinding()
    decode_io = _FakeBinding()
    bridge.bind_phase_bridge(prefill_io, decode_io, "past_key", "past_key")
    assert prefill_io.events[0][0] == "output-ptr"
    assert decode_io.events[0][0] == "input-ptr"
    assert prefill_io.events[0][3] == decode_io.events[0][3] == shared.data_ptr
    assert prefill_io.events[0][2] == "cpu"
    assert decode_io.events[0][2] == "cpu"
    try:
        bridge.bind_shared_pointer(prefill_io, "past_key", "past_key", as_output=True, device_type="vitisai")
        raise AssertionError("vitisai was accepted as the shared pointer device")
    except ValueError:
        pass

    pinned_value = shared.ort_value
    pinned_buffer = SharedBuffer(
        name="past_value",
        shape=kv.shape,
        dtype=np.dtype(np.float32),
        ort_value=pinned_value,
        allocation_kind=SharedAllocationKind.PINNED_HOST,
        data_ptr=shared.data_ptr,
        nbytes=kv.nbytes,
        npu_device_type="vitisai",
        igpu_device_type="rocm",
        bind_device_type="hip_pinned",
    )
    bridge._shared["past_value"] = pinned_buffer
    pinned_prefill = _FakeBinding()
    pinned_decode = _FakeBinding()
    bridge.bind_phase_bridge(pinned_prefill, pinned_decode, "past_value", "past_value")
    assert pinned_prefill.events[0][0] == "output-ort"
    assert pinned_decode.events[0][0] == "input-ort"
    assert pinned_prefill.events[0][2] is pinned_decode.events[0][2]
    assert pinned_prefill.events[0][3] == pinned_decode.events[0][3] == shared.data_ptr

    backing = np.zeros(kv.nbytes, dtype=np.uint8)
    os_calls = {"count": 0}

    def _fake_unified(nbytes):
        os_calls["count"] += 1
        assert nbytes == kv.nbytes
        return OsSharedAllocation(
            ptr=int(backing.ctypes.data),
            kind=SharedAllocationKind.UNIFIED,
            release=lambda: None,
            keepalive=backing,
        )

    original_os_alloc = io_binding_module.allocate_os_shared_memory
    io_binding_module.allocate_os_shared_memory = _fake_unified
    try:
        unified_bridge = HeterogeneousIOBindingManager()
        with warnings.catch_warnings(record=True) as unified_warnings:
            warnings.simplefilter("always")
            unified = unified_bridge.allocate_shared_buffer("past_key", kv.shape, np.float32, kv)
        assert os_calls["count"] == 1
        assert unified.allocation_kind == SharedAllocationKind.UNIFIED
        assert unified.data_ptr == int(backing.ctypes.data)
        assert unified.data_ptr == int(unified.ort_value.data_ptr())
        assert np.array_equal(unified.to_numpy(), kv)
        assert not any(item.category is RuntimeWarning for item in unified_warnings)
        unified_prefill = _FakeBinding()
        unified_decode = _FakeBinding()
        unified_bridge.bind_phase_bridge(unified_prefill, unified_decode, "past_key", "past_key")
        assert unified_prefill.events[0][3] == unified_decode.events[0][3] == int(backing.ctypes.data)
        assert unified_bridge.deallocate_shared_buffer("past_key") is True
        assert unified.ort_value is None
    finally:
        io_binding_module.allocate_os_shared_memory = original_os_alloc

    from pipeline_utils.io_binding import (
        StaticShapeError,
        DEFAULT_NPU_MAX_SEQ_LEN,
    )

    assert bridge.prefill.memory_mode == MemoryMode.NPU_STATIC
    assert bridge.prefill.auto_reallocate is False
    assert bridge.prefill.max_seq_len == DEFAULT_NPU_MAX_SEQ_LEN
    assert bridge.decode.auto_reallocate is True

    npu = IOBindingManager(
        device_type="cpu",
        memory_mode=MemoryMode.NPU_STATIC,
        max_seq_len=32,
        auto_reallocate=True,
    )
    assert npu.auto_reallocate is False
    pool = npu.allocate_static_kv_cache(
        "kv",
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        dtype=np.float32,
        batch_size=1,
        max_seq_len=32,
    )
    key = npu.get_buffer("kv.layer0.key")
    assert key is not None
    assert key.shape == (1, 4, 32, 8)
    assert pool.kv_shape == (1, 4, 32, 8)
    original_ptr = int(key.ort_value.data_ptr())
    prefix = np.arange(1 * 4 * 5 * 8, dtype=np.float32).reshape(1, 4, 5, 8)
    reused = npu.get_or_allocate("kv.layer0.key", prefix.shape, prefix.dtype, prefix)
    assert reused is key
    assert int(key.ort_value.data_ptr()) == original_ptr
    assert key.shape == (1, 4, 32, 8)
    assert np.array_equal(key.to_numpy()[:, :, :5, :], prefix)
    mask = npu.get_buffer("kv.attention_mask")
    assert mask is not None and mask.shape == (1, 32)
    mask_ptr = int(mask.ort_value.data_ptr())
    visible = npu.attention_mask(5, pool="kv")
    assert visible.shape == (1, 32)
    assert visible[0, :5].tolist() == [1, 1, 1, 1, 1]
    assert visible[0, 5:].tolist() == [0] * 27
    assert np.array_equal(mask.to_numpy(), visible)
    assert int(mask.ort_value.data_ptr()) == mask_ptr
    assert int(key.ort_value.data_ptr()) == original_ptr
    try:
        npu.get_or_allocate("kv.layer0.key", (1, 4, 64, 8), np.float32)
        raise AssertionError("sequence past the KV pool was reallocated")
    except StaticShapeError:
        pass
    assert int(npu.get_buffer("kv.layer0.key").ort_value.data_ptr()) == original_ptr
    npu.allocate_buffer("logits", (1, 10), np.float32)
    try:
        npu.update_buffer("logits", np.zeros((1, 11), dtype=np.float32))
        raise AssertionError("NPU_STATIC reallocated a non-KV buffer")
    except StaticShapeError:
        pass
    print(f"  NPU_STATIC pool kept ptr={original_ptr:#x} at seq 5/{pool.max_seq_len}")

    assert allocate_os_shared_memory(0) is None
    assert bridge.deallocate_shared() >= 1
    print(f"  shared bridge kind: {shared.allocation_kind.value}, ptr={shared.data_ptr:#x}")

    from pipeline_utils.io_binding import AsyncIOBindingManager

    class _WeightBinding:
        def __init__(self):
            self.inputs = []

        def bind_ortvalue_input(self, name, value):
            self.inputs.append((name, value, int(value.data_ptr())))

    with AsyncIOBindingManager(device_type="cpu", max_workers=1) as stream:
        stream.register_layer({"q_proj": ((2, 4), np.float32)})
        active_ptr = int(stream.active_buffer("q_proj").ort_value.data_ptr())
        shadow_ptr = int(stream.shadow_buffer("q_proj").ort_value.data_ptr())
        assert active_ptr != shadow_ptr
        layer0 = np.arange(8, dtype=np.float32).reshape(2, 4)
        first = stream.prefetch({"q_proj": layer0})
        assert np.array_equal(
            stream.active_buffer("q_proj").to_numpy(),
            np.zeros((2, 4), dtype=np.float32),
        )
        first_binding = _WeightBinding()
        swapped = stream.swap_and_bind(first_binding)
        assert int(swapped["q_proj"].ort_value.data_ptr()) == shadow_ptr
        assert first_binding.inputs[0][2] == shadow_ptr
        assert np.array_equal(swapped["q_proj"].to_numpy(), layer0)
        layer1 = layer0 + 10
        second = stream.prefetch({"q_proj": layer1})
        try:
            stream.prefetch({"q_proj": layer1})
            raise AssertionError("a second shadow fill was queued")
        except RuntimeError:
            pass
        second.result()
        second_binding = _WeightBinding()
        swapped_back = stream.swap_and_bind(second_binding)
        assert np.array_equal(swapped_back["q_proj"].to_numpy(), layer1)
        assert int(swapped_back["q_proj"].ort_value.data_ptr()) == active_ptr
        assert second_binding.inputs[0][2] == active_ptr
        try:
            stream.prefetch({"q_proj": np.zeros((3, 3), dtype=np.float32)})
            raise AssertionError("weight shape change was accepted")
        except StaticShapeError:
            pass
        print(f"  async shadow swap copy={stream.copy_method} ptr={active_ptr:#x}<->{shadow_ptr:#x}")

    print("io_binding module: OK\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Pipeline Utils Module Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("imports", test_imports()))
    except Exception as e:
        print(f"imports: FAILED - {e}")
        results.append(("imports", False))
    
    try:
        results.append(("dtypes", test_dtypes()))
    except Exception as e:
        print(f"dtypes: FAILED - {e}")
        results.append(("dtypes", False))
    
    try:
        results.append(("kv_cache", test_kv_cache()))
    except Exception as e:
        print(f"kv_cache: FAILED - {e}")
        results.append(("kv_cache", False))
    
    try:
        results.append(("config_loader", test_config_loader()))
    except Exception as e:
        print(f"config_loader: FAILED - {e}")
        results.append(("config_loader", False))
    
    try:
        results.append(("session_manager", test_session_manager()))
    except Exception as e:
        print(f"session_manager: FAILED - {e}")
        results.append(("session_manager", False))
    
    try:
        results.append(("io_binding", test_io_binding()))
    except Exception as e:
        print(f"io_binding: FAILED - {e}")
        results.append(("io_binding", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

