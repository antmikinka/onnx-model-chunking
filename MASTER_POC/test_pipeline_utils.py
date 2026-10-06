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
    
    from pipeline_utils.kv_cache import KVCacheManager
    
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
        get_available_providers,
        get_provider_for_device,
        SUPPORTED_PROVIDERS,
    )
    
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
    
    # Test supported devices
    devices = get_supported_devices()
    print(f"  Supported devices: {devices}")
    
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

