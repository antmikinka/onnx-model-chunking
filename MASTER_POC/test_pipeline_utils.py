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
        ExecutionPolicy,
        get_available_providers,
        get_provider_for_device,
        provider_is_npu,
        validate_npu_execution_policy,
        NPU_SEQUENTIAL_SPLIT_ERROR,
        SUPPORTED_PROVIDERS,
    )
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
    try:
        SessionManager(
            ".",
            SessionConfig(providers=["VitisAIExecutionProvider"]),
            model_info=split_info,
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
    )
    assert len(phase_manager) == 2
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

