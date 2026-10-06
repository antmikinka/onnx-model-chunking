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

