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
        ONNX_ELEMENT_TYPE_TO_NUMPY,
        NUMPY_TO_ONNX_ELEMENT_TYPE,
        numpy_dtype_from_onnx_element_type,
        QuantizedDtype,
        INT4_BLOCKWISE,
        UINT4_BLOCKWISE,
        MATMUL_NBITS_INT4,
        AWQ_INT4,
        GPTQ_INT4,
        get_quantized_dtype,
        quantized_dtype_from_matmul_nbits,
        pack_quantized,
        unpack_quantized,
        pack_weights,
        unpack_weights,
        get_recommended_dtype,
        get_dtype_preferences,
        get_recommended_quantized_dtype,
        resolve_preferred_dtype,
        PROVIDER_DTYPE_POLICY,
        KV_CACHE_DTYPE_POLICY,
        get_kv_cache_dtype,
        get_kv_cache_dtypes,
        PROVIDER_DTYPE_SUPPORT,
        is_dtype_supported_by_provider,
        supports_float8,
        get_float8_dtype,
        float8_storage_dtype,
        quantize_float8,
        dequantize_float8,
        FP8_E4M3,
        FP8_E5M2,
        supports_bfloat16,
        get_bfloat16_dtype,
        bfloat16_to_float32,
        float32_to_bfloat16,
        bfloat16_to_uint16_view,
        uint16_view_to_bfloat16,
        get_dtype_info
    )
    import pipeline_utils
    from pipeline_utils import io_binding as io_binding_module
    
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

    required_element_types = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
        11: np.float64,
        12: np.uint32,
        13: np.uint64,
        16: "bfloat16",
    }
    for code, expected in required_element_types.items():
        assert ONNX_ELEMENT_TYPE_TO_NUMPY[code] == expected
    assert ONNX_ELEMENT_TYPE_TO_NUMPY[14] == np.complex64
    assert ONNX_ELEMENT_TYPE_TO_NUMPY[15] == np.complex128
    required_reverse = {
        np.float32: 1,
        np.uint8: 2,
        np.int8: 3,
        np.uint16: 4,
        np.int16: 5,
        np.int32: 6,
        np.int64: 7,
        np.bool_: 9,
        np.float16: 10,
        np.float64: 11,
        np.uint32: 12,
        np.uint64: 13,
    }
    for np_type, code in required_reverse.items():
        assert NUMPY_TO_ONNX_ELEMENT_TYPE[np_type] == code
        assert ONNX_ELEMENT_TYPE_TO_NUMPY[code] == np_type
    assert NUMPY_TO_ONNX_ELEMENT_TYPE[np.complex64] == 14
    assert NUMPY_TO_ONNX_ELEMENT_TYPE[np.complex128] == 15
    assert io_binding_module.NUMPY_TO_ONNX_ELEMENT_TYPE is NUMPY_TO_ONNX_ELEMENT_TYPE
    assert pipeline_utils.NUMPY_TO_ONNX_ELEMENT_TYPE is NUMPY_TO_ONNX_ELEMENT_TYPE
    assert pipeline_utils.ONNX_ELEMENT_TYPE_TO_NUMPY is ONNX_ELEMENT_TYPE_TO_NUMPY
    assert pipeline_utils.numpy_dtype_from_onnx_element_type is numpy_dtype_from_onnx_element_type
    assert numpy_dtype_from_onnx_element_type(10) == np.float16
    assert numpy_dtype_from_onnx_element_type(1) == np.float32
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved_bf16 = numpy_dtype_from_onnx_element_type(16)
    if supports_bfloat16():
        assert resolved_bf16 == get_bfloat16_dtype()
    else:
        assert resolved_bf16 == np.float32
        assert any("bfloat16" in str(item.message) for item in caught)
    try:
        numpy_dtype_from_onnx_element_type(99)
        raise AssertionError("unknown element type should raise")
    except ValueError as exc:
        assert "99" in str(exc)
    print("  numeric TensorProto map round-trips and io_binding shares it")

    assert get_numpy_dtype("tensor(int4)") == np.int8
    assert get_numpy_dtype("tensor(uint4)") == np.uint8
    assert INT4_BLOCKWISE == QuantizedDtype(
        logical_dtype="int4",
        bits=4,
        signed=True,
        storage_dtype=np.int8,
        group_size=32,
        has_scale=True,
        has_zero_point=True,
        scale_dtype=np.float16,
        zero_point_dtype=np.uint8,
        layout="blocked",
    )
    assert INT4_BLOCKWISE.block_size == 32
    assert INT4_BLOCKWISE.values_per_storage_element == 2
    assert INT4_BLOCKWISE.matmul_nbits_attributes() == {"bits": 4, "block_size": 32}
    assert INT4_BLOCKWISE.packing_order()["axis"] == "k_block"
    assert INT4_BLOCKWISE.packing_order()["bit_order"] == "low_bits_first"
    assert INT4_BLOCKWISE.packing_order()["interleave"] == 1
    assert UINT4_BLOCKWISE.signed is False
    assert UINT4_BLOCKWISE.storage_dtype == np.dtype(np.uint8)
    assert get_quantized_dtype("tensor(int4)") is INT4_BLOCKWISE
    assert get_quantized_dtype("AWQ_INT4") is AWQ_INT4
    assert get_quantized_dtype("gptq") is GPTQ_INT4
    assert AWQ_INT4.storage_dtype == np.dtype(np.int32)
    assert AWQ_INT4.values_per_storage_element == 8
    assert AWQ_INT4.packing_order()["axis"] == "output"
    assert AWQ_INT4.group_size == 128
    assert AWQ_INT4.scale_shape((256, 32)) == (2, 32)
    assert GPTQ_INT4.packing_order()["axis"] == "input"
    assert GPTQ_INT4.scale_shape((32, 256)) == (32, 2)
    assert MATMUL_NBITS_INT4.matmul_nbits_attributes() == {"bits": 4, "block_size": 128}
    assert MATMUL_NBITS_INT4.signed is False
    assert MATMUL_NBITS_INT4.zero_point_dtype == np.dtype(np.uint8)
    nbits_32 = quantized_dtype_from_matmul_nbits(
        bits=4, block_size=32, signed=True, has_zero_point=True
    )
    assert nbits_32.group_size == 32
    assert nbits_32.storage_dtype == np.dtype(np.int8)
    assert nbits_32.layout == "matmul_nbits"
    no_zp = quantized_dtype_from_matmul_nbits(has_zero_point=False)
    assert no_zp.has_zero_point is False
    assert no_zp.zero_point_dtype is None
    signed_values = np.array([-1, 2, -8, 7], dtype=np.int8)
    signed_packed = pack_quantized(signed_values, INT4_BLOCKWISE)
    assert signed_packed.dtype == np.dtype(np.int8)
    assert signed_packed.tolist() == [0x2F, 0x78]
    assert unpack_quantized(signed_packed, INT4_BLOCKWISE).tolist() == [-1, 2, -8, 7]
    unsigned_values = np.array([1, 2, 15, 0], dtype=np.uint8)
    unsigned_packed = pack_quantized(unsigned_values, UINT4_BLOCKWISE)
    assert unsigned_packed.tolist() == [0x21, 0x0F]
    assert unpack_quantized(unsigned_packed, UINT4_BLOCKWISE).tolist() == [1, 2, 15, 0]
    nbits_weight = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    nbits_spec = quantized_dtype_from_matmul_nbits(bits=4, block_size=4, has_zero_point=False)
    nbits_stored = pack_weights(nbits_weight, nbits_spec)
    assert nbits_stored.shape == (1, 2, 2)
    assert nbits_stored.dtype == np.dtype(np.uint8)
    assert nbits_stored.reshape(-1).tolist() == [0x21, 0x43, 0x65, 0x87]
    assert unpack_weights(nbits_stored, nbits_spec).tolist() == nbits_weight.tolist()
    assert nbits_spec.scale_shape((1, 8)) == (1, 2)
    awq_values = np.arange(1, 9, dtype=np.int32).reshape(1, 8)
    awq_stored = pack_weights(awq_values, AWQ_INT4)
    assert awq_stored.shape == (1, 1)
    assert awq_stored.dtype == np.dtype(np.int32)
    assert int(awq_stored.view(np.uint32).reshape(-1)[0]) == sum((i + 1) << (4 * i) for i in range(8))
    assert unpack_weights(awq_stored, AWQ_INT4).tolist() == awq_values.tolist()
    gptq_stored = pack_weights(awq_values, GPTQ_INT4)
    assert gptq_stored.shape == (1, 1)
    assert int(gptq_stored.reshape(-1)[0]) == int(awq_stored.reshape(-1)[0])
    assert GPTQ_INT4.packing_axis == "input"
    assert AWQ_INT4.packing_axis == "output"
    assert pipeline_utils.QuantizedDtype is QuantizedDtype
    assert pipeline_utils.INT4_BLOCKWISE is INT4_BLOCKWISE
    assert pipeline_utils.MATMUL_NBITS_INT4 is MATMUL_NBITS_INT4
    try:
        get_quantized_dtype("awq_int8")
        raise AssertionError("awq_int8 is not a sub-byte layout")
    except ValueError as exc:
        assert "awq_int8" in str(exc)
    try:
        QuantizedDtype(
            logical_dtype="int4",
            bits=3,
            signed=True,
            storage_dtype=np.int8,
            group_size=32,
            has_scale=True,
            has_zero_point=False,
            scale_dtype=np.float16,
            zero_point_dtype=None,
            layout="blocked",
        )
        raise AssertionError("3-bit values do not pack into a byte")
    except ValueError as exc:
        assert "3-bit" in str(exc)
    try:
        pack_weights(np.ones((1, 6), dtype=np.int32), nbits_spec)
        raise AssertionError("K must be a multiple of the block")
    except ValueError as exc:
        assert "block_size" in str(exc)
    try:
        quantized_dtype_from_matmul_nbits(bits=4, block_size=3)
        raise AssertionError("block_size 3 does not fill bytes")
    except ValueError as exc:
        assert "whole bytes" in str(exc)
    print("  quantized dtype: INT4 block, MatMulNBits, AWQ, GPTQ")

    assert get_recommended_dtype("CPUExecutionProvider", "weight") == np.int8
    assert get_recommended_dtype("CPUExecutionProvider", "activation") == np.float32
    assert get_recommended_dtype("CPUExecutionProvider", "kv") == np.float16
    assert get_recommended_dtype("cpu") == np.float32
    assert get_recommended_dtype("ROCmExecutionProvider", "weight") == np.float16
    assert get_recommended_dtype("hip", "activation") == np.float16
    assert get_recommended_dtype("igpu", "kv") == np.float16
    assert get_dtype_preferences("ROCmExecutionProvider", "weight") == ("fp16", "bf16", "int8")
    assert get_dtype_preferences("ROCmExecutionProvider", "kv") == ("fp16", "fp8", "int8")
    assert get_recommended_dtype("VitisAIExecutionProvider", "weight") == np.int8
    assert get_recommended_dtype("VitisAIExecutionProvider", "activation") == np.int8
    assert get_recommended_dtype("VitisAIExecutionProvider", "kv") == np.float16
    assert get_recommended_quantized_dtype("VitisAIExecutionProvider", "weight") is INT4_BLOCKWISE
    assert get_recommended_quantized_dtype("npu", "kv") is None
    assert get_dtype_preferences("RyzenAIExecutionProvider", "weight") == ("int4", "int8")
    assert get_recommended_dtype("ryzenai", "weight") == np.int8
    assert get_recommended_dtype("DmlExecutionProvider", "weight") == np.float16
    assert get_recommended_dtype("directml", "activation") == np.float16
    assert get_recommended_dtype("dml", "kv") == np.float16
    assert get_dtype_preferences("DmlExecutionProvider", "kv") == ("fp16",)
    assert get_recommended_quantized_dtype("CPUExecutionProvider", "weight") is None
    assert set(PROVIDER_DTYPE_POLICY) == {
        "CPUExecutionProvider",
        "ROCmExecutionProvider",
        "VitisAIExecutionProvider",
        "DmlExecutionProvider",
    }
    with warnings.catch_warnings(record=True) as policy_warnings:
        warnings.simplefilter("always")
        fallen_back = resolve_preferred_dtype(("bf16", "fp32"))
    if supports_bfloat16():
        assert fallen_back == np.dtype(get_bfloat16_dtype())
    else:
        assert fallen_back == np.float32
        assert any("bf16" in str(item.message) for item in policy_warnings)
    assert pipeline_utils.get_recommended_dtype is get_recommended_dtype
    assert pipeline_utils.PROVIDER_DTYPE_POLICY is PROVIDER_DTYPE_POLICY
    try:
        get_recommended_dtype("CUDAExecutionProvider", "weight")
        raise AssertionError("CUDA has no dtype policy row")
    except ValueError as exc:
        assert "CUDAExecutionProvider" in str(exc)
    try:
        get_recommended_dtype("CPUExecutionProvider", "logits")
        raise AssertionError("logits is not a tensor role")
    except ValueError as exc:
        assert "logits" in str(exc)
    print("  provider dtype policy: CPU, ROCm, VitisAI, DirectML")

    assert supports_float8() is False
    assert FP8_E4M3 is None
    assert FP8_E5M2 is None
    assert get_float8_dtype("e4m3fn") is None
    assert get_float8_dtype("e5m2") is None
    assert float8_storage_dtype("e4m3fn") == np.dtype(np.uint8)
    assert float8_storage_dtype("e5m2") == np.dtype(np.float16)
    with warnings.catch_warnings(record=True) as float8_warnings:
        warnings.simplefilter("always")
        assert get_numpy_dtype("tensor(float8e4m3fn)") == np.float16
        assert get_numpy_dtype("float8e5m2") == np.float16
        assert numpy_dtype_from_onnx_element_type(17) == np.float16
        assert numpy_dtype_from_onnx_element_type(19) == np.float16
    assert any("float8e4m3fn" in str(item.message) for item in float8_warnings)
    fp8_values = np.array([0, 1, -1, 2, 4, 16, 32, 448], dtype=np.float32)
    fp8_stored = quantize_float8(fp8_values, scale=1.0, kind="e4m3fn")
    assert fp8_stored.dtype == np.dtype(np.uint8)
    fp8_restored = dequantize_float8(fp8_stored, scale=1.0)
    assert np.allclose(fp8_restored, fp8_values, rtol=0, atol=0)
    import pipeline_utils.dtypes as dtypes_mod
    saved_fp8 = (dtypes_mod.FP8_E4M3, dtypes_mod.FP8_E5M2)
    try:
        dtypes_mod.FP8_E4M3 = np.float16
        dtypes_mod.FP8_E5M2 = np.float16
        assert dtypes_mod.supports_float8() is True
        assert dtypes_mod.get_float8_dtype("e5m2") is np.float16
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always")
            assert get_numpy_dtype("tensor(float8e4m3fn)") == np.float16
        assert not any("falling back to float16" in str(item.message) for item in native_warnings)
        native_stored = quantize_float8(np.array([1.0, -2.0], dtype=np.float32), scale=1.0)
        assert native_stored.dtype == np.dtype(np.float16)
        assert np.allclose(dequantize_float8(native_stored), np.array([1.0, -2.0], dtype=np.float32))
    finally:
        dtypes_mod.FP8_E4M3, dtypes_mod.FP8_E5M2 = saved_fp8
    assert supports_float8() is False
    assert pipeline_utils.supports_float8 is supports_float8
    assert pipeline_utils.quantize_float8 is quantize_float8
    try:
        quantize_float8(np.array([1.0]), scale=0.0)
        raise AssertionError("a zero float8 scale should be rejected")
    except ValueError as exc:
        assert "scale" in str(exc)
    print("  float8: ml_dtypes when present, E4M3 codes otherwise")

    assert supports_bfloat16() is False
    bf16_values = np.array([0.0, -0.0, 1.0, -1.0, 2.0, np.inf, -np.inf], dtype=np.float32)
    bf16_stored = float32_to_bfloat16(bf16_values)
    assert bf16_stored.dtype == np.dtype(np.uint16)
    assert bf16_stored.tolist() == [0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x7F80, 0xFF80]
    bf16_view = bfloat16_to_uint16_view(bf16_stored)
    assert bf16_view.dtype == np.dtype(np.uint16)
    assert np.shares_memory(bf16_view, bf16_stored)
    assert uint16_view_to_bfloat16(bf16_view).dtype == np.dtype(np.uint16)
    assert np.array_equal(uint16_view_to_bfloat16(bf16_view), bf16_stored)
    bf16_restored = bfloat16_to_float32(bf16_stored)
    assert bf16_restored.dtype == np.dtype(np.float32)
    assert np.array_equal(bf16_restored[:5].view(np.uint32), bf16_values[:5].view(np.uint32))
    assert np.isposinf(bf16_restored[5]) and np.isneginf(bf16_restored[6])
    tie_even = np.array([0x3F808000], dtype=np.uint32).view(np.float32)
    tie_odd = np.array([0x3F818000], dtype=np.uint32).view(np.float32)
    above_tie = np.array([0x3F808001], dtype=np.uint32).view(np.float32)
    assert int(float32_to_bfloat16(tie_even)[0]) == 0x3F80
    assert int(float32_to_bfloat16(tie_odd)[0]) == 0x3F82
    assert int(float32_to_bfloat16(above_tie)[0]) == 0x3F81
    nan_value = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
    nan_stored = float32_to_bfloat16(nan_value)
    assert int(nan_stored[0]) != 0x7F80
    assert np.isnan(bfloat16_to_float32(nan_stored)[0])
    exact = np.arange(16, dtype=np.float32).reshape(2, 8)
    assert float32_to_bfloat16(exact).shape == exact.shape
    assert np.array_equal(bfloat16_to_float32(float32_to_bfloat16(exact)), exact)
    sparse = np.arange(8, dtype=np.float32)[::2]
    assert np.array_equal(
        bfloat16_to_float32(float32_to_bfloat16(sparse)),
        np.array([0, 2, 4, 6], dtype=np.float32),
    )
    empty_bf16 = float32_to_bfloat16(np.array([], dtype=np.float32))
    assert empty_bf16.dtype == np.dtype(np.uint16) and empty_bf16.shape == (0,)
    assert bfloat16_to_float32(empty_bf16).shape == (0,)
    try:
        bfloat16_to_float32(np.array([1.0], dtype=np.float32))
        raise AssertionError("float32 is not bfloat16 storage")
    except ValueError as exc:
        assert "bfloat16" in str(exc)
    try:
        uint16_view_to_bfloat16(np.array([1.0], dtype=np.float32))
        raise AssertionError("float32 is not a uint16 bfloat16 view")
    except ValueError as exc:
        assert "uint16" in str(exc)
    saved_bf16 = dtypes_mod._BFLOAT16_DTYPE
    try:
        dtypes_mod._BFLOAT16_DTYPE = np.dtype(np.int16)
        assert dtypes_mod.supports_bfloat16() is True
        with warnings.catch_warnings(record=True) as native_bf16_warnings:
            warnings.simplefilter("always")
            assert get_numpy_dtype("tensor(bfloat16)") == np.dtype(np.int16)
        assert not any("ml_dtypes not available" in str(item.message) for item in native_bf16_warnings)
        native_bf16 = float32_to_bfloat16(np.array([1.0, -2.0], dtype=np.float32))
        assert native_bf16.dtype == np.dtype(np.int16)
        native_bits = bfloat16_to_uint16_view(native_bf16)
        assert native_bits.tolist() == [0x3F80, 0xC000]
        assert np.shares_memory(native_bits, native_bf16)
        assert np.array_equal(
            bfloat16_to_float32(native_bf16),
            np.array([1.0, -2.0], dtype=np.float32),
        )
        viewed = uint16_view_to_bfloat16(native_bits)
        assert viewed.dtype == np.dtype(np.int16)
        assert np.shares_memory(viewed, native_bits)
    finally:
        dtypes_mod._BFLOAT16_DTYPE = saved_bf16
    assert supports_bfloat16() is False
    assert pipeline_utils.float32_to_bfloat16 is float32_to_bfloat16
    assert pipeline_utils.bfloat16_to_float32 is bfloat16_to_float32
    assert pipeline_utils.bfloat16_to_uint16_view is bfloat16_to_uint16_view
    assert pipeline_utils.uint16_view_to_bfloat16 is uint16_view_to_bfloat16
    print("  bfloat16: uint16 transport, native view when a 16-bit dtype is present")

    overflow_i = np.array([100000, -100000, 3], dtype=np.int64)
    with warnings.catch_warnings(record=True) as cast_warnings:
        warnings.simplefilter("always")
        wrapped = safe_cast(overflow_i, np.int16)
    assert wrapped.dtype == np.dtype(np.int16)
    assert wrapped.tolist() == overflow_i.astype(np.int16).tolist()
    assert wrapped.tolist() != [32767, -32768, 3]
    assert any("may lose precision" in str(item.message) for item in cast_warnings)
    try:
        safe_cast(np.array([100000], dtype=np.int64), np.int16, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("100000 does not fit in int16")
    except ValueError as exc:
        assert "100000" in str(exc)
        assert "int16" in str(exc)
        assert "out of range" in str(exc)
    with warnings.catch_warnings(record=True) as sat_warnings:
        warnings.simplefilter("always")
        saturated = safe_cast(overflow_i, np.int16, saturate=True, warn_on_precision_loss=False)
    assert saturated.tolist() == [32767, -32768, 3]
    assert any("Saturating" in str(item.message) for item in sat_warnings)
    inside = safe_cast(
        np.array([32767, -32768], dtype=np.int64),
        np.int16,
        check_range=True,
        warn_on_precision_loss=False,
    )
    assert inside.tolist() == [32767, -32768]
    unsigned = safe_cast(
        np.array([-1, 300], dtype=np.int16),
        np.uint8,
        saturate=True,
        warn_on_precision_loss=False,
    )
    assert unsigned.tolist() == [0, 255]
    wide_u = np.array([1 << 40], dtype=np.uint64)
    try:
        safe_cast(wide_u, np.int16, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("a uint64 above 32767 does not fit in int16")
    except ValueError as exc:
        assert "int16" in str(exc)
    assert safe_cast(wide_u, np.int16, saturate=True, warn_on_precision_loss=False).tolist() == [32767]
    assert safe_cast(np.array([4], dtype=np.uint64), np.int16, check_range=True).tolist() == [4]
    empty_cast = safe_cast(
        np.array([], dtype=np.int64),
        np.int16,
        check_range=True,
        warn_on_precision_loss=False,
    )
    assert empty_cast.shape == (0,) and empty_cast.dtype == np.dtype(np.int16)
    big_f = np.array([1e10, -1e10, 1.0], dtype=np.float32)
    try:
        safe_cast(big_f, np.float16, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("1e10 does not fit in float16")
    except ValueError as exc:
        assert "float16" in str(exc)
        assert "out of range" in str(exc)
    sat_f = safe_cast(big_f, np.float16, saturate=True, warn_on_precision_loss=False)
    assert sat_f.dtype == np.dtype(np.float16)
    assert sat_f[0] == np.finfo(np.float16).max
    assert sat_f[1] == -np.finfo(np.float16).max
    assert sat_f[2] == np.float16(1.0)
    tiny_f = np.array([1e-20, -1e-20, 0.0], dtype=np.float32)
    try:
        safe_cast(tiny_f, np.float16, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("1e-20 underflows float16")
    except ValueError as exc:
        assert "float16" in str(exc)
    sat_tiny = safe_cast(tiny_f, np.float16, saturate=True, warn_on_precision_loss=False)
    sub = np.finfo(np.float16).smallest_subnormal
    assert sat_tiny[0] == sub
    assert sat_tiny[1] == -sub
    assert sat_tiny[2] == np.float16(0.0)
    kept_sub = safe_cast(
        np.array([sub], dtype=np.float32),
        np.float16,
        check_range=True,
        warn_on_precision_loss=False,
    )
    assert kept_sub[0] == np.float16(sub)
    in_range_f = safe_cast(
        np.array([1.5], dtype=np.float32),
        np.float16,
        check_range=True,
        warn_on_precision_loss=False,
    )
    assert in_range_f.dtype == np.dtype(np.float16)
    assert in_range_f[0] == np.float16(1.5)
    try:
        safe_cast(np.array([np.nan], dtype=np.float32), np.int16, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("NaN does not fit in int16")
    except ValueError as exc:
        assert "int16" in str(exc)
    nan_sat = safe_cast(
        np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
        np.int16,
        saturate=True,
        warn_on_precision_loss=False,
    )
    assert nan_sat.tolist() == [0, 32767, -32768]
    try:
        safe_cast(np.array([1e20], dtype=np.float64), np.int64, check_range=True, warn_on_precision_loss=False)
        raise AssertionError("1e20 does not fit in int64")
    except ValueError as exc:
        assert "int64" in str(exc)
    int64_sat = safe_cast(
        np.array([1e20, -1e20], dtype=np.float64),
        np.int64,
        saturate=True,
        warn_on_precision_loss=False,
    )
    assert int64_sat.dtype == np.dtype(np.int64)
    assert np.all(np.isfinite(int64_sat.astype(np.float64)))
    assert int64_sat[0] > 0 and int64_sat[1] < 0
    same = safe_cast(np.array([100000], dtype=np.int64), np.int64, check_range=True)
    assert same.tolist() == [100000]
    assert pipeline_utils.safe_cast is safe_cast
    print("  safe_cast: range check raises, saturate clamps")

    assert KV_CACHE_DTYPE_POLICY == {
        "default": np.float16,
        "npu": np.float16,
        "rocm": np.float16,
        "cpu": np.float32,
        "compressed": np.int8,
    }
    assert get_kv_cache_dtype("default") == np.dtype(np.float16)
    assert get_kv_cache_dtype("npu") == np.float16
    assert get_kv_cache_dtype("VitisAIExecutionProvider") == np.float16
    assert get_kv_cache_dtype("ryzenai") == np.float16
    assert get_kv_cache_dtype("ROCmExecutionProvider") == np.float16
    assert get_kv_cache_dtype("hip") == np.float16
    assert get_kv_cache_dtype("igpu") == np.float16
    assert get_kv_cache_dtype("cpu") == np.float32
    assert get_kv_cache_dtype("CPUExecutionProvider") == np.dtype(np.float32)
    assert get_kv_cache_dtype("npu", "compressed") == np.dtype(np.int8)
    assert get_kv_cache_dtype("rocm", "int8") == np.dtype(np.int8)
    assert get_kv_cache_dtype("cpu", "fp16") == np.dtype(np.float16)
    bf16_kv = get_kv_cache_dtype("npu", "bf16")
    if supports_bfloat16():
        assert bf16_kv == np.dtype(get_bfloat16_dtype())
    else:
        assert bf16_kv == np.dtype(np.uint16)
    assert get_kv_cache_dtype("default", "fp8") == float8_storage_dtype("e4m3fn")
    assert get_kv_cache_dtype("default", "int4") == np.dtype(np.uint8)
    assert get_kv_cache_dtype("default", "fp4") == np.dtype(np.uint8)
    key_dtype, value_dtype = get_kv_cache_dtypes("npu", key="bf16", value="fp8")
    assert key_dtype == bf16_kv
    assert value_dtype == float8_storage_dtype("e4m3fn")
    split_key, split_value = get_kv_cache_dtypes("cpu", "key:fp16,value:int8")
    assert split_key == np.dtype(np.float16)
    assert split_value == np.dtype(np.int8)
    same_key, same_value = get_kv_cache_dtypes("rocm")
    assert same_key == same_value == np.dtype(np.float16)
    try:
        get_kv_cache_dtype("CUDAExecutionProvider")
        raise AssertionError("CUDA has no KV cache dtype policy")
    except ValueError as exc:
        assert "CUDAExecutionProvider" in str(exc)
    try:
        get_kv_cache_dtype("npu", "int3")
        raise AssertionError("int3 is not a KV compression")
    except ValueError as exc:
        assert "int3" in str(exc)
    try:
        get_kv_cache_dtype("npu", "key:bf16,value:fp8")
        raise AssertionError("a split scheme is not one dtype")
    except ValueError as exc:
        assert "get_kv_cache_dtypes" in str(exc)
    assert pipeline_utils.KV_CACHE_DTYPE_POLICY is KV_CACHE_DTYPE_POLICY
    assert pipeline_utils.get_kv_cache_dtype is get_kv_cache_dtype
    assert pipeline_utils.get_kv_cache_dtypes is get_kv_cache_dtypes
    print("  kv cache dtype: fp16 default, cpu fp32, compressed int8, split key and value")

    assert set(PROVIDER_DTYPE_SUPPORT) == {
        "CPUExecutionProvider",
        "ROCmExecutionProvider",
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "DmlExecutionProvider",
    }
    assert PROVIDER_DTYPE_SUPPORT["VitisAIExecutionProvider"] == ("int4", "int8", "fp16")
    assert PROVIDER_DTYPE_SUPPORT["RyzenAIExecutionProvider"] == ("int4", "int8", "fp16")
    assert PROVIDER_DTYPE_SUPPORT["DmlExecutionProvider"] == ("fp16", "fp32")
    assert "fp64" in PROVIDER_DTYPE_SUPPORT["CPUExecutionProvider"]
    assert "bf16" in PROVIDER_DTYPE_SUPPORT["ROCmExecutionProvider"]
    assert is_dtype_supported_by_provider(np.float64, "CPUExecutionProvider")
    assert is_dtype_supported_by_provider("tensor(float)", "cpu")
    assert is_dtype_supported_by_provider("int4", "cpu")
    assert is_dtype_supported_by_provider(np.complex128, "CPUExecutionProvider")
    assert is_dtype_supported_by_provider("tensor(int4)", "VitisAIExecutionProvider")
    assert is_dtype_supported_by_provider(np.int8, "npu")
    assert is_dtype_supported_by_provider(np.float16, "npu")
    assert is_dtype_supported_by_provider(np.float32, "VitisAIExecutionProvider") is False
    assert is_dtype_supported_by_provider("bf16", "ryzenai") is False
    assert is_dtype_supported_by_provider(np.float16, "RyzenAIExecutionProvider")
    assert is_dtype_supported_by_provider("bf16", "ROCmExecutionProvider")
    assert is_dtype_supported_by_provider("fp8", "hip")
    assert is_dtype_supported_by_provider(np.float64, "igpu") is False
    assert is_dtype_supported_by_provider(np.float32, "DmlExecutionProvider")
    assert is_dtype_supported_by_provider(np.float16, "directml")
    assert is_dtype_supported_by_provider(np.int8, "dml") is False
    assert is_dtype_supported_by_provider("int4", "DmlExecutionProvider") is False
    assert is_dtype_supported_by_provider("bf16", "directml") is False
    for provider in PROVIDER_DTYPE_SUPPORT:
        for role in ("weight", "activation", "kv"):
            recommended = get_recommended_dtype(provider, role)
            assert is_dtype_supported_by_provider(recommended, provider)
    assert is_dtype_supported_by_provider(get_kv_cache_dtype("cpu"), "cpu")
    assert is_dtype_supported_by_provider(get_kv_cache_dtype("npu"), "npu")
    assert is_dtype_supported_by_provider(get_kv_cache_dtype("npu", "compressed"), "VitisAIExecutionProvider")
    assert is_dtype_supported_by_provider(get_kv_cache_dtype("rocm", "compressed"), "DmlExecutionProvider") is False
    try:
        is_dtype_supported_by_provider(np.float16, "CUDAExecutionProvider")
        raise AssertionError("CUDA has no dtype support row")
    except ValueError as exc:
        assert "CUDAExecutionProvider" in str(exc)
    try:
        is_dtype_supported_by_provider("int3", "cpu")
        raise AssertionError("int3 is not a support dtype")
    except ValueError as exc:
        assert "int3" in str(exc)
    assert pipeline_utils.PROVIDER_DTYPE_SUPPORT is PROVIDER_DTYPE_SUPPORT
    assert pipeline_utils.is_dtype_supported_by_provider is is_dtype_supported_by_provider
    print("  provider dtype support: CPU broad, NPU int4/int8/fp16, ROCm fp16/bf16, DirectML fp16/fp32")

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
    omitted_dtype = KVCacheManager(1, 1, 4, max_seq_len=2)
    from pipeline_utils.dtypes import get_kv_cache_dtype as kv_dtype_for_default
    assert omitted_dtype.dtype == kv_dtype_for_default("default")
    assert omitted_dtype.dtype == np.float16
    
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
    from pipeline_utils.dtypes import float8_storage_dtype as fp8_storage
    assert fp8_cache.cache_dtype == fp8_storage("e4m3fn")
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

    cursor = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float32,
        max_seq_len=8,
    )
    cursor_key = cursor.get_past_tensors(0)[0]
    assert cursor.reserve(8) == 8
    assert cursor.reserve(4) == 8
    assert cursor.get_past_tensors(0)[0] is cursor_key
    assert cursor.current_seq_len == 0
    assert cursor.max_seq_len == 8
    prefill = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    cursor_key[:, :, :3, :] = prefill
    cursor.get_past_tensors(0)[1][:, :, :3, :] = prefill + 1
    cursor.get_past_tensors(1)[0][:, :, :3, :] = prefill
    cursor.get_past_tensors(1)[1][:, :, :3, :] = prefill + 1
    assert cursor.advance_sequence_length(3) == 3
    assert cursor.current_seq_len == 3
    assert cursor.layer_length(1) == 3
    assert int(cursor.sequence_length_scalar()) == 3
    cursor_mask = cursor.attention_mask()
    assert int(cursor_mask[0, 2]) == 1
    assert int(cursor_mask[0, 3]) == 0
    cursor_key[:, :, 3, :] = 9
    cursor.get_past_tensors(0)[1][:, :, 3, :] = 9
    with cursor._lock:
        assert cursor.advance_sequence_length(1) == 4
    assert cursor.get_past_tensors(0)[0] is cursor_key
    assert cursor.seen_tokens == 4
    assert cursor.truncate_to_length(3) == 3
    assert int(cursor.attention_mask()[0, 3]) == 0
    decode_token = np.full((1, 2, 1, 4), 5, dtype=np.float32)
    cursor.update_from_present(0, decode_token, decode_token)
    assert cursor.current_seq_len == 4
    assert cursor.layer_length(0) == 4
    assert np.allclose(cursor.get_past_tensors(0)[0][:, :, 3, :], 5)
    assert np.allclose(cursor.get_past_tensors(0)[0][:, :, :3, :], prefill)
    try:
        cursor.advance_sequence_length(5)
        raise AssertionError("advance past max_seq_len was accepted")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    try:
        cursor.truncate_to_length(cursor.current_seq_len + 1)
        raise AssertionError("truncate past the cursor was accepted")
    except ValueError as exc:
        assert "current_seq_len" in str(exc)
    try:
        cursor.advance_sequence_length(-1)
        raise AssertionError("negative advance was accepted")
    except ValueError as exc:
        assert "num_tokens" in str(exc)

    dynamic_cursor = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float32,
    )
    try:
        dynamic_cursor.advance_sequence_length(1)
        raise AssertionError("advance on a dynamic cache was accepted")
    except ValueError as exc:
        assert "reserve" in str(exc)
    stored = np.arange(1 * 2 * 2 * 4, dtype=np.float32).reshape(1, 2, 2, 4)
    dynamic_cursor.update_from_present(0, stored, stored + 1)
    assert dynamic_cursor.reserve(8) == 8
    assert dynamic_cursor.static
    assert dynamic_cursor.current_seq_len == 2
    assert dynamic_cursor.get_past_tensors(0)[0].shape[2] == 8
    assert np.allclose(dynamic_cursor.get_past_tensors(0)[0][:, :, :2, :], stored)
    reserved_key = dynamic_cursor.get_past_tensors(0)[0]
    assert dynamic_cursor.reserve(8) is not None
    assert dynamic_cursor.get_past_tensors(0)[0] is reserved_key
    try:
        dynamic_cursor.reserve(16)
        raise AssertionError("reserve past max_seq_len was accepted")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)

    sliding_cursor = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float32,
        max_seq_len=8,
        policy=KVCachePolicy.SLIDING_WINDOW,
        window_size=4,
    )
    assert sliding_cursor.advance_sequence_length(4) == 4
    try:
        sliding_cursor.advance_sequence_length(1)
        raise AssertionError("sliding advance past the window was accepted")
    except ValueError as exc:
        assert "sliding_window" in str(exc)
    assert sliding_cursor.truncate_to_length(2) == 2

    paged_cursor = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float32,
        max_seq_len=8,
        policy=KVCachePolicy.PAGED_BLOCKS,
        block_size=4,
    )
    try:
        paged_cursor.advance_sequence_length(1)
        raise AssertionError("paged advance was accepted")
    except ValueError as exc:
        assert "paged_blocks" in str(exc)

    quant_cursor = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float32,
        kv_quantization="int8",
        max_seq_len=4,
    )
    quant_token = np.ones((1, 1, 1, 4), dtype=np.float32)
    quant_cursor.update_from_present(0, quant_token, quant_token)
    quant_cursor.quantize_kv(0)
    quant_cursor.advance_sequence_length(1)
    assert np.isnan(quant_cursor.key_scale[0])

    device_cursor = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=8,
    )
    device_ptr = int(device_cursor.get_device_tensors(0)[0].data_ptr())
    assert device_cursor.reserve(8) == 8
    device_cursor.advance_sequence_length(2)
    device_cursor.truncate_to_length(1)
    assert int(device_cursor.get_device_tensors(0)[0].data_ptr()) == device_ptr
    assert device_cursor.current_seq_len == 1
    assert int(device_cursor.attention_mask()[0, 0]) == 1
    assert int(device_cursor.attention_mask()[0, 1]) == 0
    print("  cursor: reserve, advance_sequence_length, truncate_to_length")

    phase = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
        max_seq_len=8,
    )
    phase_buffer = phase.get_past_tensors(0)[0]
    prefill = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    assert phase.update_prefill(0, prefill, prefill) == 2
    assert phase.phase == "prefill"
    assert phase.execution_phase == "prefill"
    assert phase.role == "target"
    assert phase.get_past_tensors(0)[0] is phase_buffer
    assert np.array_equal(phase_buffer[:, :, :2, :], prefill)
    usage = phase.get_memory_usage()
    assert usage["phase"] == "prefill"
    assert usage["execution_phase"] == "prefill"
    assert usage["role"] == "target"
    decode_token = np.full((1, 1, 1, 2), 7, dtype=np.float32)
    assert phase.update_decode(0, decode_token, decode_token) == 3
    assert phase.phase == "decode"
    assert phase.execution_phase == "decode"
    assert np.array_equal(phase_buffer[:, :, :2, :], prefill)
    assert np.array_equal(phase_buffer[:, :, 2, :], decode_token[:, :, 0, :])
    try:
        phase.update_decode(0, prefill, prefill)
        raise AssertionError("decode of more than one token was accepted")
    except ValueError as exc:
        assert "decode" in str(exc)
        assert "one token" in str(exc)
    assert phase.layer_length(0) == 3
    assert phase.phase == "decode"
    draft = np.arange(10, 16, dtype=np.float32).reshape(1, 1, 3, 2)
    assert phase.update_verify(0, draft, draft, 2) == 5
    assert phase.phase == "verify"
    assert phase.execution_phase == "verify"
    assert np.array_equal(phase_buffer[:, :, :2, :], prefill)
    assert np.array_equal(phase_buffer[:, :, 2, :], decode_token[:, :, 0, :])
    assert np.array_equal(phase_buffer[:, :, 3:5, :], draft[:, :, :2, :])
    assert np.all(phase_buffer[:, :, 5, :] == 0)
    assert phase.update_verify(0, draft, draft, 0) == 5
    assert phase.phase == "verify"
    try:
        phase.update_verify(0, draft, draft, 4)
        raise AssertionError("accepted_length past the draft span was accepted")
    except ValueError as exc:
        assert "accepted_length" in str(exc)
        assert "draft span" in str(exc)
    assert phase.layer_length(0) == 5
    phase.reset()
    assert phase.phase is None
    assert phase.execution_phase is None
    assert phase.role == "target"
    assert phase.layer_length(0) == 0
    empty = np.zeros((1, 1, 0, 2), dtype=np.float32)
    try:
        phase.update_prefill(0, empty, empty)
        raise AssertionError("empty prefill was accepted")
    except ValueError as exc:
        assert "prefill" in str(exc)
    assert phase.phase is None

    chunked = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
        max_seq_len=8,
    )
    first_chunk = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    second_chunk = np.arange(4, 10, dtype=np.float32).reshape(1, 1, 3, 2)
    assert chunked.update_prefill(0, first_chunk, first_chunk) == 2
    assert chunked.update_prefill(0, second_chunk, second_chunk) == 5
    assert chunked.phase == "prefill"
    written = chunked.get_past_tensors(0)[0][:, :, :5, :].reshape(-1)
    assert np.array_equal(written[:10], np.arange(10, dtype=np.float32))

    sliding = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=np.float32,
        max_seq_len=8,
        policy=KVCachePolicy.SLIDING_WINDOW,
        window_size=4,
    )
    slide_prompt = np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1)
    assert sliding.update_prefill(0, slide_prompt, slide_prompt) == 4
    assert sliding.phase == "sliding"
    assert sliding.execution_phase == "prefill"
    assert sliding.layer_seen_tokens(0) == 6
    assert np.array_equal(
        sliding.get_past_tensors(0)[0][:, :, :4, 0].reshape(-1),
        np.array([2, 3, 4, 5], dtype=np.float32),
    )
    slide_token = np.array([[[[9.0]]]], dtype=np.float32)
    assert sliding.update_decode(0, slide_token, slide_token) == 4
    assert sliding.phase == "sliding"
    assert sliding.execution_phase == "decode"
    assert np.array_equal(
        sliding.get_past_tensors(0)[0][:, :, :4, 0].reshape(-1),
        np.array([3, 4, 5, 9], dtype=np.float32),
    )
    slide_draft = np.array([[[[11.0]]]], dtype=np.float32)
    assert sliding.update_verify(0, slide_draft, slide_draft, 1) == 4
    assert sliding.phase == "verify"
    assert sliding.execution_phase == "verify"
    assert np.array_equal(
        sliding.get_past_tensors(0)[0][:, :, :4, 0].reshape(-1),
        np.array([4, 5, 9, 11], dtype=np.float32),
    )

    dynamic_phase = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
    )
    assert dynamic_phase.update_prefill(0, first_chunk, first_chunk) == 2
    dynamic_token = np.array([[[[8.0, 9.0]]]], dtype=np.float32)
    assert dynamic_phase.update_decode(0, dynamic_token, dynamic_token) == 3
    assert dynamic_phase.get_past_tensors(0)[0].shape[2] == 3
    assert np.array_equal(
        dynamic_phase.get_past_tensors(0)[0].reshape(-1),
        np.array([0, 1, 2, 3, 8, 9], dtype=np.float32),
    )
    assert dynamic_phase.phase == "decode"

    locked = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
        max_seq_len=4,
    )
    with locked._lock:
        assert locked.update_decode(0, dynamic_token, dynamic_token) == 1
    assert locked.phase == "decode"

    device_phase = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=8,
    )
    device_ptr = int(device_phase.get_device_tensors(0)[0].data_ptr())
    device_prefill = np.arange(4, dtype=np.float16).reshape(1, 1, 2, 2)
    device_token = np.ones((1, 1, 1, 2), dtype=np.float16)
    device_draft = np.arange(6, dtype=np.float16).reshape(1, 1, 3, 2)
    assert device_phase.update_prefill(0, device_prefill, device_prefill) == 2
    assert device_phase.update_decode(0, device_token, device_token) == 3
    assert device_phase.update_verify(0, device_draft, device_draft, 2) == 5
    assert int(device_phase.get_device_tensors(0)[0].data_ptr()) == device_ptr
    assert device_phase.phase == "verify"
    assert np.array_equal(
        device_phase.get_device_tensors(0)[0].numpy()[:, :, :2, :],
        device_prefill,
    )

    hybrid_phase = HybridKVCacheManager(
        layer_types=["dense"],
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
        max_seq_len=4,
        draft=1,
        verify=1,
    )
    assert hybrid_phase.role == "target"
    assert hybrid_phase.draft_cache.role == "draft"
    assert hybrid_phase.verify_cache.role == "verify"
    assert hybrid_phase.draft_cache.update_decode(0, dynamic_token, dynamic_token) == 1
    assert hybrid_phase.draft_cache.phase == "speculative_draft"
    assert hybrid_phase.draft_cache.execution_phase == "decode"
    hybrid_phase.draft_cache.reset()
    assert hybrid_phase.draft_cache.phase is None
    assert hybrid_phase.draft_cache.execution_phase is None
    assert hybrid_phase.draft_cache.role == "draft"
    print("  phases: update_prefill, update_decode, update_verify")

    print("kv_cache module: OK\n")
    return True


def test_config_loader():
    """Test config_loader module."""
    print("=" * 50)
    print("Testing config_loader module...")
    print("=" * 50)
    
    from pipeline_utils.config_loader import (
        ModelConfig,
        load_model_config,
        validate_config_against_onnx,
    )
    
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
    assert config.max_seq_len == 0
    assert d["max_seq_len"] == 0
    assert config.quantization_method is None
    assert config.weight_dtype is None
    assert config.activation_dtype is None
    assert config.quantization_group_size is None
    assert config.quantization_format is None
    assert config.has_zero_point is None
    assert d["quantization_method"] is None
    assert d["has_zero_point"] is None
    assert config.is_moe is False
    assert config.num_experts == 0
    assert config.num_experts_per_token == 0
    assert config.attention_variant == "dense"
    assert config.architecture_extras == {}
    assert d["is_moe"] is False
    assert d["num_experts"] == 0
    assert d["attention_variant"] == "dense"
    assert d["architecture_extras"] == {}
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, num_experts=-1).validate()
        raise AssertionError("a negative expert count must be rejected")
    except ValueError as exc:
        assert "num_experts" in str(exc)
    try:
        ModelConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, num_experts=2, num_experts_per_token=4
        ).validate()
        raise AssertionError("selecting more experts than exist must be rejected")
    except ValueError as exc:
        assert "num_experts_per_token" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, attention_variant="").validate()
        raise AssertionError("an empty attention variant must be rejected")
    except ValueError as exc:
        assert "attention_variant" in str(exc)
    ModelConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, is_moe=True, num_experts=8, num_experts_per_token=2
    ).validate()
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=-1).validate()
        raise AssertionError("a negative max_seq_len must be rejected")
    except ValueError as exc:
        assert "max_seq_len must be non-negative" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, quantization_group_size=0).validate()
        raise AssertionError("a zero group size must be rejected")
    except ValueError as exc:
        assert "quantization_group_size" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, quantization_group_size=-2).validate()
        raise AssertionError("a group size below -1 must be rejected")
    except ValueError as exc:
        assert "per-channel" in str(exc)
    ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, quantization_group_size=-1).validate()
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, has_zero_point="yes").validate()
        raise AssertionError("a non-boolean zero point must be rejected")
    except ValueError as exc:
        assert "has_zero_point" in str(exc)

    import json
    import os
    import tempfile
    import warnings

    from pipeline_utils import load_model_config as exported_loader
    from pipeline_utils import validate_config_against_onnx as exported_validate
    assert exported_loader is load_model_config
    assert exported_validate is validate_config_against_onnx

    def _write_json(directory, name, payload):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def _encode_varint(number):
        out = bytearray()
        value = int(number)
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    def _proto_key(field, wire):
        return _encode_varint((field << 3) | wire)

    def _proto_int(field, number):
        return _proto_key(field, 0) + _encode_varint(number)

    def _proto_bytes(field, payload):
        data = payload.encode() if isinstance(payload, str) else payload
        return _proto_key(field, 2) + _encode_varint(len(data)) + data

    def _write_past_key_onnx(path, shapes, elem=1, external=None, present=True):
        def dim(item):
            if isinstance(item, int):
                return _proto_bytes(1, _proto_int(1, item))
            return _proto_bytes(1, _proto_bytes(2, item))

        def value_info(name, dims):
            shape = b"".join(dim(item) for item in dims)
            tensor = _proto_int(1, elem) + _proto_bytes(2, shape)
            return _proto_bytes(1, name) + _proto_bytes(2, _proto_bytes(1, tensor))

        def node(op_type, inputs, outputs):
            body = b"".join(_proto_bytes(1, item) for item in inputs)
            body += b"".join(_proto_bytes(2, item) for item in outputs)
            body += _proto_bytes(4, op_type)
            return body

        graph = _proto_bytes(2, "g")
        if external:
            entry = _proto_bytes(1, "location") + _proto_bytes(2, external)
            initializer = (
                _proto_int(1, 1)
                + _proto_int(2, elem)
                + _proto_bytes(8, "weights")
                + _proto_bytes(13, entry)
                + _proto_int(14, 1)
            )
            graph += _proto_bytes(5, initializer)
        for name, dims in shapes:
            output = ("present_" if present else "hidden_") + name.replace(".", "_")
            graph += _proto_bytes(1, node("Identity", [name], [output]))
            graph += _proto_bytes(11, value_info(name, dims))
            graph += _proto_bytes(12, value_info(output, dims))
        model = _proto_int(1, 8) + _proto_bytes(7, graph) + _proto_bytes(8, _proto_int(2, 13))
        with open(path, "wb") as handle:
            handle.write(model)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                },
            },
        })
        genai = load_model_config(directory, require_all_fields=True)
        assert genai.num_layers == 3
        assert genai.num_kv_heads == 4
        assert genai.head_dim == 96
        assert genai.max_seq_len == 0
        assert genai.quantization_method is None
        assert genai.quantization_group_size is None
        assert genai.dtype == np.dtype(np.float16)
        assert genai.is_moe is False
        assert genai.attention_variant == "dense"
        assert genai.architecture_extras == {}

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "config.json", {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 384,
            "model_type": "llama",
        })
        hf = load_model_config(directory)
        assert hf.num_kv_heads == 2
        assert hf.head_dim == 96
        assert hf.max_seq_len == 0
        assert hf.quantization_method is None
        assert hf.weight_dtype is None
        assert hf.is_moe is False
        assert hf.num_experts == 0
        assert hf.attention_variant == "dense"
        assert hf.architecture_extras == {}

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": {"num_kv_heads": 4, "head_dim": 96, "hidden_size": 384},
        })
        with open(os.path.join(directory, "model.onnx"), "wb") as handle:
            handle.write(b"not an onnx model")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            explicit = load_model_config(directory, require_all_fields=True)
        assert explicit.num_kv_heads == 4
        assert explicit.head_dim == 96
        assert not any("Could not read ONNX" in str(item.message) for item in caught)
        assert not any("missing" in str(item.message) for item in caught)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": "llama",
            "num_key_value_heads": 4,
            "head_dim": 96,
        })
        named = load_model_config(directory, require_all_fields=True)
        assert named.num_kv_heads == 4
        assert named.head_dim == 96

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": "llama",
            "num_kv_heads": 4,
            "hidden_size": 384,
            "num_attention_heads": 4,
        })
        ratio = load_model_config(directory, require_all_fields=True)
        assert ratio.num_kv_heads == 4
        assert ratio.head_dim == 96

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": "llama",
        })
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("incomplete model_info should fail loudly")
        except ValueError as exc:
            message = str(exc)
            assert "missing num_kv_heads, head_dim" in message
            assert "Refusing defaults" in message
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback = load_model_config(directory, require_all_fields=False)
        assert fallback.num_kv_heads == 8
        assert fallback.head_dim == 128
        assert any("missing num_kv_heads, head_dim" in str(item.message) for item in caught)
        assert any("require_all_fields=True" in str(item.message) for item in caught)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": {"num_kv_heads": 4, "head_dim": 96},
        })
        # head_dim is present; only a missing field may use the explicit default.
        info = json.load(open(os.path.join(directory, "model_info.json"), encoding="utf-8"))
        info["architecture"].pop("num_kv_heads")
        _write_json(directory, "model_info.json", info)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partial = load_model_config(directory)
        assert partial.num_kv_heads == 8
        assert partial.head_dim == 96
        assert any("missing num_kv_heads" in str(item.message) for item in caught)
        assert not any("missing num_kv_heads, head_dim" in str(item.message) for item in caught)
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("missing num_kv_heads should raise")
        except ValueError as exc:
            assert "missing num_kv_heads" in str(exc)
            assert "missing num_kv_heads, head_dim" not in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "architecture": {"num_kv_heads": 0, "head_dim": 96},
        })
        try:
            load_model_config(directory)
            raise AssertionError("explicit zero heads must not become 8")
        except ValueError as exc:
            assert "num_kv_heads must be positive" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        onnx_path = os.path.join(directory, "layer_0.onnx")
        _write_past_key_onnx(onnx_path, [
            ("past_key_values.0.key", [1, 4, "seq", 96]),
            ("past_key_values.0.value", [1, 8, "seq", 32]),
        ])
        _write_json(directory, "model_info.json", {
            "num_layers": 24,
            "architecture": "llama",
            "components": {"layer_0": {"file": "layer_0.onnx"}},
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inferred = load_model_config(directory, require_all_fields=True)
        assert inferred.num_kv_heads == 4
        assert inferred.head_dim == 96
        assert not any("Using explicit defaults" in str(item.message) for item in caught)

        dynamic_path = os.path.join(directory, "layer_0.onnx")
        _write_past_key_onnx(dynamic_path, [
            ("past_key_values.0.key", [1, 4, "seq", "head_dim"]),
        ])
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("a symbolic head_dim must not be invented")
        except ValueError as exc:
            assert "missing head_dim" in str(exc)
            assert "missing num_kv_heads" not in str(exc)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dynamic = load_model_config(directory)
        assert dynamic.num_kv_heads == 4
        assert dynamic.head_dim == 128

        _write_past_key_onnx(onnx_path, [
            ("past_key_values.0.key", [1, 4, "seq", 96]),
            ("past_key_values.1.key", [1, 2, "seq", 96]),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            disagreed = load_model_config(directory, require_all_fields=True)
        assert disagreed.num_kv_heads == 4
        assert disagreed.head_dim == 96
        assert any("disagrees" in str(item.message) for item in caught)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "architecture": "llama",
            "components": {"layer_0": {"file": "layer_0.onnx"}},
        })
        with open(os.path.join(directory, "layer_0.onnx"), "wb") as handle:
            handle.write(b"not onnx")
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("an unreadable graph must still fail loudly")
        except ValueError as exc:
            assert "missing num_kv_heads, head_dim" in str(exc)

    kv_shapes = [
        ("past_key_values.0.key", [1, 4, "seq", 96]),
        ("past_key_values.0.value", [1, 4, "seq", 96]),
    ]

    def _matching_config(**overrides):
        fields = dict(num_layers=2, num_kv_heads=4, head_dim=96, dtype=np.float16)
        fields.update(overrides)
        return ModelConfig(**fields)

    def _write_split(directory, shapes=None, elem=10, external="weights.onnx.data", present=True):
        shapes = kv_shapes if shapes is None else shapes
        _write_past_key_onnx(
            os.path.join(directory, "embeddings.onnx"),
            [("hidden", [1, "seq", 384])],
            elem=1,
        )
        for index in (0, 1):
            _write_past_key_onnx(
                os.path.join(directory, f"layer_{index}.onnx"),
                shapes,
                elem=elem,
                external=external if index == 0 else None,
                present=present,
            )
        if external:
            with open(os.path.join(directory, external), "wb") as handle:
                handle.write(b"\x00\x00")
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "architecture": "llama",
            "components": {
                "embedding": {"type": "embedding", "file": "embeddings.onnx"},
                "layer_0": {"type": "layer", "layer_id": 0, "file": "layer_0.onnx"},
                "layer_1": {"type": "layer", "layer_id": 1, "file": "layer_1.onnx"},
                "norm": {"type": "norm", "file": "norm.onnx"},
            },
        })

    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory)
        report = validate_config_against_onnx(_matching_config(), directory)
        assert report["num_layers"] == 2
        assert report["layer_components"] == ["layer_0", "layer_1"]
        assert report["num_kv_heads"] == 4
        assert report["head_dim"] == 96
        assert report["dtype"] == np.dtype(np.float16)
        assert report["onnx_dtype"] == "tensor(float16)"
        assert "past_key_values.0.key" in report["past_key_values"]
        assert "past_key_values.0.value" in report["past_key_values"]
        assert any("present" in name and name.endswith("key") for name in report["present_outputs"])
        assert any(name.endswith("value") for name in report["present_outputs"])
        assert report["external_data"] == ["weights.onnx.data"]
        assert any(
            item["name"] == "hidden" and item["dtype"] == "tensor(float)"
            for item in report["inputs"]
        )
        assert any(
            item["name"] == "past_key_values.0.key" and item["shape"][1] == 4 and item["shape"][3] == 96
            for item in report["inputs"]
        )

    def _expect_mismatch(directory, config, text):
        try:
            validate_config_against_onnx(config, directory)
            raise AssertionError(f"expected mismatch containing {text}")
        except ValueError as exc:
            message = str(exc)
            assert "does not match" in message
            assert text in message
            return message

    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory)
        _expect_mismatch(directory, _matching_config(num_layers=3), "num_layers 3 != 2")
        _expect_mismatch(directory, _matching_config(num_kv_heads=8), "num_kv_heads 8 != 4")
        _expect_mismatch(directory, _matching_config(head_dim=128), "head_dim 128 != 96")
        _expect_mismatch(directory, _matching_config(dtype=np.float32), "dtype float32 != tensor(float16)")
        os.remove(os.path.join(directory, "weights.onnx.data"))
        _expect_mismatch(directory, _matching_config(), "missing external data weights.onnx.data")
        os.remove(os.path.join(directory, "layer_1.onnx"))
        _expect_mismatch(directory, _matching_config(), "layer component layer_1 is missing layer_1.onnx")

    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory, present=False)
        _expect_mismatch(
            directory,
            _matching_config(),
            "has a past key input and no present key output",
        )

    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory, shapes=[
            ("past_key_values.0.key", [1, 4, "seq", "head_dim"]),
            ("past_key_values.0.value", [1, 4, "seq", "head_dim"]),
        ])
        _expect_mismatch(directory, _matching_config(), "head_dim is symbolic")

    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory)
        _write_past_key_onnx(
            os.path.join(directory, "layer_1.onnx"),
            [("past_key_values.0.key", [1, 2, "seq", 96]),
             ("past_key_values.0.value", [1, 2, "seq", 96])],
            elem=10,
        )
        _expect_mismatch(directory, _matching_config(), "num_kv_heads 2 from")

    with tempfile.TemporaryDirectory() as directory:
        _write_past_key_onnx(os.path.join(directory, "model.onnx"), [
            ("past_key_values.0.key", [1, 4, "seq", 96]),
            ("past_key_values.0.value", [1, 4, "seq", 96]),
            ("past_key_values.1.key", [1, 4, "seq", 96]),
            ("past_key_values.1.value", [1, 4, "seq", 96]),
        ], elem=10)
        monolithic = validate_config_against_onnx(_matching_config(), directory)
        assert monolithic["num_layers"] == 2
        assert monolithic["layer_components"] == ["past_key_0", "past_key_1"]
        assert monolithic["num_kv_heads"] == 4
        assert monolithic["head_dim"] == 96

    with tempfile.TemporaryDirectory() as directory:
        try:
            validate_config_against_onnx(_matching_config(), directory)
            raise AssertionError("an empty directory has no ONNX graph")
        except ValueError as exc:
            assert "No ONNX graph" in str(exc)
    try:
        validate_config_against_onnx(_matching_config(), os.path.join(tempfile.gettempdir(), "missing-model-dir"))
        raise AssertionError("a missing directory should raise")
    except FileNotFoundError as exc:
        assert "model directory not found" in str(exc)
    try:
        validate_config_against_onnx({"num_layers": 2}, tempfile.gettempdir())
        raise AssertionError("a plain dict is not a ModelConfig")
    except TypeError as exc:
        assert "ModelConfig" in str(exc)

    hf_base = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 384,
        "model_type": "llama",
    }
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["max_position_embeddings"] = 4096
        payload["seq_length"] = 128
        _write_json(directory, "config.json", payload)
        hf_limit = load_model_config(directory)
        assert hf_limit.max_seq_len == 4096
        assert hf_limit.to_dict()["max_seq_len"] == 4096
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["max_position_embeddings"] = 0
        payload["max_sequence_length"] = 8192
        _write_json(directory, "config.json", payload)
        assert load_model_config(directory).max_seq_len == 8192
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["seq_length"] = 1024
        _write_json(directory, "config.json", payload)
        assert load_model_config(directory).max_seq_len == 1024
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["n_positions"] = 512
        _write_json(directory, "config.json", payload)
        assert load_model_config(directory).max_seq_len == 512
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                    "max_position_embeddings": 4096,
                },
            },
        })
        assert load_model_config(directory).max_seq_len == 4096
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "architecture": "llama",
            "num_kv_heads": 4,
            "head_dim": 96,
            "max_position_embeddings": 2048,
        })
        info_limit = load_model_config(directory, require_all_fields=True)
        assert info_limit.num_kv_heads == 4
        assert info_limit.max_seq_len == 2048
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["max_position_embeddings"] = "nope"
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a non-integer context length must be rejected")
        except ValueError as exc:
            assert "max_position_embeddings" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "float16"
        _write_json(directory, "config.json", payload)
        plain = load_model_config(directory)
        assert plain.quantization_method is None
        assert plain.weight_dtype is None
        assert plain.activation_dtype is None
        assert plain.has_zero_point is None
        assert plain.dtype == np.dtype(np.float16)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "float16"
        payload["quantization_method"] = "rtn"
        payload["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        }
        _write_json(directory, "config.json", payload)
        awq = load_model_config(directory)
        assert awq.quantization_method == "rtn"
        assert awq.weight_dtype == "int4"
        assert awq.activation_dtype == "fp16"
        assert awq.quantization_group_size == 128
        assert awq.quantization_format == "gemm"
        assert awq.has_zero_point is True
        assert awq.to_dict()["weight_dtype"] == "int4"
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["quantization_config"] = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": -1,
            "sym": True,
        }
        _write_json(directory, "config.json", payload)
        gptq = load_model_config(directory)
        assert gptq.quantization_method == "gptq"
        assert gptq.weight_dtype == "int4"
        assert gptq.quantization_group_size == -1
        assert gptq.has_zero_point is False
        assert gptq.quantization_format is None
        assert gptq.activation_dtype is None
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["quantization_config"] = {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "int4", "group_size": 32, "zero_point": True},
                "activation": {"dtype": "int8"},
            },
        }
        _write_json(directory, "config.json", payload)
        quark = load_model_config(directory)
        assert quark.quantization_method == "quark"
        assert quark.weight_dtype == "int4"
        assert quark.activation_dtype == "int8"
        assert quark.quantization_group_size == 32
        assert quark.has_zero_point is True
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "quantization_group_size": 0,
            "block_size": 64,
            "weight_dtype": "uint4",
        }
        _write_json(directory, "config.json", payload)
        fallen = load_model_config(directory)
        assert fallen.quantization_method == "awq"
        assert fallen.weight_dtype == "uint4"
        assert fallen.quantization_group_size == 64
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["quantization_config"] = {
            "quant_method": "awq",
            "group_size": "nope",
        }
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a non-integer group size must be rejected")
        except ValueError as exc:
            assert "group_size" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                    "quantization": {
                        "quant_method": "MatMulNBits",
                        "bits": 4,
                        "block_size": 32,
                        "is_symmetric": True,
                        "format": "MatMulNBits",
                        "signed": False,
                    },
                },
            },
        })
        nbits = load_model_config(directory)
        assert nbits.quantization_method == "matmul_nbits"
        assert nbits.weight_dtype == "uint4"
        assert nbits.quantization_group_size == 32
        assert nbits.quantization_format == "matmul_nbits"
        assert nbits.has_zero_point is False
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "architecture": "llama",
            "num_kv_heads": 4,
            "head_dim": 96,
            "quantization_method": "rtn",
            "weight_dtype": "int4",
            "activation_dtype": "fp16",
            "quantization_group_size": 32,
            "quantization_format": "QDQ",
            "has_zero_point": False,
        })
        info_quant = load_model_config(directory, require_all_fields=True)
        assert info_quant.quantization_method == "rtn"
        assert info_quant.weight_dtype == "int4"
        assert info_quant.activation_dtype == "fp16"
        assert info_quant.quantization_group_size == 32
        assert info_quant.quantization_format == "qdq"
        assert info_quant.has_zero_point is False
        assert info_quant.dtype == np.dtype(np.float16)

    from pipeline_utils.config_loader import get_dtype_from_config
    from pipeline_utils.dtypes import get_numpy_dtype

    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "float16"
        payload["dtype"] = "float32"
        _write_json(directory, "config.json", payload)
        preferred = load_model_config(directory)
        assert preferred.dtype == np.dtype(np.float16)
        assert preferred.to_dict()["dtype"] == str(np.dtype(np.float16))
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = ""
        payload["dtype"] = "fp32"
        _write_json(directory, "config.json", payload)
        aliased = load_model_config(directory)
        assert aliased.dtype == np.dtype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "bfloat16"
        _write_json(directory, "config.json", payload)
        bf16_config = load_model_config(directory)
        assert bf16_config.dtype == np.dtype(get_numpy_dtype("bfloat16"))
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "nope"
        payload["dtype"] = "float32"
        _write_json(directory, "config.json", payload)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovered = load_model_config(directory)
        assert recovered.dtype == np.dtype(np.float32)
        assert any("torch_dtype" in str(item.message) for item in caught)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "torch_dtype": "float16",
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                    "dtype": "float32",
                },
            },
        })
        genai_dtype = load_model_config(directory)
        assert genai_dtype.dtype == np.dtype(np.float16)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                    "dtype": "float32",
                },
            },
        })
        decoder_dtype = load_model_config(directory)
        assert decoder_dtype.dtype == np.dtype(np.float32)
        assert get_dtype_from_config({"model": {"decoder": {"dtype": "bf16"}}}) == np.dtype(
            get_numpy_dtype("bfloat16")
        )
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "architecture": "llama",
            "num_kv_heads": 4,
            "head_dim": 96,
            "dtype": "float64",
        })
        info_dtype = load_model_config(directory, require_all_fields=True)
        assert info_dtype.dtype == np.dtype(np.float64)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "architecture": {"dtype": "float32", "num_kv_heads": 4, "head_dim": 96},
        })
        arch_dtype = load_model_config(directory, require_all_fields=True)
        assert arch_dtype.dtype == np.dtype(np.float32)

    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({
            "num_experts": 0,
            "num_local_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1408,
            "router_aux_loss_coef": 0.25,
        })
        _write_json(directory, "config.json", payload)
        qwen_moe = load_model_config(directory)
        assert qwen_moe.is_moe is True
        assert qwen_moe.num_experts == 128
        assert qwen_moe.num_experts_per_token == 8
        assert qwen_moe.attention_variant == "dense"
        assert qwen_moe.architecture_extras["num_local_experts"] == 128
        assert qwen_moe.architecture_extras["moe_intermediate_size"] == 1408
        assert qwen_moe.architecture_extras["router_aux_loss_coef"] == 0.25
        assert "sliding_window" not in qwen_moe.architecture_extras
        assert qwen_moe.to_dict()["num_experts"] == 128
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "router_aux_loss_coef": 0,
        })
        _write_json(directory, "config.json", payload)
        mixtral = load_model_config(directory)
        assert mixtral.is_moe is True
        assert mixtral.num_experts == 8
        assert mixtral.num_experts_per_token == 2
        assert mixtral.architecture_extras["num_local_experts"] == 8
        assert mixtral.architecture_extras["router_aux_loss_coef"] == 0.0
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["is_moe"] = True
        _write_json(directory, "config.json", payload)
        flagged = load_model_config(directory)
        assert flagged.is_moe is True
        assert flagged.num_experts == 0
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["num_experts"] = "nope"
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a non-integer expert count must be rejected")
        except ValueError as exc:
            assert "num_experts" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({"num_experts": 4, "num_experts_per_tok": 8})
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("too many experts per token must be rejected")
        except ValueError as exc:
            assert "num_experts_per_token" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({
            "sliding_window": 4096,
            "use_linear_attention": False,
        })
        _write_json(directory, "config.json", payload)
        sliding = load_model_config(directory)
        assert sliding.is_moe is False
        assert sliding.attention_variant == "sliding_window"
        assert sliding.architecture_extras["sliding_window"] == 4096
        assert sliding.architecture_extras["use_linear_attention"] is False
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["sliding_window"] = 0
        _write_json(directory, "config.json", payload)
        unset_window = load_model_config(directory)
        assert unset_window.attention_variant == "dense"
        assert unset_window.architecture_extras == {}
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["use_linear_attention"] = True
        _write_json(directory, "config.json", payload)
        linear = load_model_config(directory)
        assert linear.attention_variant == "linear"
        assert linear.architecture_extras["use_linear_attention"] is True
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["use_sparse_attention"] = True
        _write_json(directory, "config.json", payload)
        sparse = load_model_config(directory)
        assert sparse.attention_variant == "sparse"
        assert sparse.architecture_extras["use_sparse_attention"] is True
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({
            "attention_type": "full_attention",
            "sliding_window": 2048,
            "use_linear_attention": True,
            "layer_types": ["linear_attention", "full_attention"],
            "moe": {"num_experts": 64, "num_experts_per_tok": 4},
        })
        _write_json(directory, "config.json", payload)
        hybrid = load_model_config(directory)
        assert hybrid.is_moe is True
        assert hybrid.num_experts == 64
        assert hybrid.num_experts_per_token == 4
        assert hybrid.attention_variant == "hybrid"
        assert hybrid.architecture_extras["attention_type"] == "full_attention"
        assert hybrid.architecture_extras["sliding_window"] == 2048
        assert hybrid.architecture_extras["use_linear_attention"] is True
        assert hybrid.architecture_extras["layer_types"] == ["linear_attention", "full_attention"]
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["attention_type"] = "chunked"
        _write_json(directory, "config.json", payload)
        chunked = load_model_config(directory)
        assert chunked.attention_variant == "chunked"
        assert chunked.architecture_extras["attention_type"] == "chunked"
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["use_linear_attention"] = "yes"
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a non-boolean linear flag must be rejected")
        except ValueError as exc:
            assert "use_linear_attention" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["sliding_window"] = -4
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a negative sliding window must be rejected")
        except ValueError as exc:
            assert "sliding_window" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "qwen3_moe",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 3,
                    "num_key_value_heads": 4,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                    "num_local_experts": 32,
                    "num_experts_per_tok": 4,
                    "attention_type": "linear",
                    "moe_intermediate_size": 768,
                },
            },
        })
        genai_moe = load_model_config(directory)
        assert genai_moe.is_moe is True
        assert genai_moe.num_experts == 32
        assert genai_moe.num_experts_per_token == 4
        assert genai_moe.attention_variant == "linear"
        assert genai_moe.architecture_extras["num_local_experts"] == 32
        assert genai_moe.architecture_extras["moe_intermediate_size"] == 768
        assert genai_moe.architecture_extras["attention_type"] == "linear"
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "architecture": {
                "num_kv_heads": 4,
                "head_dim": 96,
                "num_experts": 16,
                "num_experts_per_token": 2,
                "attention_type": "sparse",
            },
            "sliding_window": 1024,
        })
        info_moe = load_model_config(directory, require_all_fields=True)
        assert info_moe.is_moe is True
        assert info_moe.num_experts == 16
        assert info_moe.num_experts_per_token == 2
        assert info_moe.attention_variant == "hybrid"
        assert info_moe.architecture_extras["attention_type"] == "sparse"
        assert info_moe.architecture_extras["sliding_window"] == 1024
        assert info_moe.architecture_extras["use_sparse_attention"] is True

    print("  kv shape: refuse silent 8/128, infer 4 x 96 from ONNX")
    print("  max_seq_len: max_position_embeddings, sequence aliases, unset stays 0")
    print("  quantization: AWQ, GPTQ, Quark, MatMulNBits, unset stays empty")
    print("  dtype: torch_dtype, dtype, decoder dtype, missing stays float16")
    print("  architecture: moe experts, hybrid attention, dense stays empty")
    print("  onnx graph: layers, KV axes, dtype, present outputs, external data")
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
    assert onnx_type == 10
    assert NUMPY_TO_ONNX_ELEMENT_TYPE[np.float16] == 10
    
    onnx_type_int = get_onnx_element_type(np.int64)
    print(f"  np.int64 -> ONNX type: {onnx_type_int}")
    assert onnx_type_int == 7
    assert get_onnx_element_type(np.dtype(np.float32)) == 1
    
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


def test_backend():
    """OnnxRuntimeBackend.generate runs a split pipeline or a monolithic loop."""
    print("=" * 50)
    print("Testing backend module...")
    print("=" * 50)

    import os
    import tempfile
    import warnings

    import onnxruntime as ort

    from pipeline_utils.backend import (
        DEVICE_PROVIDER_MAP,
        BackendConfig,
        BackendType,
        DeviceType,
        LlamaCppBackend,
        ModelFormat,
        OnnxRuntimeBackend,
        TorchBackend,
        VLLMBackend,
        detect_model_format,
        filter_onnx_providers,
        get_backend,
        get_backend_for_model,
        _pair_present_to_past,
        _sample_next_token,
        _sequential_pipeline_class,
        build_execution_plan,
        providers_for_device,
        resolve_execution_policy,
        resolve_onnx_providers,
    )
    from pipeline_utils.kv_cache import KVCacheManager

    fresh = OnnxRuntimeBackend()
    assert fresh.kv_cache is None
    fresh.reset_cache()
    empty_stats = fresh.get_cache_stats()
    assert empty_stats["attached"] is False
    assert empty_stats["sequence_length"] == 0
    assert empty_stats["current_seq_len"] == 0
    assert empty_stats["num_layers"] == 0
    assert empty_stats["total_bytes"] == 0
    assert empty_stats["total_mb"] == 0.0

    unloaded = OnnxRuntimeBackend()
    try:
        unloaded.generate("Hello")
        raise AssertionError("generate before load was accepted")
    except RuntimeError as exc:
        assert "load()" in str(exc)
    try:
        unloaded.warmup()
        raise AssertionError("warmup before load was accepted")
    except RuntimeError as exc:
        assert "load()" in str(exc)
    try:
        unloaded.warmup(-1)
        raise AssertionError("a negative warmup count was accepted")
    except ValueError as exc:
        assert "iterations" in str(exc)
    try:
        unloaded.warmup(True)
        raise AssertionError("a boolean warmup count was accepted")
    except TypeError as exc:
        assert "int" in str(exc)

    assert _sample_next_token(np.array([1.0, 3.0, 2.0]), temperature=0, top_p=0.9) == 1
    assert _sample_next_token(np.array([0.0, 0.0, 5.0]), temperature=1.0, top_p=0.1) == 2
    assert _pair_present_to_past(
        "present.0.key",
        ["past_key_values.0.key", "past_key_values.0.value"],
    ) == "past_key_values.0.key"
    assert _pair_present_to_past(
        "present.0.value",
        ["past_key_values.0.key", "past_key_values.0.value"],
    ) == "past_key_values.0.value"

    class _Node:
        def __init__(self, name, shape=None, type_name="tensor(float)"):
            self.name = name
            self.shape = shape or []
            self.type = type_name

    class _Tokenizer:
        eos_token_id = None

        def __call__(self, prompt, return_tensors="np"):
            assert prompt == "Hello"
            return {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}

        def decode(self, ids):
            return ",".join(str(int(token)) for token in ids)

    class _GreedySession:
        def __init__(self):
            self.lengths = []

        def get_inputs(self):
            return [
                _Node("input_ids", [1, "seq"], "tensor(int64)"),
                _Node("attention_mask", [1, "seq"], "tensor(int64)"),
                _Node("position_ids", [1, "seq"], "tensor(int64)"),
            ]

        def get_outputs(self):
            return [_Node("logits", [1, "seq", 6])]

        def run(self, names, feeds):
            self.lengths.append(int(feeds["input_ids"].shape[1]))
            seq = self.lengths[-1]
            logits = np.zeros((1, seq, 6), dtype=np.float32)
            logits[:, -1, 4] = 8
            return [logits]

    greedy = _GreedySession()
    monolithic = OnnxRuntimeBackend()
    monolithic.is_loaded = True
    monolithic._is_split = False
    monolithic.model = greedy
    monolithic._tokenizer = _Tokenizer()
    assert monolithic.generate("Hello", max_tokens=2, temperature=0) == "1,2,3,4,4"
    assert greedy.lengths == [3, 4]
    assert monolithic.kv_cache is None
    assert monolithic.get_cache_stats()["attached"] is False
    assert monolithic.get_memory_usage()["kv_cache_attached"] is False
    assert monolithic.get_memory_usage()["kv_sequence_length"] == 0
    assert monolithic.generate("Hello", max_tokens=0, temperature=0) == "1,2,3"

    class _EosTokenizer(_Tokenizer):
        eos_token_id = 4

    monolithic._tokenizer = _EosTokenizer()
    assert monolithic.generate("Hello", max_tokens=5, temperature=0) == "1,2,3,4"

    stream_session = _GreedySession()
    stream_backend = OnnxRuntimeBackend()
    stream_backend.is_loaded = True
    stream_backend._is_split = False
    stream_backend.model = stream_session
    stream_backend._tokenizer = _Tokenizer()
    streamed = []

    def _on_token(text, index):
        streamed.append((text, index, list(stream_session.lengths)))

    assert stream_backend.generate("Hello", 2, _on_token, temperature=0) == "1,2,3,4,4"
    assert streamed == [("4", 0, [3]), ("4", 1, [3, 4])]
    streamed.clear()
    assert stream_backend.generate(
        "Hello", max_tokens=0, temperature=0, stream_callback=_on_token
    ) == "1,2,3"
    assert streamed == []
    stream_backend._tokenizer = _EosTokenizer()
    assert stream_backend.generate(
        "Hello", max_tokens=5, temperature=0, stream_callback=_on_token
    ) == "1,2,3,4"
    assert streamed == [("4", 0, [3, 4, 3])]

    class _KvSession:
        def __init__(self):
            self.past_lengths = []
            self.masks = []

        def get_inputs(self):
            return [
                _Node("input_ids", [1, "seq"], "tensor(int64)"),
                _Node("attention_mask", [1, "total"], "tensor(int64)"),
                _Node("past_key_values.0.key", [1, 1, "past", 2]),
                _Node("past_key_values.0.value", [1, 1, "past", 2]),
            ]

        def get_outputs(self):
            return [
                _Node("logits"),
                _Node("present.0.key"),
                _Node("present.0.value"),
            ]

        def run(self, names, feeds):
            past = feeds["past_key_values.0.key"]
            self.past_lengths.append(int(past.shape[2]))
            self.masks.append(int(feeds["attention_mask"].shape[1]))
            seq = int(feeds["input_ids"].shape[1])
            total = int(past.shape[2]) + seq
            logits = np.zeros((1, seq, 6), dtype=np.float32)
            logits[:, -1, 3] = 4
            present = np.ones((1, 1, total, 2), dtype=np.float32)
            return [logits, present, present + 1]

    kv_session = _KvSession()
    kv_backend = OnnxRuntimeBackend()
    kv_backend.is_loaded = True
    kv_backend.model = kv_session
    kv_backend._tokenizer = _Tokenizer()
    assert kv_backend.kv_cache is None
    assert kv_backend.generate("Hello", max_tokens=2, temperature=0) == "1,2,3,3,3"
    assert kv_session.past_lengths == [0, 3]
    assert kv_session.masks == [3, 4]
    assert isinstance(kv_backend.kv_cache, KVCacheManager)
    assert kv_backend.kv_cache.get_sequence_length() == 4
    assert kv_backend.kv_cache.num_layers == 1
    assert kv_backend.kv_cache.num_kv_heads == 1
    assert kv_backend.kv_cache.head_dim == 2
    kv_stats = kv_backend.get_cache_stats()
    assert kv_stats["attached"] is True
    assert kv_stats["num_layers"] == 1
    assert kv_stats["sequence_length"] == 4
    assert kv_stats["current_seq_len"] == 4
    usage = kv_backend.get_memory_usage()
    assert usage["split"] is False
    assert usage["kv_cache_attached"] is True
    assert usage["kv_sequence_length"] == 4
    kv_backend.reset_cache()
    assert kv_backend.kv_cache.get_sequence_length() == 0
    assert kv_backend.get_cache_stats()["sequence_length"] == 0
    kv_session.past_lengths.clear()
    kv_session.masks.clear()
    assert kv_backend.generate("Hello", max_tokens=2, temperature=0) == "1,2,3,3,3"
    assert kv_session.past_lengths == [0, 3]
    assert kv_session.masks == [3, 4]
    kv_backend.cleanup()
    assert kv_backend.is_loaded is False
    assert kv_backend.kv_cache is not None
    assert kv_backend.get_cache_stats()["attached"] is True
    assert kv_backend.get_cache_stats()["sequence_length"] == 4
    assert kv_backend.warmed_up is False

    warm_session = _GreedySession()
    warm_backend = OnnxRuntimeBackend()
    warm_backend.is_loaded = True
    warm_backend._is_split = False
    warm_backend.model = warm_session
    warm_backend._tokenizer = _Tokenizer()
    assert warm_backend.warmup(0) == {"iterations": 0, "warmed": False}
    assert warm_session.lengths == []
    assert warm_backend.warmup() == {"iterations": 2, "warmed": True}
    assert warm_backend.warmed_up is True
    assert warm_session.lengths == [3, 4, 3, 4]
    assert warm_backend.get_memory_usage()["warmed_up"] is True
    warm_backend.cleanup()
    assert warm_backend.warmed_up is False

    warm_kv_session = _KvSession()
    warm_kv = OnnxRuntimeBackend()
    warm_kv.is_loaded = True
    warm_kv.model = warm_kv_session
    warm_kv._tokenizer = _Tokenizer()
    assert warm_kv.warmup(1) == {"iterations": 1, "warmed": True}
    assert warm_kv_session.past_lengths == [0, 3]
    assert warm_kv.kv_cache.get_sequence_length() == 0
    warm_kv_session.past_lengths.clear()
    assert warm_kv.generate("Hello", max_tokens=1, temperature=0) == "1,2,3,3"
    assert warm_kv_session.past_lengths == [0]

    class _AnonSession:
        def __init__(self):
            self.feeds = []

        def get_inputs(self):
            return [
                _Node("input_ids", [1, "seq"], "tensor(int64)"),
                _Node("past_key_values.0.key", [1, 1, "past", 2]),
                _Node("past_key_values.0.value", [1, 1, "past", 2]),
            ]

        def run(self, names, feeds):
            self.feeds.append(int(feeds["past_key_values.0.key"].shape[2]))
            return [np.zeros((1, 1, 2), dtype=np.float32)]

    anon = OnnxRuntimeBackend()
    anon.is_loaded = True
    anon._is_split = False
    anon.model = _AnonSession()
    assert anon.warmup(1)["warmed"] is True
    assert anon.model.feeds == [0, 1]
    assert anon.kv_cache is None

    class _OddSession(_GreedySession):
        def get_inputs(self):
            return [_Node("input_ids"), _Node("custom_bias")]

    class _LlamaState:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    llama = LlamaCppBackend()
    llama.model = _LlamaState()
    llama.kv_cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
        max_seq_len=4,
    )
    llama.kv_cache.update_from_present(
        0,
        np.ones((1, 1, 2, 2), dtype=np.float32),
        np.ones((1, 1, 2, 2), dtype=np.float32),
    )
    assert llama.kv_cache.get_sequence_length() == 2
    llama.reset_cache()
    assert llama.model.resets == 1
    assert llama.kv_cache.get_sequence_length() == 0
    assert llama.get_memory_usage()["kv_cache_attached"] is True
    assert llama.get_memory_usage()["kv_sequence_length"] == 0

    class _StreamingLlama:
        def __init__(self):
            self.stream_flags = []

        def __call__(
            self,
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stream=False,
        ):
            self.stream_flags.append(stream)
            if stream:
                return [
                    {"choices": [{"text": "hi"}]},
                    {"choices": [{"text": ""}]},
                    {"choices": [{"text": "!"}]},
                ]
            return {"choices": [{"text": "hi!"}]}

    llama_stream = LlamaCppBackend()
    llama_stream.is_loaded = True
    llama_stream.model = _StreamingLlama()
    llama_pieces = []
    assert llama_stream.generate(
        "Hello",
        max_tokens=2,
        stream_callback=lambda text, index: llama_pieces.append((text, index)),
        temperature=0,
    ) == "hi!"
    assert llama_pieces == [("hi", 0), ("!", 1)]
    assert llama_stream.generate("Hello", max_tokens=2, temperature=0) == "hi!"
    assert llama_stream.model.stream_flags == [True, False]

    class _PlainLlama:
        def __call__(self, prompt, max_tokens=50, temperature=0.7, top_p=0.9, echo=False, stream=False):
            if stream:
                raise TypeError("stream is not supported")
            return {"choices": [{"text": "done"}]}

    plain = LlamaCppBackend()
    plain.is_loaded = True
    plain.model = _PlainLlama()
    plain_pieces = []
    assert plain.generate(
        "Hello",
        stream_callback=lambda text, index: plain_pieces.append((text, index)),
    ) == "done"
    assert plain_pieces == [("done", 0)]

    class _WarmLlama:
        def __init__(self):
            self.calls = []
            self.resets = 0

        def __call__(
            self,
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stream=False,
        ):
            self.calls.append((prompt, max_tokens, temperature, stream))
            return {"choices": [{"text": "ok"}]}

        def reset(self):
            self.resets += 1

    warm_llama = LlamaCppBackend()
    warm_llama.is_loaded = True
    warm_llama.model = _WarmLlama()
    assert warm_llama.warmup(2) == {"iterations": 2, "warmed": True}
    assert warm_llama.model.calls == [
        ("Hello", 2, 0.0, False),
        ("Hello", 2, 0.0, False),
    ]
    assert warm_llama.model.resets == 3
    assert warm_llama.get_memory_usage()["warmed_up"] is True
    warm_llama.cleanup()
    assert warm_llama.warmed_up is False

    odd = OnnxRuntimeBackend()
    odd.is_loaded = True
    odd.model = _OddSession()
    odd._tokenizer = _Tokenizer()
    try:
        odd.generate("Hello", max_tokens=1, temperature=0)
        raise AssertionError("an unknown ONNX input was accepted")
    except RuntimeError as exc:
        assert "custom_bias" in str(exc)

    class _FakePipeline:
        def __init__(
            self,
            model_path,
            providers=None,
            verbose=True,
            persist_sessions=False,
            phase_providers=None,
            component_providers=None,
            **kwargs,
        ):
            self.model_path = model_path
            self.providers = list(providers or [])
            self.verbose = verbose
            self.persist_sessions = persist_sessions
            self.phase_providers = phase_providers
            self.component_providers = component_providers
            self.extra = dict(kwargs)
            self.session_manager = type("Manager", (), {
                "get_memory_usage": lambda self: {"sessions": 2},
            })()
            self.tokenizer = _Tokenizer()
            self.calls = []
            self.cleaned = False

        def generate(self, prompt, max_new_tokens=20):
            self.calls.append((prompt, max_new_tokens))
            return f"split:{prompt}:{max_new_tokens}"

        def cleanup(self):
            self.cleaned = True

    created = {}

    def _fake_pipeline_class():
        def _factory(*args, **kwargs):
            pipeline = _FakePipeline(*args, **kwargs)
            created["pipeline"] = pipeline
            return pipeline
        return _factory

    import pipeline_utils.backend as backend_mod
    original_factory = backend_mod._sequential_pipeline_class
    backend_mod._sequential_pipeline_class = _fake_pipeline_class
    split_dir = tempfile.mkdtemp(prefix="onnx-split-")
    try:
        split = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            low_memory=False,
            execution_policy="monolithic",
        ))
        split.load(split_dir)
        assert split._is_split is True
        assert split.is_loaded is True
        pipeline = created["pipeline"]
        assert pipeline.persist_sessions is True
        assert pipeline.providers == ["CPUExecutionProvider"]
        assert pipeline.verbose is False
        assert split.kv_cache is None
        assert split.generate("Hello", max_tokens=7, temperature=0.2, top_p=0.5) == "split:Hello:7"
        assert pipeline.calls == [("Hello", 7)]
        fallback_tokens = []
        assert split.generate(
            "Hello",
            max_tokens=7,
            stream_callback=lambda text, index: fallback_tokens.append((text, index)),
        ) == "split:Hello:7"
        assert fallback_tokens == [("split:Hello:7", 0)]
        assert pipeline.calls == [("Hello", 7), ("Hello", 7)]
        assert split.warmup(2) == {"iterations": 2, "warmed": True}
        assert pipeline.calls == [("Hello", 7), ("Hello", 7), ("Hello", 2), ("Hello", 2)]
        assert split.get_memory_usage()["warmed_up"] is True

        class _StreamingPipeline(_FakePipeline):
            def generate(self, prompt, max_new_tokens=20, stream_callback=None):
                self.calls.append((prompt, max_new_tokens, stream_callback is not None))
                if stream_callback is not None:
                    stream_callback("tok", 0)
                    stream_callback("en", 1)
                return "token"

        def _streaming_pipeline_class():
            def _factory(*args, **kwargs):
                streaming_pipeline = _StreamingPipeline(*args, **kwargs)
                created["streaming"] = streaming_pipeline
                return streaming_pipeline
            return _factory

        backend_mod._sequential_pipeline_class = _streaming_pipeline_class
        streaming = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            execution_policy="monolithic",
        ))
        streaming.load(split_dir)
        streamed_split = []
        assert streaming.generate(
            "Hello",
            max_tokens=2,
            stream_callback=lambda text, index: streamed_split.append((text, index)),
        ) == "token"
        assert streamed_split == [("tok", 0), ("en", 1)]
        assert created["streaming"].calls == [("Hello", 2, True)]
        streaming.cleanup()
        backend_mod._sequential_pipeline_class = _fake_pipeline_class
        assert split.get_memory_usage()["sessions"] == 2
        assert split.get_memory_usage()["split"] is True
        split.cleanup()
        assert pipeline.cleaned is True
        assert split.is_loaded is False
        assert split.warmed_up is False

        preset = KVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=np.float32,
            max_seq_len=8,
        )
        preset_backend = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            execution_policy="monolithic",
        ))
        preset_backend.kv_cache = preset
        preset_backend.load(split_dir)
        assert created["pipeline"].kv_cache is preset
        assert preset_backend.kv_cache is preset
        preset_backend.cleanup()

        class _PipelineWithCache(_FakePipeline):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.kv_cache = KVCacheManager(
                    num_layers=1,
                    num_kv_heads=1,
                    head_dim=2,
                    dtype=np.float32,
                    max_seq_len=8,
                )

        def _cached_pipeline_class():
            def _factory(*args, **kwargs):
                cached = _PipelineWithCache(*args, **kwargs)
                created["cached"] = cached
                return cached
            return _factory

        backend_mod._sequential_pipeline_class = _cached_pipeline_class
        adopted = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            execution_policy="monolithic",
        ))
        adopted.load(split_dir)
        assert adopted.kv_cache is created["cached"].kv_cache
        installed = KVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            dtype=np.float32,
            max_seq_len=8,
        )
        installed_backend = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            execution_policy="monolithic",
        ))
        installed_backend.kv_cache = installed
        installed_backend.load(split_dir)
        assert created["cached"].kv_cache is installed
        assert installed_backend.kv_cache is installed
        installed_backend.cleanup()
        backend_mod._sequential_pipeline_class = _fake_pipeline_class

        low_memory = OnnxRuntimeBackend(BackendConfig(low_memory=True))
        low_memory.load(split_dir)
        assert created["pipeline"].persist_sessions is False
        low_memory.cleanup()
    finally:
        backend_mod._sequential_pipeline_class = original_factory
        os.rmdir(split_dir)

    captured = {}
    real_session = ort.InferenceSession

    class _PatchedSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            captured["path"] = model_path
            captured["level"] = sess_options.graph_optimization_level
            captured["threads"] = sess_options.intra_op_num_threads
            captured["providers"] = list(providers)

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    ort.InferenceSession = _PatchedSession
    model_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    model_file.close()
    try:
        single = OnnxRuntimeBackend(BackendConfig(
            onnx_providers=["CPUExecutionProvider"],
            onnx_optimization_level="DISABLE",
            n_threads=2,
            execution_policy="monolithic",
        ))
        single.load(model_file.name)
        assert single._is_split is False
        assert single._pipeline is None
        assert single.is_loaded is True
        assert captured["providers"] == ["CPUExecutionProvider"]
        assert captured["threads"] == 2
        assert captured["level"] == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    finally:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ort.InferenceSession = _PatchedSession
                single_hybrid = OnnxRuntimeBackend(BackendConfig(execution_policy="hybrid"))
                single_hybrid.load(model_file.name)
                assert any("single ONNX file" in str(item.message) for item in caught)
                single_hybrid.cleanup()
        finally:
            ort.InferenceSession = real_session
            os.remove(model_file.name)

    import sys
    import types
    inserted_transformers = "transformers" not in sys.modules
    if inserted_transformers:
        stub = types.ModuleType("transformers")

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("stub tokenizer")

        stub.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = stub
    try:
        pipeline_cls = _sequential_pipeline_class()
        assert pipeline_cls.__name__ == "SequentialModelPipeline"
    finally:
        if inserted_transformers:
            sys.modules.pop("transformers", None)

    assert DEVICE_PROVIDER_MAP[DeviceType.NPU] == [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert DEVICE_PROVIDER_MAP[DeviceType.GPU] == [
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert DEVICE_PROVIDER_MAP[DeviceType.CPU] == ["CPUExecutionProvider"]
    assert DEVICE_PROVIDER_MAP[DeviceType.AUTO] is None
    full = [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for_device(DeviceType.NPU, full) == [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for_device("gpu", full) == [
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for_device(DeviceType.NPU, ["CPUExecutionProvider"]) == [
        "CPUExecutionProvider",
    ]
    assert providers_for_device(DeviceType.AUTO, [
        "CUDAExecutionProvider",
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ]) == [
        "VitisAIExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for_device(DeviceType.AUTO, ["CPUExecutionProvider"]) == [
        "CPUExecutionProvider",
    ]
    try:
        providers_for_device("tpu")
        raise AssertionError("unknown device was accepted")
    except ValueError as exc:
        assert "tpu" in str(exc)
    explicit = BackendConfig(
        device=DeviceType.NPU,
        onnx_providers=["CPUExecutionProvider"],
    )
    assert resolve_onnx_providers(explicit, full) == ["CPUExecutionProvider"]
    try:
        resolve_onnx_providers(
            BackendConfig(onnx_providers=["VitisAIExecutionProvider"]),
            ["CPUExecutionProvider"],
        )
        raise AssertionError("a missing explicit provider was accepted")
    except RuntimeError as exc:
        assert "VitisAIExecutionProvider" in str(exc)

    kept, dropped = filter_onnx_providers(
        ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    )
    assert kept == ["CPUExecutionProvider"]
    assert dropped == ["VitisAIExecutionProvider"]
    assert filter_onnx_providers([], ["CPUExecutionProvider"]) == (
        ["CPUExecutionProvider"],
        [],
    )
    real_available = ort.get_available_providers
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    try:
        kept, dropped = filter_onnx_providers(["ROCmExecutionProvider", "CPUExecutionProvider"])
        assert kept == ["CPUExecutionProvider"]
        assert dropped == ["ROCmExecutionProvider"]
    finally:
        ort.get_available_providers = real_available

    real_available = ort.get_available_providers
    ort.get_available_providers = lambda: [
        "CPUExecutionProvider",
        "ROCmExecutionProvider",
    ]
    ort.InferenceSession = _PatchedSession
    try:
        gpu = OnnxRuntimeBackend(BackendConfig(
            device=DeviceType.GPU,
            execution_policy="monolithic",
        ))
        assert gpu.providers == ["ROCmExecutionProvider", "CPUExecutionProvider"]
        gpu.load(model_file.name)
        assert captured["providers"] == ["ROCmExecutionProvider", "CPUExecutionProvider"]
        gpu.cleanup()
    finally:
        ort.get_available_providers = real_available
        ort.InferenceSession = real_session

    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    ort.InferenceSession = _PatchedSession
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cpu_only = OnnxRuntimeBackend(BackendConfig(
                device=DeviceType.NPU,
                execution_policy="monolithic",
            ))
            cpu_only.load(model_file.name)
            assert captured["providers"] == ["CPUExecutionProvider"]
            assert cpu_only.providers == ["CPUExecutionProvider"]
            assert cpu_only.dropped_providers == [
                "VitisAIExecutionProvider",
                "RyzenAIExecutionProvider",
            ]
            assert any(
                "Dropped unavailable execution providers" in str(item.message)
                and "VitisAIExecutionProvider" in str(item.message)
                for item in caught
            )
            cpu_only.cleanup()
        opted_out = OnnxRuntimeBackend(BackendConfig(
            device=DeviceType.NPU,
            execution_policy="monolithic",
            use_available_providers_only=False,
        ))
        opted_out.load(model_file.name)
        assert captured["providers"] == [
            "VitisAIExecutionProvider",
            "RyzenAIExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert opted_out.dropped_providers == []
        opted_out.cleanup()

        backend_mod._sequential_pipeline_class = _fake_pipeline_class
        filtered_dir = tempfile.mkdtemp(prefix="onnx-filter-")
        try:
            filtered = OnnxRuntimeBackend(BackendConfig(
                device=DeviceType.NPU,
                execution_policy="hybrid",
                prefill_device=DeviceType.NPU,
                decode_device=DeviceType.GPU,
                npu_layer_range=(0, 1),
            ))
            filtered.load(filtered_dir)
            received = created["pipeline"]
            assert received.providers == ["CPUExecutionProvider"]
            assert received.phase_providers["prefill"] == ["CPUExecutionProvider"]
            assert received.phase_providers["decode"] == ["CPUExecutionProvider"]
            assert received.component_providers["layer_0"] == ["CPUExecutionProvider"]
            assert "VitisAIExecutionProvider" in filtered.dropped_providers
            assert "ROCmExecutionProvider" in filtered.dropped_providers
            filtered.cleanup()

            raw = OnnxRuntimeBackend(BackendConfig(
                device=DeviceType.NPU,
                execution_policy="hybrid",
                prefill_device=DeviceType.NPU,
                decode_device=DeviceType.GPU,
                use_available_providers_only=False,
            ))
            raw.load(filtered_dir)
            assert created["pipeline"].phase_providers["prefill"][0] == "VitisAIExecutionProvider"
            assert created["pipeline"].phase_providers["decode"][0] == "ROCmExecutionProvider"
            assert raw.dropped_providers == []
            raw.cleanup()
        finally:
            os.rmdir(filtered_dir)
    finally:
        ort.get_available_providers = real_available
        ort.InferenceSession = real_session
        backend_mod._sequential_pipeline_class = original_factory

    backend_mod._sequential_pipeline_class = _fake_pipeline_class
    ort.get_available_providers = lambda: [
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ]
    npu_dir = tempfile.mkdtemp(prefix="onnx-npu-")
    try:
        npu = OnnxRuntimeBackend(BackendConfig(device=DeviceType.NPU))
        npu.load(npu_dir)
        assert created["pipeline"].providers == [
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert npu.get_memory_usage()["providers"] == [
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
        ]
        npu.cleanup()
    finally:
        backend_mod._sequential_pipeline_class = original_factory
        ort.get_available_providers = real_available
        os.rmdir(npu_dir)

    installed = [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    hybrid = build_execution_plan(BackendConfig(), installed)
    assert hybrid.policy == "hybrid"
    assert hybrid.persist_sessions is False
    assert hybrid.phase_providers["prefill"][0] == "VitisAIExecutionProvider"
    assert hybrid.phase_providers["decode"][0] == "ROCmExecutionProvider"
    assert hybrid.component_providers["router"] == ["CPUExecutionProvider"]
    assert hybrid.component_providers["expert_*"][0] == "VitisAIExecutionProvider"
    ranged = build_execution_plan(
        BackendConfig(execution_policy="hybrid", npu_layer_range=(0, 2)),
        installed,
    )
    assert ranged.npu_layer_range == (0, 2)
    assert ranged.component_providers["layer_0"][0] == "VitisAIExecutionProvider"
    assert ranged.component_providers["layer_1"][0] == "VitisAIExecutionProvider"
    assert ranged.component_providers["layer_*"] == ["CPUExecutionProvider"]
    assert resolve_execution_policy(BackendConfig(
        execution_policy="auto",
        prefill_device=DeviceType.CPU,
        decode_device=DeviceType.CPU,
    )) == "monolithic"
    assert resolve_execution_policy(BackendConfig(
        execution_policy="auto",
        prefill_device=DeviceType.CPU,
        decode_device=DeviceType.CPU,
        low_memory=True,
    )) == "sequential"
    assert build_execution_plan(
        BackendConfig(execution_policy="monolithic"),
        installed,
    ).phase_providers == {}
    try:
        resolve_execution_policy(BackendConfig(execution_policy="pipeline"))
        raise AssertionError("unknown execution policy was accepted")
    except ValueError as exc:
        assert "pipeline" in str(exc)
    try:
        build_execution_plan(BackendConfig(
            execution_policy="monolithic",
            npu_layer_range=(0, 2),
        ))
        raise AssertionError("a monolithic layer range was accepted")
    except ValueError as exc:
        assert "npu_layer_range" in str(exc)
    try:
        build_execution_plan(BackendConfig(npu_layer_range=(2, 2)))
        raise AssertionError("an empty layer range was accepted")
    except ValueError as exc:
        assert "npu_layer_range" in str(exc)

    backend_mod._sequential_pipeline_class = _fake_pipeline_class
    hybrid_dir = tempfile.mkdtemp(prefix="onnx-hybrid-")
    try:
        placed = OnnxRuntimeBackend(BackendConfig(
            execution_policy="hybrid",
            prefill_device=DeviceType.NPU,
            decode_device=DeviceType.GPU,
            npu_layer_range=(0, 2),
        ))
        placed.load(hybrid_dir)
        loaded = created["pipeline"]
        assert loaded.persist_sessions is False
        assert loaded.phase_providers["prefill"]
        assert loaded.phase_providers["decode"]
        assert loaded.component_providers["layer_0"]
        assert loaded.component_providers["router"] == ["CPUExecutionProvider"]
        assert placed.get_memory_usage()["execution_policy"] == "hybrid"
        placed.cleanup()
    finally:
        backend_mod._sequential_pipeline_class = original_factory
        os.rmdir(hybrid_dir)

    pipeline_cls = _sequential_pipeline_class()
    routed = pipeline_cls.__new__(pipeline_cls)
    routed.component_providers = ranged.component_providers
    routed.phase_providers = ranged.phase_providers
    routed.providers = ["CPUExecutionProvider"]
    routed.phase = "prefill"
    assert routed._get_component_providers("layer_0")[0] == "VitisAIExecutionProvider"
    assert routed._get_component_providers("layer_9") == ["CPUExecutionProvider"]
    assert routed._get_component_providers("embedding")[0] == "VitisAIExecutionProvider"
    routed.phase = "decode"
    assert routed._get_component_providers("embedding")[0] == "ROCmExecutionProvider"
    assert routed._get_component_providers("router") == ["CPUExecutionProvider"]
    assert routed._get_component_providers("expert_1")[0] == "VitisAIExecutionProvider"

    import sys
    import types

    assert isinstance(get_backend(BackendType.TORCH), TorchBackend)
    assert isinstance(get_backend(BackendType.VLLM), VLLMBackend)
    assert get_backend(BackendType.ONNX_RUNTIME).backend_type == BackendType.ONNX_RUNTIME
    assert get_backend(BackendType.LLAMA_CPP).supported_formats == [ModelFormat.GGUF]
    torch_backend = get_backend(BackendType.TORCH, BackendConfig(device=DeviceType.CPU))
    assert torch_backend.backend_type == BackendType.TORCH
    assert torch_backend.supported_formats == [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN]
    vllm_backend = get_backend(BackendType.VLLM)
    assert vllm_backend.backend_type == BackendType.VLLM
    assert vllm_backend.supported_formats == [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN]
    try:
        get_backend("missing")
        raise AssertionError("an unknown backend name was accepted")
    except ValueError as exc:
        assert "TORCH" in str(exc) and "VLLM" in str(exc)

    weights_dir = tempfile.mkdtemp(prefix="torch-weights-")
    bin_dir = tempfile.mkdtemp(prefix="torch-bin-")
    gguf_dir = tempfile.mkdtemp(prefix="gguf-file-")
    try:
        safetensors_path = os.path.join(weights_dir, "model.safetensors")
        with open(safetensors_path, "wb") as handle:
            handle.write(b"")
        bin_path = os.path.join(bin_dir, "pytorch_model.bin")
        with open(bin_path, "wb") as handle:
            handle.write(b"")
        gguf_path = os.path.join(gguf_dir, "model.gguf")
        with open(gguf_path, "wb") as handle:
            handle.write(b"")
        assert detect_model_format(weights_dir) == ModelFormat.SAFETENSORS
        assert isinstance(get_backend_for_model(weights_dir), TorchBackend)
        assert isinstance(get_backend_for_model(safetensors_path), TorchBackend)
        assert detect_model_format(bin_path) == ModelFormat.PYTORCH_BIN
        assert isinstance(get_backend_for_model(bin_path), TorchBackend)
        assert isinstance(get_backend_for_model(gguf_path), LlamaCppBackend)
        empty_dir = tempfile.mkdtemp(prefix="empty-model-")
        try:
            detect_model_format(empty_dir)
            raise AssertionError("an empty directory was detected as a model")
        except ValueError as exc:
            assert "Cannot detect" in str(exc)
        finally:
            os.rmdir(empty_dir)
    finally:
        os.remove(safetensors_path)
        os.remove(bin_path)
        os.remove(gguf_path)
        os.rmdir(weights_dir)
        os.rmdir(bin_dir)
        os.rmdir(gguf_dir)

    def _hide_modules(*prefixes):
        saved = {}
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                saved[name] = sys.modules.pop(name)
        return saved

    def _restore_modules(saved):
        for name in list(sys.modules):
            if name in saved or any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in ("torch", "transformers", "vllm")
            ):
                if name not in saved:
                    sys.modules.pop(name, None)
        sys.modules.update(saved)

    hidden = _hide_modules("torch", "transformers", "vllm")
    try:
        missing_torch = TorchBackend()
        try:
            missing_torch.load("unused")
            raise AssertionError("Torch load without torch was accepted")
        except ImportError as exc:
            assert "pip install torch" in str(exc)
        sys.modules["torch"] = types.ModuleType("torch")
        missing_transformers = TorchBackend()
        try:
            missing_transformers.load("unused")
            raise AssertionError("Torch load without transformers was accepted")
        except ImportError as exc:
            assert "pip install transformers" in str(exc)
        missing_vllm = VLLMBackend()
        try:
            missing_vllm.load("unused")
            raise AssertionError("vLLM load without vllm was accepted")
        except ImportError as exc:
            assert "pip install vllm" in str(exc)
    finally:
        _restore_modules(hidden)

    class _TorchTensor:
        def __init__(self, data):
            self.data = np.asarray(data)
            self.shape = self.data.shape

        def to(self, device):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    def _install_torch(cuda=False, npu=False, mps=False):
        torch_mod = types.ModuleType("torch")

        def tensor(data, dtype=None):
            return _TorchTensor(data)

        def cat(items, dim=-1):
            return _TorchTensor(np.concatenate([item.numpy() for item in items], axis=dim))

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_mod.tensor = tensor
        torch_mod.cat = cat
        torch_mod.no_grad = _NoGrad
        torch_mod.float16 = "float16"
        torch_mod.float32 = "float32"
        torch_mod.bfloat16 = "bfloat16"
        torch_mod.long = "long"
        torch_mod.threads = None

        def set_num_threads(count):
            torch_mod.threads = count

        torch_mod.set_num_threads = set_num_threads
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch_mod.npu = types.SimpleNamespace(is_available=lambda: npu)
        torch_mod.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        )
        sys.modules["torch"] = torch_mod
        return torch_mod

    class _Causal:
        def __init__(self, with_past=True):
            self.with_past = with_past
            self.eval_called = False
            self.lengths = []

        def eval(self):
            self.eval_called = True
            return self

        def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
            length = int(np.asarray(input_ids.numpy()).shape[-1])
            self.lengths.append(length)
            logits = np.zeros((1, length, 6), dtype=np.float32)
            logits[:, -1, 4] = 8
            past = ("past", length) if self.with_past else None
            return types.SimpleNamespace(logits=_TorchTensor(logits), past_key_values=past)

    class _WordTokenizer:
        eos_token_id = None

        def __call__(self, prompt, return_tensors="pt"):
            assert prompt == "Hello"
            assert return_tensors == "pt"
            torch_mod = sys.modules["torch"]
            return {
                "input_ids": torch_mod.tensor([[1, 2, 3]]),
                "attention_mask": torch_mod.tensor([[1, 1, 1]]),
            }

        def decode(self, ids):
            return ",".join(str(int(token)) for token in ids)

    class _Transformers:
        def __init__(self, model):
            self.model = model
            self.loads = []
            self.tokenizers = []

            class _AutoModel:
                def from_pretrained(_self, path, **kwargs):
                    self.loads.append((path, kwargs))
                    return model

            class _AutoTokenizer:
                def from_pretrained(_self, path, **kwargs):
                    self.tokenizers.append(path)
                    return _WordTokenizer()

            self.AutoModelForCausalLM = _AutoModel()
            self.AutoTokenizer = _AutoTokenizer()

    import pipeline_utils

    assert pipeline_utils.TorchBackend is TorchBackend
    assert pipeline_utils.VLLMBackend is VLLMBackend

    hidden = _hide_modules("torch", "transformers", "vllm")
    model_dir = tempfile.mkdtemp(prefix="torch-load-")
    try:
        torch_mod = _install_torch(cuda=False)
        causal = _Causal()
        transformers_mod = _Transformers(causal)
        sys.modules["transformers"] = transformers_mod
        cpu_torch = TorchBackend(BackendConfig(
            device=DeviceType.CPU,
            n_threads=3,
            low_memory=True,
        ))
        try:
            cpu_torch.generate("Hello")
            raise AssertionError("torch generate before load was accepted")
        except RuntimeError as exc:
            assert "load()" in str(exc)
        cpu_torch.load(model_dir)
        assert cpu_torch.is_loaded is True
        assert causal.eval_called is True
        assert torch_mod.threads == 3
        assert transformers_mod.loads[0][0] == model_dir
        assert transformers_mod.tokenizers == [model_dir]
        assert transformers_mod.loads[0][1]["device_map"] == "cpu"
        assert transformers_mod.loads[0][1]["low_cpu_mem_usage"] is True
        assert "torch_dtype" not in transformers_mod.loads[0][1]
        assert cpu_torch.get_memory_usage()["backend"] == "torch"
        assert cpu_torch.get_memory_usage()["device"] == "cpu"
        try:
            cpu_torch.generate("Hello", max_tokens=-1)
            raise AssertionError("a negative torch max_tokens was accepted")
        except ValueError as exc:
            assert "max_tokens" in str(exc)
        streamed_torch = []
        assert cpu_torch.generate(
            "Hello",
            2,
            lambda text, index: streamed_torch.append((text, index)),
            temperature=0,
        ) == "1,2,3,4,4"
        assert streamed_torch == [("4", 0), ("4", 1)]
        assert causal.lengths == [3, 1]
        streamed_torch.clear()
        assert cpu_torch.generate(
            "Hello",
            max_tokens=0,
            temperature=0,
            stream_callback=lambda text, index: streamed_torch.append((text, index)),
        ) == "1,2,3"
        assert streamed_torch == []
        assert causal.lengths == [3, 1]

        class _EosWords(_WordTokenizer):
            eos_token_id = 4

        cpu_torch._tokenizer = _EosWords()
        causal.lengths.clear()
        assert cpu_torch.generate("Hello", max_tokens=5, temperature=0) == "1,2,3,4"
        assert causal.lengths == [3]
        cpu_torch._tokenizer = _WordTokenizer()
        causal.lengths.clear()
        assert cpu_torch.warmup(1) == {"iterations": 1, "warmed": True}
        assert causal.lengths == [3, 1]
        assert cpu_torch.warmed_up is True
        assert cpu_torch.get_memory_usage()["warmed_up"] is True
        cpu_torch.cleanup()
        assert cpu_torch.is_loaded is False
        assert cpu_torch._tokenizer is None
        assert cpu_torch.warmed_up is False

        class _EmptyWords(_WordTokenizer):
            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": sys.modules["torch"].tensor(np.zeros((1, 0), dtype=np.int64))}

        empty_words = TorchBackend(BackendConfig(device=DeviceType.CPU))
        empty_words.load(model_dir)
        empty_words._tokenizer = _EmptyWords()
        try:
            empty_words.generate("Hello", max_tokens=1, temperature=0)
            raise AssertionError("an empty torch prompt was accepted")
        except ValueError as exc:
            assert "zero tokens" in str(exc)

        class _BareCausal:
            def __init__(self):
                self.lengths = []

            def eval(self):
                return self

            def __call__(self, input_ids):
                length = int(np.asarray(input_ids.numpy()).shape[-1])
                self.lengths.append(length)
                logits = np.zeros((1, length, 6), dtype=np.float32)
                logits[:, -1, 4] = 8
                return types.SimpleNamespace(logits=_TorchTensor(logits))

        bare_model = _BareCausal()
        sys.modules["transformers"] = _Transformers(bare_model)
        bare = TorchBackend(BackendConfig(device=DeviceType.CPU))
        bare.load(model_dir)
        assert bare.generate("Hello", max_tokens=2, temperature=0) == "1,2,3,4,4"
        assert bare_model.lengths == [3, 4]

        no_past_model = _Causal(with_past=False)
        sys.modules["transformers"] = _Transformers(no_past_model)
        no_past = TorchBackend(BackendConfig(device=DeviceType.CPU))
        no_past.load(model_dir)
        assert no_past.generate("Hello", max_tokens=2, temperature=0) == "1,2,3,4,4"
        assert no_past_model.lengths == [3, 4]

        weight_file = os.path.join(model_dir, "model.safetensors")
        with open(weight_file, "wb") as handle:
            handle.write(b"")
        file_model = _Causal()
        file_tf = _Transformers(file_model)
        sys.modules["transformers"] = file_tf
        file_backend = TorchBackend(BackendConfig(device=DeviceType.CPU))
        file_backend.load(weight_file)
        assert file_tf.loads[0][0] == model_dir

        _install_torch(cuda=True)
        gpu_tf = _Transformers(_Causal())
        sys.modules["transformers"] = gpu_tf
        gpu = TorchBackend(BackendConfig(device=DeviceType.GPU))
        gpu.load(model_dir)
        assert gpu_tf.loads[0][1]["device_map"] == "cuda"
        assert gpu_tf.loads[0][1]["torch_dtype"] == "float16"

        _install_torch(mps=True)
        mps_tf = _Transformers(_Causal())
        sys.modules["transformers"] = mps_tf
        mps_backend = TorchBackend(BackendConfig(device=DeviceType.GPU))
        mps_backend.load(model_dir)
        assert mps_tf.loads[0][1]["device_map"] == "mps"

        _install_torch(cuda=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback_tf = _Transformers(_Causal())
            sys.modules["transformers"] = fallback_tf
            fallback = TorchBackend(BackendConfig(device=DeviceType.GPU))
            fallback.load(model_dir)
        assert fallback_tf.loads[0][1]["device_map"] == "cpu"
        assert any("CUDA or MPS" in str(item.message) for item in caught)

        _install_torch(npu=True)
        npu_tf = _Transformers(_Causal())
        sys.modules["transformers"] = npu_tf
        npu_torch = TorchBackend(BackendConfig(device=DeviceType.NPU))
        npu_torch.load(model_dir)
        assert npu_tf.loads[0][1]["device_map"] == "npu"
        assert npu_tf.loads[0][1]["torch_dtype"] == "float16"

        _install_torch(cuda=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            missing_npu_tf = _Transformers(_Causal())
            sys.modules["transformers"] = missing_npu_tf
            missing_npu = TorchBackend(BackendConfig(device=DeviceType.NPU))
            missing_npu.load(model_dir)
        assert missing_npu_tf.loads[0][1]["device_map"] == "cpu"
        assert any("no NPU" in str(item.message) for item in caught)

        _install_torch(cuda=True)
        auto_tf = _Transformers(_Causal())
        sys.modules["transformers"] = auto_tf
        auto_torch = TorchBackend(BackendConfig(device=DeviceType.AUTO))
        auto_torch.load(model_dir)
        assert auto_tf.loads[0][1]["device_map"] == "auto"
        assert auto_tf.loads[0][1]["torch_dtype"] == "float16"

        _install_torch()
        int8_tf = _Transformers(_Causal())
        sys.modules["transformers"] = int8_tf
        int8 = TorchBackend(BackendConfig(device=DeviceType.CPU, quantization="int8"))
        int8.load(model_dir)
        assert int8_tf.loads[0][1]["load_in_8bit"] is True
        int4_tf = _Transformers(_Causal())
        sys.modules["transformers"] = int4_tf
        int4 = TorchBackend(BackendConfig(device=DeviceType.CPU, quantization="int4"))
        int4.load(model_dir)
        assert int4_tf.loads[0][1]["load_in_4bit"] is True
        bf_tf = _Transformers(_Causal())
        sys.modules["transformers"] = bf_tf
        bf16 = TorchBackend(BackendConfig(device=DeviceType.CPU, quantization="bf16"))
        bf16.load(model_dir)
        assert bf_tf.loads[0][1]["torch_dtype"] == "bfloat16"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llama_quant_tf = _Transformers(_Causal())
            sys.modules["transformers"] = llama_quant_tf
            llama_quant = TorchBackend(BackendConfig(
                device=DeviceType.CPU,
                quantization="Q4_K_M",
            ))
            llama_quant.load(model_dir)
        assert "torch_dtype" not in llama_quant_tf.loads[0][1]
        assert "load_in_4bit" not in llama_quant_tf.loads[0][1]
        assert any("Q4_K_M" in str(item.message) for item in caught)

        class _SamplingParams:
            def __init__(self, max_tokens=None, temperature=None, top_p=None):
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.top_p = top_p

        class _LLM:
            def __init__(
                self,
                model=None,
                max_model_len=None,
                device=None,
                dtype=None,
                quantization=None,
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
            ):
                self.model = model
                self.max_model_len = max_model_len
                self.device = device
                self.dtype = dtype
                self.quantization = quantization
                self.tensor_parallel_size = tensor_parallel_size
                self.gpu_memory_utilization = gpu_memory_utilization
                self.calls = []

            def get_tokenizer(self):
                return _WordTokenizer()

            def generate(self, prompts, params):
                self.calls.append((
                    list(prompts),
                    params.max_tokens,
                    params.temperature,
                    params.top_p,
                ))
                completion = types.SimpleNamespace(text="hi!", token_ids=[7, 8])
                return [types.SimpleNamespace(outputs=[completion])]

        class _NoTokLLM(_LLM):
            def get_tokenizer(self):
                return None

        class _NarrowLLM:
            def __init__(self, model):
                self.source = model

            def generate(self, prompts, params):
                return ["plain"]

        vllm_mod = types.ModuleType("vllm")
        vllm_mod.SamplingParams = _SamplingParams
        vllm_mod.LLM = _LLM
        sys.modules["vllm"] = vllm_mod
        engine = VLLMBackend(BackendConfig(
            device=DeviceType.GPU,
            quantization="awq",
            low_memory=True,
            n_ctx=128,
        ))
        try:
            engine.generate("Hello")
            raise AssertionError("vLLM generate before load was accepted")
        except RuntimeError as exc:
            assert "load()" in str(exc)
        engine.load(model_dir)
        assert engine.model.model == model_dir
        assert engine.model.device == "cuda"
        assert engine.model.dtype is None
        assert engine.model.quantization == "awq"
        assert engine.model.max_model_len == 128
        assert engine.model.tensor_parallel_size == 1
        assert engine.model.gpu_memory_utilization == 0.5
        vllm_pieces = []
        assert engine.generate(
            "Hello",
            max_tokens=4,
            temperature=0,
            top_p=0.3,
            stream_callback=lambda text, index: vllm_pieces.append((text, index)),
        ) == "hi!"
        assert vllm_pieces == [("7", 0), ("8", 1)]
        assert engine.model.calls == [(["Hello"], 4, 0.0, 0.3)]
        assert engine.generate(
            "Hello",
            max_tokens=0,
            stream_callback=lambda text, index: vllm_pieces.append((text, index)),
        ) == ""
        assert vllm_pieces == [("7", 0), ("8", 1)]
        assert engine.model.calls == [(["Hello"], 4, 0.0, 0.3)]
        try:
            engine.generate("Hello", max_tokens=-2)
            raise AssertionError("a negative vLLM max_tokens was accepted")
        except ValueError as exc:
            assert "max_tokens" in str(exc)
        assert engine.warmup(2) == {"iterations": 2, "warmed": True}
        assert engine.model.calls[1:] == [
            (["Hello"], 2, 0.0, 0.9),
            (["Hello"], 2, 0.0, 0.9),
        ]
        assert engine.get_memory_usage()["backend"] == "vllm"
        assert engine.get_memory_usage()["device"] == "cuda"
        assert engine.get_memory_usage()["warmed_up"] is True
        engine.cleanup()
        assert engine.is_loaded is False
        assert engine.warmed_up is False

        vllm_mod.LLM = _NoTokLLM
        silent = VLLMBackend(BackendConfig(device=DeviceType.CPU, quantization="fp16"))
        silent.load(weight_file)
        assert silent.model.model == model_dir
        assert silent.model.device == "cpu"
        assert silent.model.dtype == "float16"
        one_piece = []
        assert silent.generate(
            "Hello",
            stream_callback=lambda text, index: one_piece.append((text, index)),
        ) == "hi!"
        assert one_piece == [("hi!", 0)]

        vllm_mod.LLM = _LLM
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            npu_vllm = VLLMBackend(BackendConfig(device=DeviceType.NPU))
            npu_vllm.load(model_dir)
        assert npu_vllm.model.device == "cpu"
        assert any("NPU" in str(item.message) for item in caught)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            auto_vllm = VLLMBackend(BackendConfig(
                device=DeviceType.AUTO,
                quantization="nope",
            ))
            auto_vllm.load(model_dir)
        assert auto_vllm.model.device is None
        assert auto_vllm._device == "auto"
        assert any("nope" in str(item.message) for item in caught)

        vllm_mod.LLM = _NarrowLLM
        narrow = VLLMBackend(BackendConfig(
            device=DeviceType.GPU,
            low_memory=True,
            quantization="gptq",
        ))
        narrow.load(model_dir)
        assert narrow.model.source == model_dir
        narrow_pieces = []
        assert narrow.generate(
            "Hello",
            stream_callback=lambda text, index: narrow_pieces.append((text, index)),
        ) == "plain"
        assert narrow_pieces == [("plain", 0)]

        vllm_mod.LLM = None
        try:
            VLLMBackend().load(model_dir)
            raise AssertionError("vLLM load without LLM was accepted")
        except ImportError as exc:
            assert "vllm.LLM" in str(exc)
    finally:
        _restore_modules(hidden)
        weight_path = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(weight_path):
            os.remove(weight_path)
        os.rmdir(model_dir)

    print("  generate: split pipeline and monolithic prefill/decode")
    print("  providers: NPU, GPU, CPU, and auto-detect")
    print("  hybrid: prefill/decode, NPU layer range, MoE router and experts")
    print("backend module: OK\n")
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

    try:
        results.append(("backend", test_backend()))
    except Exception as e:
        print(f"backend: FAILED - {e}")
        results.append(("backend", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

