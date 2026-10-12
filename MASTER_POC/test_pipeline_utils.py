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
    
    from pipeline_utils.kv_cache import (
        KVCacheManager,
        KVCacheBackend,
        KVCachePolicy,
        KVUpdateMode,
    )
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
        spatial = backend in {
            KVCacheBackend.VITISAI,
            KVCacheBackend.RYZENAI,
            KVCacheBackend.HYBRID_NPU_IGPU,
        }
        resident.update_from_present(0, present, present.copy())
        resident.update_from_present(1, present, present.copy())
        assert resident.get_sequence_length() == 4
        if spatial:
            assert resident.update_mode is KVUpdateMode.CONCAT_SWAP
            assert int(key.ort_value.data_ptr()) != before
            assert resident.get_memory_usage()["shadow_bytes"] > 0
        else:
            assert resident.update_mode is KVUpdateMode.SLOT_WRITE
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
        if spatial:
            shadow_key, shadow_value = resident.shadow_ort_values(0)
            assert bound_io.outputs["present.0.key"] is shadow_key
            assert bound_io.outputs["present.0.value"] is shadow_value
            assert shadow_key is not device_key
        else:
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
    assert device_phase.update_mode is KVUpdateMode.CONCAT_SWAP
    assert int(device_phase.get_device_tensors(0)[0].data_ptr()) != device_ptr
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

    swapped = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float16,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=8,
    )
    swapped.update_from_present(0, present, present.copy())
    active_ptr = int(swapped.get_device_tensors(0)[0].data_ptr())
    shadow_buffer = swapped._cache[0]["key_shadow"]
    staged = np.zeros((1, 2, 5, 8), dtype=np.float16)
    staged[:, :, :4, :] = present
    staged[:, :, 4, :] = 9
    shadow_host = shadow_buffer.ort_value.numpy()
    shadow_host[:, :, :5, :] = staged
    if int(shadow_host.ctypes.data) != int(shadow_buffer.ort_value.data_ptr()):
        shadow_buffer.ort_value.update_inplace(np.ascontiguousarray(shadow_host))
    assert swapped.commit_present_swap(0, 5) == 5
    assert int(swapped.get_device_tensors(0)[0].data_ptr()) != active_ptr
    assert np.array_equal(swapped.get_past_tensors(0)[0].to_numpy()[:, :, :5, :], staged)
    assert int(swapped.attention_mask()[0, 4]) == 1
    assert int(swapped.attention_mask()[0, 5]) == 0

    cpu_swap = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=np.float16,
        max_seq_len=8,
        update_mode="concat_swap",
    )
    assert cpu_swap.update_mode is KVUpdateMode.CONCAT_SWAP
    cpu_key = cpu_swap.get_past_tensors(0)[0]
    cpu_swap.update_from_present(0, present, present.copy())
    assert cpu_swap.get_past_tensors(0)[0] is not cpu_key
    assert np.array_equal(cpu_swap.get_past_tensors(0)[0][:, :, :4, :], present)
    cpu_token = np.full((1, 2, 1, 8), 7, dtype=np.float16)
    cpu_swap.update_from_present(0, cpu_token, cpu_token.copy())
    assert cpu_swap.get_past_tensors(0)[0] is cpu_key
    assert np.array_equal(cpu_key[:, :, :4, :], present)
    assert np.array_equal(cpu_key[:, :, 4, :], cpu_token[:, :, 0, :])

    from chunking_utils.onnx.kv_concat import (
        assert_no_kv_scatter,
        static_kv_concat_recipe,
    )

    prefix_recipe = static_kv_concat_recipe(8, 1)
    assert prefix_recipe["layout"] == "host_prefix"
    assert prefix_recipe["output_seq"] == 1
    assert "ScatterND" not in prefix_recipe["op_types"]
    window_recipe = static_kv_concat_recipe(8, 1, layout="fixed_window")
    assert window_recipe["op_types"] == ["Slice", "Concat"]
    assert window_recipe["output_seq"] == 8
    assert window_recipe["nodes"][0]["begin"][2] == 1
    assert window_recipe["nodes"][0]["end"][2] == 8
    try:
        assert_no_kv_scatter(["Concat", "ScatterND"])
        raise AssertionError("ScatterND was accepted in a spatial KV update")
    except ValueError as exc:
        assert "ScatterND" in str(exc)
    try:
        static_kv_concat_recipe(8, 1, layout="scatter")
        raise AssertionError("unknown concat layout was accepted")
    except ValueError as exc:
        assert "scatter" in str(exc)
    print("  concat_swap: shadow present, pointer swap, static concat recipe")

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
        load_generation_metadata,
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
    assert config.kv_cache_dtype is None
    assert config.kv_cache_layout == "BNSD"
    assert config.kv_cache_max_len is None
    assert config.kv_cache_page_size is None
    assert config.use_sliding_window is False
    assert d["kv_cache_dtype"] is None
    assert d["kv_cache_layout"] == "BNSD"
    assert d["kv_cache_max_len"] is None
    assert config.execution_hints["npu_safe"] is True
    assert config.execution_hints["recommended_npu_mode"] == "monolithic"
    assert config.execution_hints["recommended_prefill_device"] == "npu"
    assert config.execution_hints["recommended_decode_device"] == "igpu"
    assert config.execution_hints["recommended_execution"] == "hybrid"
    assert config.execution_hints["sequential_split_safe"] is False
    assert config.execution_hints["supports_static_kv_cache"] is False
    assert d["execution_hints"] == config.execution_hints
    assert d["execution_hints"] is not config.execution_hints
    wide = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096)
    assert wide.execution_hints["supports_static_kv_cache"] is True
    assert wide.execution_hints["recommended_decode_device"] == "igpu"
    cpu_only = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, dtype=np.dtype(np.float32))
    assert cpu_only.execution_hints["npu_safe"] is False
    assert cpu_only.execution_hints["recommended_prefill_device"] == "igpu"
    assert cpu_only.execution_hints["recommended_execution"] == "monolithic"
    assert cpu_only.execution_hints["sequential_split_safe"] is True
    assert cpu_only.execution_hints["recommended_npu_mode"] == "monolithic"
    int4 = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, weight_dtype="int4")
    assert int4.execution_hints["npu_safe"] is True
    fp32_weights = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, weight_dtype="float32")
    assert fp32_weights.execution_hints["npu_safe"] is False
    sliding = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, use_sliding_window=True)
    assert sliding.execution_hints["supports_static_kv_cache"] is True
    pinned = ModelConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        execution_hints={"recommended_decode_device": "CPU"},
    )
    assert pinned.execution_hints["recommended_decode_device"] == "cpu"
    assert pinned.execution_hints["recommended_prefill_device"] == "npu"
    assert pinned.execution_hints["npu_safe"] is True
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, execution_hints={"turbo": True})
        raise AssertionError("an unknown execution hint must be rejected")
    except ValueError as exc:
        assert "turbo" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, execution_hints="no")
        raise AssertionError("a non-dict execution hint must be rejected")
    except ValueError as exc:
        assert "execution_hints" in str(exc)
    broken = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8)
    broken.execution_hints = {}
    try:
        broken.validate()
        raise AssertionError("an empty execution hint dict must be rejected")
    except ValueError as exc:
        assert "execution_hints" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, kv_cache_max_len=0).validate()
        raise AssertionError("a zero kv cache length must be rejected")
    except ValueError as exc:
        assert "kv_cache_max_len" in str(exc)
    try:
        ModelConfig(num_layers=1, num_kv_heads=1, head_dim=8, kv_cache_layout="").validate()
        raise AssertionError("an empty kv layout must be rejected")
    except ValueError as exc:
        assert "kv_cache_layout" in str(exc)
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
    from pipeline_utils import load_generation_metadata as exported_generation
    assert exported_loader is load_model_config
    assert exported_validate is validate_config_against_onnx
    assert exported_generation is load_generation_metadata

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
        assert genai.kv_cache_dtype is None
        assert genai.kv_cache_layout == "BNSD"
        assert genai.kv_cache_max_len is None
        assert genai.use_sliding_window is False

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
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("require_all_fields must reject zero heads")
        except ValueError as exc:
            message = str(exc)
            assert "num_kv_heads=0" in message
            assert "Refusing defaults" in message
            assert "num_kv_heads must be positive" not in message

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "architecture": {"num_kv_heads": 4, "head_dim": -16},
        })
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("require_all_fields must reject a negative head_dim")
        except ValueError as exc:
            message = str(exc)
            assert "head_dim=-16" in message
            assert "Refusing defaults" in message
        try:
            load_model_config(directory)
            raise AssertionError("a negative head_dim must be rejected")
        except ValueError as exc:
            assert "head_dim must be positive" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "config.json", {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 384,
            "head_dim": -8,
            "model_type": "llama",
        })
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("require_all_fields must reject a negative Hugging Face head_dim")
        except ValueError as exc:
            assert "head_dim=-8" in str(exc)
            assert "Refusing defaults" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": -2,
                    "head_size": 96,
                    "hidden_size": 384,
                    "num_attention_heads": 4,
                },
            },
        })
        try:
            load_model_config(directory, require_all_fields=True)
            raise AssertionError("require_all_fields must reject negative GenAI heads")
        except ValueError as exc:
            assert "num_kv_heads=-2" in str(exc)
            assert "Refusing defaults" in str(exc)

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
        assert info_moe.use_sliding_window is True

    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["max_position_embeddings"] = 4096
        _write_json(directory, "config.json", payload)
        context_only = load_model_config(directory)
        assert context_only.max_seq_len == 4096
        assert context_only.kv_cache_max_len is None
        assert context_only.kv_cache_dtype is None
        assert context_only.kv_cache_layout == "BNSD"
        assert context_only.use_sliding_window is False
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({
            "kv_cache_dtype": "float16",
            "kv_cache_layout": "bnsd",
            "kv_cache_max_len": 0,
            "kv_cache_max_length": 32,
            "kv_cache_page_size": 16,
            "use_sliding_window": True,
        })
        _write_json(directory, "config.json", payload)
        cached = load_model_config(directory)
        assert cached.kv_cache_dtype == "fp16"
        assert cached.kv_cache_layout == "BNSD"
        assert cached.kv_cache_max_len == 32
        assert cached.kv_cache_page_size == 16
        assert cached.use_sliding_window is True
        assert cached.to_dict()["kv_cache_layout"] == "BNSD"
        from pipeline_utils.kv_cache import KVCacheManager
        linked = KVCacheManager(
            cached.num_layers,
            cached.num_kv_heads,
            cached.head_dim,
            dtype=cached.dtype,
            max_seq_len=cached.kv_cache_max_len,
            kv_quantization=cached.kv_cache_dtype,
            block_size=cached.kv_cache_page_size,
            policy="sliding_window",
            window_size=cached.kv_cache_max_len,
        )
        assert linked.max_seq_len == 32
        assert linked.block_size == 16
        assert linked.kv_quantization == "fp16"
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload.update({"sliding_window": 4096, "use_sliding_window": False})
        _write_json(directory, "config.json", payload)
        forced = load_model_config(directory)
        assert forced.attention_variant == "sliding_window"
        assert forced.use_sliding_window is False
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["sliding_window"] = 2048
        _write_json(directory, "config.json", payload)
        implied = load_model_config(directory)
        assert implied.use_sliding_window is True
        assert implied.kv_cache_max_len is None
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["kv_cache_dtype"] = "nope"
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("an unknown kv cache dtype must be rejected")
        except ValueError as exc:
            assert "kv_cache_dtype" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["kv_cache_page_size"] = -8
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a negative kv page size must be rejected")
        except ValueError as exc:
            assert "kv_cache_page_size" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["use_sliding_window"] = "yes"
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a non-boolean sliding flag must be rejected")
        except ValueError as exc:
            assert "use_sliding_window" in str(exc)
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
                    "kv_cache_dtype": "bf16",
                    "kv_cache_layout": "BSHD",
                    "kv_cache_max_len": 128,
                },
            },
        })
        genai_kv = load_model_config(directory)
        assert genai_kv.kv_cache_dtype == "bf16"
        assert genai_kv.kv_cache_layout == "BSHD"
        assert genai_kv.kv_cache_max_len == 128
        assert genai_kv.kv_cache_page_size is None
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 4,
            "num_kv_heads": 4,
            "head_dim": 96,
            "kv_cache": {
                "dtype": "int8",
                "layout": "bnsh",
                "max_len": 64,
                "block_size": 8,
            },
        })
        info_kv = load_model_config(directory, require_all_fields=True)
        assert info_kv.kv_cache_dtype == "int8"
        assert info_kv.kv_cache_layout == "BNSD"
        assert info_kv.kv_cache_max_len == 64
        assert info_kv.kv_cache_page_size == 8
        assert info_kv.use_sliding_window is False

    try:
        load_generation_metadata(os.path.join(tempfile.gettempdir(), "missing-generation-dir"))
        raise AssertionError("a missing model directory must be rejected")
    except FileNotFoundError as exc:
        assert "generation metadata" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        empty = load_generation_metadata(directory)
        assert empty["eos_token_id"] is None
        assert empty["bos_token_id"] is None
        assert empty["pad_token_id"] is None
        assert empty["chat_template"] is None
        assert empty["chat_templates"] == {}
        assert empty["source_files"] == []
        assert empty["max_new_tokens"] is None
        assert empty["temperature"] is None
        assert empty["do_sample"] is None
        _write_json(directory, "config.json", dict(hf_base))
        assert load_generation_metadata(directory)["source_files"] == []
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {
            "eos_token_id": [2, 128001],
            "bos_token_id": 1,
            "pad_token_id": 0,
            "temperature": 0,
            "top_p": 0.5,
            "top_k": 20,
            "do_sample": True,
            "max_new_tokens": 32,
            "repetition_penalty": 1.0,
        })
        _write_json(directory, "tokenizer_config.json", {
            "eos_token_id": 99,
            "chat_template": "CHAT",
        })
        merged = load_generation_metadata(directory)
        assert merged["eos_token_id"] == [2, 128001]
        assert merged["bos_token_id"] == 1
        assert merged["pad_token_id"] == 0
        assert merged["chat_template"] == "CHAT"
        assert merged["chat_templates"] == {"default": "CHAT"}
        assert merged["temperature"] == 0.0
        assert merged["top_p"] == 0.5
        assert merged["top_k"] == 20
        assert merged["do_sample"] is True
        assert merged["max_new_tokens"] == 32
        assert merged["repetition_penalty"] == 1.0
        assert merged["source_files"] == ["generation_config.json", "tokenizer_config.json"]
        with open(os.path.join(directory, "model.onnx"), "wb") as handle:
            handle.write(b"onnx")
        from_file = load_generation_metadata(os.path.join(directory, "model.onnx"))
        assert from_file["eos_token_id"] == [2, 128001]
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {
            "eos_token_id": None,
            "bos_token_id": 7,
        })
        _write_json(directory, "tokenizer_config.json", {
            "eos_token": "<|endoftext|>",
            "bos_token": {"content": "<|begin|>", "lstrip": False},
            "pad_token": "<|endoftext|>",
            "added_tokens_decoder": {
                "128001": {"content": "<|endoftext|>", "special": True},
                "128000": {"content": "<|begin|>", "special": True},
            },
            "chat_template": [
                {"name": "default", "template": "DEFAULT"},
                {"name": "tool", "template": "TOOL"},
            ],
        })
        resolved = load_generation_metadata(directory)
        assert resolved["eos_token_id"] == 128001
        assert resolved["bos_token_id"] == 7
        assert resolved["pad_token_id"] == 128001
        assert resolved["chat_template"] == "DEFAULT"
        assert resolved["chat_templates"]["tool"] == "TOOL"
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "tokenizer_config.json", {"chat_template": "  "})
        assert load_generation_metadata(directory)["chat_template"] is None
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {"eos_token_id": "nope"})
        try:
            load_generation_metadata(directory)
            raise AssertionError("a non-integer eos id must be rejected")
        except ValueError as exc:
            assert "eos_token_id" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {"pad_token_id": -1})
        try:
            load_generation_metadata(directory)
            raise AssertionError("a negative pad id must be rejected")
        except ValueError as exc:
            assert "pad_token_id" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "tokenizer_config.json", {"bos_token_id": True})
        try:
            load_generation_metadata(directory)
            raise AssertionError("a boolean bos id must be rejected")
        except ValueError as exc:
            assert "bos_token_id" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {"do_sample": "yes"})
        try:
            load_generation_metadata(directory)
            raise AssertionError("a non-boolean do_sample must be rejected")
        except ValueError as exc:
            assert "do_sample" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "generation_config.json", {"top_p": 1.5})
        try:
            load_generation_metadata(directory)
            raise AssertionError("top_p above 1 must be rejected")
        except ValueError as exc:
            assert "top_p" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
            handle.write("{")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            skipped = load_generation_metadata(directory)
        assert skipped["source_files"] == []
        assert any("tokenizer_config.json" in str(item.message) for item in caught)

    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["max_position_embeddings"] = 2048
        payload["execution_hints"] = {"recommended_decode_device": "i-gpu"}
        _write_json(directory, "config.json", payload)
        hinted = load_model_config(directory)
        assert hinted.execution_hints["npu_safe"] is True
        assert hinted.execution_hints["recommended_npu_mode"] == "monolithic"
        assert hinted.execution_hints["recommended_prefill_device"] == "npu"
        assert hinted.execution_hints["recommended_decode_device"] == "igpu"
        assert hinted.execution_hints["recommended_execution"] == "hybrid"
        assert hinted.execution_hints["sequential_split_safe"] is False
        assert hinted.execution_hints["supports_static_kv_cache"] is True
        assert hinted.max_seq_len == 2048
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["torch_dtype"] = "float32"
        _write_json(directory, "config.json", payload)
        wide_compute = load_model_config(directory)
        assert wide_compute.execution_hints["npu_safe"] is False
        assert wide_compute.execution_hints["recommended_prefill_device"] == "igpu"
        assert wide_compute.execution_hints["recommended_execution"] == "monolithic"
        assert wide_compute.execution_hints["sequential_split_safe"] is True
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "genai_config.json", {
            "model": {
                "type": "llama",
                "vocab_size": 10,
                "decoder": {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "head_size": 64,
                    "hidden_size": 256,
                    "num_attention_heads": 4,
                    "num_local_experts": 8,
                    "num_experts_per_tok": 2,
                    "kv_cache_max_len": 128,
                },
            },
        })
        genai_hints = load_model_config(directory)
        assert genai_hints.is_moe is True
        assert genai_hints.execution_hints["npu_safe"] is True
        assert genai_hints.execution_hints["recommended_npu_mode"] == "monolithic"
        assert genai_hints.execution_hints["recommended_execution"] == "hybrid"
        assert genai_hints.execution_hints["sequential_split_safe"] is False
        assert genai_hints.execution_hints["supports_static_kv_cache"] is True
        assert genai_hints.execution_hints["recommended_decode_device"] == "igpu"
    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 64,
            "amd_execution_hints": {"recommended_prefill_device": "cpu"},
        })
        info_hints = load_model_config(directory, require_all_fields=True)
        assert info_hints.execution_hints["recommended_prefill_device"] == "cpu"
        assert info_hints.execution_hints["recommended_decode_device"] == "igpu"
        assert info_hints.execution_hints["supports_static_kv_cache"] is False
    with tempfile.TemporaryDirectory() as directory:
        payload = dict(hf_base)
        payload["execution_hints"] = []
        _write_json(directory, "config.json", payload)
        try:
            load_model_config(directory)
            raise AssertionError("a list of execution hints must be rejected")
        except ValueError as exc:
            assert "execution_hints" in str(exc)

    print("  kv shape: refuse silent 8/128, infer 4 x 96 from ONNX")
    print("  require_all_fields: non-positive KV shape raises")
    print("  max_seq_len: max_position_embeddings, sequence aliases, unset stays 0")
    print("  quantization: AWQ, GPTQ, Quark, MatMulNBits, unset stays empty")
    print("  dtype: torch_dtype, dtype, decoder dtype, missing stays float16")
    print("  architecture: moe experts, hybrid attention, dense stays empty")
    print("  kv cache: dtype, BNSD layout, max len, page size, sliding window")
    print("  generation metadata: eos, bos, pad, chat template")
    print("  execution hints: monolithic NPU, iGPU decode, sequential split unsafe")
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
        ExecutionPolicy.SEQUENTIAL,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["RyzenAIExecutionProvider"],
        ExecutionPolicy.PERSISTENT,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["VitisAIExecutionProvider"],
        ExecutionPolicy.MONOLITHIC,
        split_layers,
    )
    _expect_npu_split_rejected(
        ["RyzenAIExecutionProvider"],
        "sequential",
        monolithic,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.MONOLITHIC,
        ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        monolithic,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.HYBRID_NPU_IGPU,
        ["VitisAIExecutionProvider", "DmlExecutionProvider"],
        split_layers,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.MOE_ROUTING,
        ["RyzenAIExecutionProvider", "ROCmExecutionProvider"],
        split_layers,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.SEQUENTIAL,
        ["CPUExecutionProvider"],
        split_layers,
    )
    _expect_npu_split_rejected(
        ["VitisAIExecutionProvider"],
        ExecutionPolicy.SPECULATIVE,
        split_layers,
    )
    validate_npu_execution_policy(
        ExecutionPolicy.SPECULATIVE,
        ["VitisAIExecutionProvider"],
        ["draft", "verify"],
    )
    validate_npu_execution_policy(
        ExecutionPolicy.PHASE_SPLIT,
        ["VitisAIExecutionProvider", "DmlExecutionProvider"],
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
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.SEQUENTIAL,
        ),
        model_info=split_info,
    )
    assert "layer_0" in cpu_manager
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_npu_execution_policy(
            ExecutionPolicy.MOE_ROUTING,
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
        execution_policy=ExecutionPolicy.PERSISTENT,
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
        execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.SEQUENTIAL,
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
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.MOE_ROUTING,
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
            execution_policy=ExecutionPolicy.MONOLITHIC,
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
            execution_policy=ExecutionPolicy.PHASE_SPLIT,
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
        assert "phase_split" in str(exc)

    speculative = SessionManager(
        ".",
        SessionConfig(
            providers=["VitisAIExecutionProvider", "DmlExecutionProvider"],
            execution_policy=ExecutionPolicy.SPECULATIVE,
        ),
        model_info=_plan_info("draft", "verify"),
        available_providers=installed,
    )
    try:
        speculative.load_from_plan(ExecutionPlan(steps=[
            ExecutionStep(
                "draft", "DmlExecutionProvider", "igpu", "speculative_draft",
                [], ["hidden"],
            ),
        ]))
        raise AssertionError("speculative plan without verification was accepted")
    except ValueError as exc:
        assert "verification" in str(exc)
    speculative.load_from_plan(ExecutionPlan(steps=[
        ExecutionStep(
            "draft", "DmlExecutionProvider", "igpu", "draft",
            [], ["hidden"],
        ),
        ExecutionStep(
            "verify", "VitisAIExecutionProvider", "npu", "verify",
            ["draft"], ["logits"],
        ),
    ]))
    assert speculative.get_session_info("draft").phase == "speculative_draft"
    assert speculative.get_session_info("verify").phase == "verification"

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
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.MOE_ROUTING,
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
            execution_policy=ExecutionPolicy.HYBRID_NPU_IGPU,
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
            execution_policy=ExecutionPolicy.SEQUENTIAL,
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
    assert defaults.resolved_execution_policy() == ExecutionPolicy.MONOLITHIC
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
    assert plan["execution_policy"] == "monolithic"
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

    import json
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

    class _ListEosTokenizer(_Tokenizer):
        eos_token_id = [4, 9]

    monolithic._tokenizer = _ListEosTokenizer()
    assert monolithic.generate("Hello", max_tokens=5, temperature=0) == "1,2,3,4"
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "generation_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"eos_token_id": 4}, handle)
        monolithic.model_path = directory
        monolithic._tokenizer = _Tokenizer()
        assert monolithic.generate("Hello", max_tokens=5, temperature=0) == "1,2,3,4"
        monolithic.model_path = None
        monolithic._tokenizer = _EosTokenizer()

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
        from chunking_utils.common.types import ModelFormat as canonical_format
        from chunking_utils import ModelFormat as package_format
        from chunking_utils import auto_detect_format, split_model
        assert ModelFormat is canonical_format is package_format
        assert canonical_format.__module__ == "chunking_utils.common.types"
        assert [item.value for item in ModelFormat] == [
            "onnx",
            "safetensors",
            "gguf",
            "pytorch_bin",
        ]
        assert not hasattr(ModelFormat, "PYTORCH")
        assert ModelFormat.ONNX == "onnx"
        assert ModelFormat.SAFETENSORS == "safetensors"
        assert ModelFormat.GGUF == "gguf"
        assert ModelFormat.PYTORCH_BIN == "pytorch_bin"
        assert auto_detect_format(bin_path) is ModelFormat.PYTORCH_BIN
        assert auto_detect_format(gguf_path) is ModelFormat.GGUF
        assert auto_detect_format(safetensors_path) is ModelFormat.SAFETENSORS
        assert auto_detect_format(weights_dir) is ModelFormat.SAFETENSORS
        try:
            split_model(gguf_path, gguf_dir)
            raise AssertionError("a gguf file was sent to a splitter")
        except ValueError as exc:
            assert "gguf" in str(exc)
        from chunking_utils.common.types import (
            DeviceType as canonical_device,
            ExecutionPolicy as canonical_policy,
            PROVIDER_FALLBACK,
            SplitConfig,
            QuantizationConfig,
            KVCachePolicy,
            KVCacheConfig,
            TensorDtype,
            ComponentInfo,
            ComponentType,
            BoundaryTensor,
            SplitResult,
            ExecutionStep,
            ExecutionPlan,
            MemoryMode,
        )
        from pipeline_utils.session_manager import ExecutionPolicy as session_policy
        assert DeviceType is canonical_device
        assert session_policy is canonical_policy
        assert [item.value for item in DeviceType] == ["cpu", "gpu", "npu", "ipu", "auto"]
        assert DeviceType.IPU == "ipu"
        assert PROVIDER_FALLBACK[DeviceType.NPU] == [
            "VitisAIExecutionProvider",
            "RyzenAIExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert PROVIDER_FALLBACK[DeviceType.IPU] == [
            "DmlExecutionProvider",
            "ROCmExecutionProvider",
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert DEVICE_PROVIDER_MAP[DeviceType.IPU] == PROVIDER_FALLBACK[DeviceType.IPU]
        plain = SplitConfig(model="weights", output="out")
        other = SplitConfig(model="weights", output="out")
        assert plain.device == "cpu"
        assert plain.providers == []
        assert plain.target_device is DeviceType.AUTO
        assert plain.execution_policy is canonical_policy.AUTO
        assert plain.resolved_providers() == [
            "VitisAIExecutionProvider",
            "RyzenAIExecutionProvider",
            "DmlExecutionProvider",
            "ROCmExecutionProvider",
            "MIGraphXExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        plain.providers.append("CPUExecutionProvider")
        assert other.providers == []
        npu_plan = SplitConfig(model="weights", output="out", target_device="npu")
        assert npu_plan.target_device is DeviceType.NPU
        assert npu_plan.resolved_providers() == PROVIDER_FALLBACK[DeviceType.NPU]
        ipu_plan = SplitConfig(model="weights", output="out", target_device=DeviceType.IPU)
        assert ipu_plan.resolved_providers() == PROVIDER_FALLBACK[DeviceType.IPU]
        hybrid_plan = SplitConfig(
            model="weights",
            output="out",
            providers=["VitisAIExecutionProvider", "DmlExecutionProvider"],
            execution_policy="hybrid_npu_igpu",
        )
        assert hybrid_plan.resolved_providers() == [
            "VitisAIExecutionProvider",
            "DmlExecutionProvider",
        ]
        assert hybrid_plan.execution_policy is canonical_policy.HYBRID_NPU_IGPU
        assert [item.value for item in canonical_policy] == [
            "auto",
            "monolithic",
            "sequential",
            "persistent",
            "hybrid_npu_igpu",
            "phase_split",
            "moe_routing",
            "speculative",
        ]
        cpu_seq = SplitConfig(
            model="weights",
            output="out",
            target_device="cpu",
            execution_policy="sequential",
        )
        assert cpu_seq.execution_policy is canonical_policy.SEQUENTIAL
        for safe_policy in ("monolithic", "hybrid_npu_igpu", "phase_split", "moe_routing", "speculative"):
            accepted = SplitConfig(
                model="weights",
                output="out",
                target_device="npu",
                execution_policy=safe_policy,
            )
            assert accepted.execution_policy.value == safe_policy
        try:
            SplitConfig(
                model="weights",
                output="out",
                target_device="npu",
                execution_policy="sequential",
            )
            raise AssertionError("sequential NPU execution was accepted")
        except ValueError as exc:
            assert "sequential" in str(exc)
        try:
            SplitConfig(
                model="weights",
                output="out",
                providers=["VitisAIExecutionProvider"],
                execution_policy="persistent",
            )
            raise AssertionError("persistent NPU execution was accepted")
        except ValueError as exc:
            assert "persistent" in str(exc)
        from chunking_utils.common import ComponentInfo as package_component
        assert package_component is ComponentInfo
        bare = ComponentInfo(component_type="embedding")
        other_component = ComponentInfo(component_type=ComponentType.NORM)
        assert bare.phase is None
        assert bare.device_preference is None
        assert bare.resident_policy == "auto"
        assert bare.execution_order == 0
        assert bare.input_names == []
        bare.input_names.append("hidden")
        assert other_component.input_names == []
        assert other_component.name == "norm"
        layer = ComponentInfo(component_type="layer", layer_id=0)
        assert layer.name == "layer_0"
        assert layer.phase is None
        prefill = ComponentInfo(
            component_type="layer",
            layer_id=1,
            file_path="layer_1.onnx",
            num_tensors=4,
            size_bytes=256,
            input_names=["hidden", "mask"],
            output_names=["hidden"],
            input_types={"hidden": "float16", "mask": "int64"},
            output_types={"hidden": "float16"},
            input_shapes={"hidden": [1, None, 16], "mask": [1, None]},
            output_shapes={"hidden": [1, None, 16]},
            external_data_files=["layer_1.onnx.data"],
            depends_on=["embedding"],
            execution_order=2,
            device_preference="xdna",
            resident_policy="resident",
            phase="prefill",
        )
        assert prefill.name == "layer_1"
        assert prefill.device_preference == "npu"
        assert prefill.phase == "prefill"
        assert prefill.input_shapes["hidden"] == [1, None, 16]
        recorded = prefill.to_dict()
        assert recorded["external_data_files"] == ["layer_1.onnx.data"]
        assert recorded["depends_on"] == ["embedding"]
        assert recorded["execution_order"] == 2
        recorded["input_names"].append("extra")
        assert prefill.input_names == ["hidden", "mask"]
        draft = ComponentInfo(
            component_type="other",
            phase="draft",
            resident_policy="transient",
            device_preference="rdna",
        )
        assert draft.phase == "speculative_draft"
        assert draft.resident_policy == "on_demand"
        assert draft.device_preference == "igpu"
        verify = ComponentInfo(component_type="other", phase="verify")
        assert verify.phase == "verification"
        try:
            ComponentInfo(component_type="embedding", input_names="hidden")
            raise AssertionError("a string of input names was accepted")
        except ValueError as exc:
            assert "input_names" in str(exc)
        try:
            ComponentInfo(component_type="embedding", phase="sideband")
            raise AssertionError("an unknown phase was accepted")
        except ValueError as exc:
            assert "sideband" in str(exc)
        try:
            ComponentInfo(
                component_type="embedding",
                input_names=["hidden"],
                input_types={"mask": "float16"},
            )
            raise AssertionError("a type for an unnamed input was accepted")
        except ValueError as exc:
            assert "input_types" in str(exc)
        try:
            ComponentInfo(component_type="layer", layer_id=0, depends_on=["layer_0"])
            raise AssertionError("a self dependency was accepted")
        except ValueError as exc:
            assert "depends_on" in str(exc)
        from chunking_utils.common import BoundaryTensor as package_boundary
        assert package_boundary is BoundaryTensor
        hidden = BoundaryTensor(
            name="hidden_states",
            dtype="float16",
            shape=[1, None, 16],
            producer_component="embedding",
            consumer_components=["layer_0"],
            device_preference="xdna",
        )
        mask = BoundaryTensor(
            name="attention_mask",
            dtype="int64",
            shape=[1, None],
            producer_component="embedding",
            consumer_components=["layer_0"],
        )
        positions = BoundaryTensor(
            name="position_ids",
            dtype="int64",
            shape=[1, None],
            producer_component="embedding",
            consumer_components=["layer_0"],
        )
        past = BoundaryTensor(
            name="past_key_values.0.key",
            dtype="float16",
            shape=[1, 8, None, 128],
            producer_component="layer_0",
            consumer_components=["layer_1"],
        )
        present = {
            "name": "present.0.key",
            "dtype": "float16",
            "shape": [1, 8, None, 128],
            "producer_component": "layer_0",
            "consumer_components": ["layer_1"],
            "is_stateful": False,
        }
        assert hidden.is_stateful is False
        assert hidden.device_preference == "npu"
        assert hidden.shape[1] is None
        assert past.is_stateful is True
        first_result = SplitResult(
            output_dir="out",
            num_layers=2,
            component_files={"embedding": "embedding.onnx"},
            model_info={},
            boundary_tensors=[hidden, mask, positions, past, present],
        )
        second_result = SplitResult(
            output_dir="out",
            num_layers=2,
            component_files={},
            model_info={},
        )
        assert second_result.boundary_tensors == []
        assert [item.name for item in first_result.boundary_tensors] == [
            "hidden_states",
            "attention_mask",
            "position_ids",
            "past_key_values.0.key",
            "present.0.key",
        ]
        assert first_result.boundary_by_name()["present.0.key"].is_stateful is True
        copied = hidden.to_dict()
        copied["consumer_components"].append("norm")
        assert hidden.consumer_components == ["layer_0"]
        first_result.boundary_tensors.append(hidden)
        assert second_result.boundary_tensors == []
        try:
            BoundaryTensor(
                name="hidden_states",
                dtype="float16",
                shape=[1, None],
                producer_component="embedding",
                consumer_components="layer_0",
            )
            raise AssertionError("a consumer string was accepted")
        except ValueError as exc:
            assert "consumer_components" in str(exc)
        try:
            SplitResult(
                output_dir="out",
                num_layers=1,
                component_files={},
                model_info={},
                boundary_tensors=[hidden, hidden],
            )
            raise AssertionError("a duplicate boundary name was accepted")
        except ValueError as exc:
            assert "hidden_states" in str(exc)
        import importlib.util
        import types as pytypes
        saved_onnx = sys.modules.get("chunking_utils.onnx")
        saved_onnx_types = sys.modules.get("chunking_utils.onnx.types")
        onnx_dir = os.path.join("/workspace/MASTER_POC", "chunking_utils", "onnx")
        onnx_pkg = pytypes.ModuleType("chunking_utils.onnx")
        onnx_pkg.__path__ = [onnx_dir]
        onnx_pkg.__package__ = "chunking_utils.onnx"
        sys.modules["chunking_utils.onnx"] = onnx_pkg
        spec = importlib.util.spec_from_file_location(
            "chunking_utils.onnx.types",
            os.path.join(onnx_dir, "types.py"),
        )
        onnx_types = importlib.util.module_from_spec(spec)
        onnx_types.__package__ = "chunking_utils.onnx"
        sys.modules["chunking_utils.onnx.types"] = onnx_types
        spec.loader.exec_module(onnx_types)
        onnx_result = onnx_types.OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            boundary_tensors=[hidden],
        )
        assert onnx_result.format is ModelFormat.ONNX
        assert onnx_result.boundary_tensors[0] is hidden
        assert onnx_result.execution_plan.steps == []
        assert onnx_result.execution_plan.policy is canonical_policy.AUTO
        if saved_onnx is None:
            sys.modules.pop("chunking_utils.onnx", None)
        else:
            sys.modules["chunking_utils.onnx"] = saved_onnx
        if saved_onnx_types is None:
            sys.modules.pop("chunking_utils.onnx.types", None)
        else:
            sys.modules["chunking_utils.onnx.types"] = saved_onnx_types
        from chunking_utils.common import QuantizationConfig as package_quant
        from chunking_utils import QuantizationConfig as root_quant
        assert package_quant is QuantizationConfig is root_quant
        plain_quant = SplitConfig(model="weights", output="out")
        other_quant = SplitConfig(model="weights", output="out")
        assert plain_quant.quantization.method is None
        assert plain_quant.quantization.weight_dtype is None
        assert plain_quant.quantization.activation_dtype is None
        assert plain_quant.quantization.kv_cache_dtype is None
        assert plain_quant.quantization.group_size is None
        assert plain_quant.quantization.has_zero_point is False
        assert plain_quant.quantization.packing_format is None
        assert plain_quant.quantization is not other_quant.quantization
        plain_quant.quantization.group_size = 32
        assert other_quant.quantization.group_size is None
        awq = QuantizationConfig(
            method="AWQ",
            weight_dtype="int4",
            activation_dtype="fp16",
            kv_cache_dtype="float16",
            group_size=128,
            packing_format="gemm",
        )
        assert awq.method == "awq"
        assert awq.weight_dtype is TensorDtype.INT4
        assert awq.activation_dtype is TensorDtype.FLOAT16
        assert awq.kv_cache_dtype is TensorDtype.FLOAT16
        assert awq.group_size == 128
        assert awq.has_zero_point is False
        assert awq.packing_format == "gemm"
        shared = QuantizationConfig(method="awq", weight_dtype="int4", group_size=128)
        one = SplitConfig(model="weights", output="out", quantization=shared)
        two = SplitConfig(model="weights", output="out", quantization=shared)
        assert one.quantization is not shared
        assert one.quantization is not two.quantization
        assert one.quantization.packing_format == "awq"
        one.quantization.group_size = 64
        assert two.quantization.group_size == 128
        assert shared.group_size == 128
        gptq = SplitConfig(
            model="weights",
            output="out",
            quantization={
                "method": "gptq",
                "weight_dtype": "int4",
                "group_size": -1,
                "has_zero_point": False,
            },
        )
        assert gptq.quantization.method == "gptq"
        assert gptq.quantization.weight_dtype is TensorDtype.INT4
        assert gptq.quantization.group_size == -1
        assert gptq.quantization.packing_format == "gptq"
        nbits = QuantizationConfig(
            method="matmulnbits",
            weight_dtype=TensorDtype.INT4,
            activation_dtype="int8",
            kv_cache_dtype="fp16",
            group_size=32,
            has_zero_point=True,
        )
        assert nbits.method == "matmul_nbits"
        assert nbits.activation_dtype is TensorDtype.INT8
        assert nbits.kv_cache_dtype is TensorDtype.FLOAT16
        assert nbits.group_size == 32
        assert nbits.has_zero_point is True
        assert nbits.packing_format == "matmul_nbits"
        recorded = nbits.to_dict()
        recorded["group_size"] = 0
        assert nbits.group_size == 32
        assert recorded["weight_dtype"] == "int4"
        assert recorded["packing_format"] == "matmul_nbits"
        try:
            QuantizationConfig(group_size=0)
            raise AssertionError("a zero group size was accepted")
        except ValueError as exc:
            assert "group_size" in str(exc)
        try:
            QuantizationConfig(has_zero_point="yes")
            raise AssertionError("a zero-point string was accepted")
        except ValueError as exc:
            assert "has_zero_point" in str(exc)
        try:
            QuantizationConfig(weight_dtype="fp8")
            raise AssertionError("an fp8 weight dtype was accepted")
        except ValueError as exc:
            assert "weight_dtype" in str(exc)
        try:
            QuantizationConfig(method="sideband")
            raise AssertionError("an unknown quantization method was accepted")
        except ValueError as exc:
            assert "sideband" in str(exc)
        from chunking_utils.common import KVCacheConfig as package_kv
        from chunking_utils import KVCachePolicy as root_kv_policy
        from pipeline_utils.kv_cache import KVCachePolicy as manager_kv_policy
        assert package_kv is KVCacheConfig
        assert root_kv_policy is KVCachePolicy
        assert manager_kv_policy is not KVCachePolicy
        assert manager_kv_policy.DENSE_CONTIGUOUS.value == "dense_contiguous"
        assert manager_kv_policy.PAGED_BLOCKS.value == "paged_blocks"
        assert [item.value for item in KVCachePolicy] == [
            "dense",
            "sliding_window",
            "paged",
            "hybrid_sparse",
        ]
        plain_cache = KVCacheConfig()
        other_cache = SplitConfig(model="weights", output="out")
        another_cache = SplitConfig(model="weights", output="out")
        assert plain_cache.dtype is TensorDtype.FLOAT16
        assert plain_cache.layout == "BNSD"
        assert plain_cache.max_seq_len == 0
        assert plain_cache.policy is KVCachePolicy.DENSE
        assert plain_cache.page_size == 1
        assert plain_cache.sliding_window is None
        assert other_cache.kv_cache is not another_cache.kv_cache
        other_cache.kv_cache.max_seq_len = 32
        assert another_cache.kv_cache.max_seq_len == 0
        window = KVCacheConfig(
            dtype="fp16",
            layout="bnsd",
            max_seq_len=8192,
            policy="sliding",
            sliding_window=4096,
        )
        assert window.dtype is TensorDtype.FLOAT16
        assert window.layout == "BNSD"
        assert window.policy is KVCachePolicy.SLIDING_WINDOW
        assert window.sliding_window == 4096
        recorded_cache = window.to_dict()
        recorded_cache["max_seq_len"] = 1
        assert window.max_seq_len == 8192
        assert recorded_cache["dtype"] == "float16"
        assert recorded_cache["policy"] == "sliding_window"
        paged = SplitConfig(
            model="weights",
            output="out",
            kv_cache={
                "dtype": "int8",
                "layout": "BSHD",
                "max_seq_len": 2048,
                "policy": "paged_blocks",
                "page_size": 16,
            },
        )
        assert paged.kv_cache.dtype is TensorDtype.INT8
        assert paged.kv_cache.layout == "BSHD"
        assert paged.kv_cache.policy is KVCachePolicy.PAGED
        assert paged.kv_cache.page_size == 16
        shared_cache = KVCacheConfig(policy="hybrid_sparse", max_seq_len=1024, page_size=4)
        one_cache = SplitConfig(model="weights", output="out", kv_cache=shared_cache)
        two_cache = SplitConfig(model="weights", output="out", kv_cache=shared_cache)
        assert one_cache.kv_cache is not shared_cache
        assert one_cache.kv_cache is not two_cache.kv_cache
        assert one_cache.kv_cache.policy is KVCachePolicy.HYBRID_SPARSE
        one_cache.kv_cache.page_size = 8
        assert shared_cache.page_size == 4
        assert two_cache.kv_cache.page_size == 4
        try:
            KVCacheConfig(page_size=0)
            raise AssertionError("a zero page size was accepted")
        except ValueError as exc:
            assert "page_size" in str(exc)
        try:
            KVCacheConfig(policy="sliding_window")
            raise AssertionError("a sliding policy without a window was accepted")
        except ValueError as exc:
            assert "sliding_window" in str(exc)
        try:
            KVCacheConfig(max_seq_len=-1)
            raise AssertionError("a negative max sequence length was accepted")
        except ValueError as exc:
            assert "max_seq_len" in str(exc)
        try:
            KVCacheConfig(dtype="fp8")
            raise AssertionError("an fp8 cache dtype was accepted")
        except ValueError as exc:
            assert "dtype" in str(exc)
        from chunking_utils.common import ExecutionPlan as package_plan
        from chunking_utils import ExecutionStep as root_step
        from pipeline_utils.session_manager import (
            ExecutionStep as session_step,
            ExecutionPlan as session_plan,
        )
        assert package_plan is ExecutionPlan
        assert root_step is ExecutionStep
        assert session_step is not ExecutionStep
        assert session_plan is not ExecutionPlan
        session_kept = session_step(
            "embedding", "CPUExecutionProvider", "cpu", "embedding"
        )
        assert session_kept.device_type == "cpu"
        assert session_kept.provider == "CPUExecutionProvider"
        empty_plan = ExecutionPlan()
        assert empty_plan.steps == []
        assert empty_plan.policy is canonical_policy.AUTO
        assert empty_plan.prefill_device is None
        assert empty_plan.decode_device is None
        assert empty_plan.static_shapes is True
        assert empty_plan.warmup_iterations == 0
        first_split = SplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
        )
        second_split = SplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
        )
        assert first_split.execution_plan is not second_split.execution_plan
        first_split.execution_plan.warmup_iterations = 3
        assert second_split.execution_plan.warmup_iterations == 0
        prefill = ExecutionStep(
            component_name="embedding",
            provider="VitisAIExecutionProvider",
            device="xdna",
            phase="prefill",
        )
        decode = {
            "component_name": "layer_0",
            "provider": "DmlExecutionProvider",
            "device": "rdna",
            "phase": "decode",
            "depends_on": ["embedding"],
            "resident": True,
        }
        draft = ExecutionStep(component_name="draft_head", phase="draft", resident=False)
        assert prefill.device == "npu"
        assert prefill.phase == "prefill"
        assert prefill.resident is True
        assert draft.phase == "speculative_draft"
        assert draft.resident is False
        hybrid = SplitResult(
            output_dir="out",
            num_layers=1,
            component_files={"embedding": "embedding.onnx", "layer_0": "layer_0.onnx"},
            model_info={},
            execution_plan={
                "policy": "hybrid_npu_igpu",
                "prefill_device": "xdna",
                "decode_device": "rdna",
                "static_shapes": True,
                "warmup_iterations": 2,
                "steps": [prefill, decode],
            },
        )
        assert hybrid.execution_plan.policy is canonical_policy.HYBRID_NPU_IGPU
        assert hybrid.execution_plan.prefill_device == "npu"
        assert hybrid.execution_plan.decode_device == "igpu"
        assert hybrid.execution_plan.warmup_iterations == 2
        assert [step.component_name for step in hybrid.execution_plan.steps] == [
            "embedding",
            "layer_0",
        ]
        assert hybrid.execution_plan.steps[1].device == "igpu"
        assert hybrid.execution_plan.steps[1].depends_on == ["embedding"]
        assert hybrid.execution_plan.steps[0] is not prefill
        recorded_plan = hybrid.execution_plan.to_dict()
        recorded_plan["steps"][1]["depends_on"].append("norm")
        assert hybrid.execution_plan.steps[1].depends_on == ["embedding"]
        assert recorded_plan["policy"] == "hybrid_npu_igpu"
        try:
            ExecutionStep(component_name="layer_0", depends_on="embedding")
            raise AssertionError("a dependency string was accepted")
        except ValueError as exc:
            assert "depends_on" in str(exc)
        try:
            ExecutionPlan(steps=[prefill, prefill])
            raise AssertionError("a repeated component was accepted")
        except ValueError as exc:
            assert "embedding" in str(exc)
        try:
            ExecutionPlan(steps=[
                ExecutionStep(component_name="layer_0", depends_on=["missing"]),
            ])
            raise AssertionError("a missing dependency was accepted")
        except ValueError as exc:
            assert "missing" in str(exc)
        try:
            ExecutionPlan(
                policy="sequential",
                steps=[
                    ExecutionStep(
                        component_name="layer_0",
                        provider="VitisAIExecutionProvider",
                        device="npu",
                    ),
                ],
            )
            raise AssertionError("a sequential NPU step was accepted")
        except ValueError as exc:
            assert "sequential" in str(exc)
            assert "NPU" in str(exc)
        try:
            ExecutionPlan(policy="hybrid_npu_igpu", steps=[prefill])
            raise AssertionError("a hybrid plan without decode was accepted")
        except ValueError as exc:
            assert "decode" in str(exc)
        try:
            ExecutionPlan(warmup_iterations=-1)
            raise AssertionError("a negative warmup count was accepted")
        except ValueError as exc:
            assert "warmup_iterations" in str(exc)
        eager = SplitConfig(
            model="weights",
            output="out",
            memory_mode="eager",
            dtype="fp16",
        )
        assert eager.memory_mode is MemoryMode.EAGER
        assert eager.dtype is TensorDtype.FLOAT16
        assert eager.validate() is None
        assert ExecutionPlan().validate() is None
        assert ComponentInfo(
            component_type="embedding",
            input_names=["hidden", "mask"],
            output_names=["hidden"],
            input_types={"hidden": "fp16", "mask": "int64"},
            output_types={"hidden": "float16"},
        ).validate() is None
        eager.memory_mode = "tape"
        try:
            eager.validate()
            raise AssertionError("an invalid memory mode was accepted")
        except ValueError as exc:
            assert "memory_mode" in str(exc)
        unsafe = SplitConfig(
            model="weights",
            output="out",
            target_device="cpu",
            execution_policy="sequential",
        )
        unsafe.validate()
        unsafe.target_device = DeviceType.NPU
        try:
            unsafe.validate()
            raise AssertionError("a sequential NPU policy was accepted by validate")
        except ValueError as exc:
            assert "NPU" in str(exc)
        missing_model = SplitConfig(model="missing-split-model.onnx", output="out")
        try:
            missing_model.validate()
            raise AssertionError("a missing model file was accepted")
        except ValueError as exc:
            assert "missing file" in str(exc)
        mixed_dtype = SplitConfig(
            model="weights",
            output="out",
            dtype="float16",
            quantization=QuantizationConfig(activation_dtype="int8"),
        )
        try:
            mixed_dtype.validate()
            raise AssertionError("an inconsistent dtype was accepted")
        except ValueError as exc:
            assert "inconsistent dtype" in str(exc)
        try:
            ComponentInfo(component_type="layer").validate()
            raise AssertionError("a layer without an id was accepted")
        except ValueError as exc:
            assert "layer" in str(exc)
        try:
            ComponentInfo(component_type="embedding", layer_id=3).validate()
            raise AssertionError("a layer id on an embedding was accepted")
        except ValueError as exc:
            assert "layer" in str(exc)
        try:
            ComponentInfo(
                component_type="embedding",
                file_path="missing-component.onnx",
            ).validate()
            raise AssertionError("a missing component file was accepted")
        except ValueError as exc:
            assert "missing file" in str(exc)
        try:
            ComponentInfo(
                component_type="embedding",
                input_names=["hidden"],
                output_names=["hidden"],
                input_types={"hidden": "float16"},
                output_types={"hidden": "float32"},
            ).validate()
            raise AssertionError("an inconsistent component dtype was accepted")
        except ValueError as exc:
            assert "inconsistent dtype" in str(exc)
        try:
            SplitResult(
                output_dir="out",
                num_layers=0,
                component_files={},
                model_info={},
            ).validate()
            raise AssertionError("an empty component map was accepted")
        except ValueError as exc:
            assert "empty" in str(exc)
        checked_dir = tempfile.mkdtemp(prefix="split-validate-")
        embed_path = os.path.join(checked_dir, "embedding.onnx")
        layer_path = os.path.join(checked_dir, "layer_0.onnx")
        extra_path = os.path.join(checked_dir, "layer_2.onnx")
        try:
            for path in (embed_path, layer_path, extra_path):
                with open(path, "wb") as handle:
                    handle.write(b"onnx")
            ready = SplitResult(
                output_dir=checked_dir,
                num_layers=1,
                component_files={
                    "embedding": embed_path,
                    "layer_0": layer_path,
                },
                model_info={},
                boundary_tensors=[
                    BoundaryTensor(
                        name="hidden_states",
                        dtype="fp16",
                        shape=[1, None, 16],
                        producer_component="embedding",
                        consumer_components=["layer_0"],
                    )
                ],
                execution_plan=ExecutionPlan(
                    policy="hybrid_npu_igpu",
                    prefill_device="npu",
                    decode_device="igpu",
                    steps=[
                        ExecutionStep(
                            component_name="embedding",
                            provider="VitisAIExecutionProvider",
                            device="npu",
                            phase="prefill",
                        ),
                        ExecutionStep(
                            component_name="layer_0",
                            provider="DmlExecutionProvider",
                            device="igpu",
                            phase="decode",
                            depends_on=["embedding"],
                        ),
                    ],
                ),
            )
            ready.validate()
            ready.execution_plan.policy = canonical_policy.SEQUENTIAL
            try:
                ready.execution_plan.validate()
                raise AssertionError("a mutated sequential NPU plan was accepted")
            except ValueError as exc:
                assert "NPU" in str(exc)
            missing_layer = SplitResult(
                output_dir=checked_dir,
                num_layers=1,
                component_files={"embedding": embed_path},
                model_info={},
            )
            try:
                missing_layer.validate()
                raise AssertionError("a missing layer file was accepted")
            except ValueError as exc:
                assert "layer_0" in str(exc)
            orphan = SplitResult(
                output_dir=checked_dir,
                num_layers=1,
                component_files={
                    "embedding": embed_path,
                    "layer_0": layer_path,
                },
                model_info={},
                execution_plan=ExecutionPlan(
                    steps=[
                        ExecutionStep(
                            component_name="embedding",
                            device="cpu",
                            phase="embedding",
                        ),
                        ExecutionStep(
                            component_name="norm",
                            device="cpu",
                            phase="norm",
                            depends_on=["embedding"],
                        ),
                    ],
                ),
            )
            try:
                orphan.validate()
                raise AssertionError("a missing component dependency was accepted")
            except ValueError as exc:
                assert "norm" in str(exc)
            extra_layer = SplitResult(
                output_dir=checked_dir,
                num_layers=1,
                component_files={
                    "embedding": embed_path,
                    "layer_0": layer_path,
                    "layer_2": extra_path,
                },
                model_info={},
            )
            try:
                extra_layer.validate()
                raise AssertionError("an invalid layer id was accepted")
            except ValueError as exc:
                assert "layer" in str(exc)
            ComponentInfo(
                component_type="layer",
                layer_id=0,
                file_path=layer_path,
            ).validate()
        finally:
            for path in (embed_path, layer_path, extra_path):
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(checked_dir)
        import json
        manifest_component = ComponentInfo(
            component_type="layer",
            layer_id=0,
            file_path="layer_0.onnx",
            input_names=["hidden", "mask"],
            output_names=["hidden"],
            input_types={"hidden": "fp16", "mask": "int64"},
            output_types={"hidden": "float16"},
            input_shapes={"hidden": [1, None, 16], "mask": [1, None]},
            output_shapes={"hidden": [1, None, 16]},
            depends_on=["embedding"],
            device_preference="xdna",
            phase="prefill",
        )
        manifest = {"components": [manifest_component.to_dict()]}
        manifest_raw = json.loads(json.dumps(manifest))
        loaded_component = ComponentInfo.from_dict(manifest_raw["components"][0])
        assert loaded_component.name == "layer_0"
        assert loaded_component.device_preference == "npu"
        assert loaded_component.phase == "prefill"
        assert loaded_component.input_shapes["hidden"][1] is None
        assert loaded_component.to_dict() == manifest_component.to_dict()
        manifest_raw["components"][0]["input_names"].append("extra")
        assert loaded_component.input_names == ["hidden", "mask"]
        saved_plan = ExecutionPlan(
            policy="hybrid_npu_igpu",
            prefill_device="xdna",
            decode_device="rdna",
            warmup_iterations=2,
            steps=[
                ExecutionStep(
                    component_name="embedding",
                    provider="VitisAIExecutionProvider",
                    device="npu",
                    phase="prefill",
                ),
                ExecutionStep(
                    component_name="layer_0",
                    device="igpu",
                    phase="decode",
                    depends_on=["embedding"],
                ),
            ],
        )
        plan_raw = json.loads(json.dumps(saved_plan.to_dict()))
        loaded_plan = ExecutionPlan.from_dict(plan_raw)
        second_plan = ExecutionPlan.from_dict(plan_raw)
        assert loaded_plan.policy is canonical_policy.HYBRID_NPU_IGPU
        assert loaded_plan.prefill_device == "npu"
        assert loaded_plan.decode_device == "igpu"
        assert loaded_plan.warmup_iterations == 2
        assert loaded_plan.steps[1].depends_on == ["embedding"]
        plan_raw["steps"][1]["depends_on"].append("norm")
        assert loaded_plan.steps[1].depends_on == ["embedding"]
        loaded_plan.steps[1].depends_on.append("lm_head")
        assert second_plan.steps[1].depends_on == ["embedding"]
        saved_result = SplitResult(
            output_dir="out",
            num_layers=1,
            component_files={"embedding": "embedding.onnx", "layer_0": "layer_0.onnx"},
            model_info={"original_model": "weights", "num_layers": 1},
            format="onnx",
            boundary_tensors=[
                BoundaryTensor(
                    name="hidden_states",
                    dtype="fp16",
                    shape=[1, None, 16],
                    producer_component="embedding",
                    consumer_components=["layer_0"],
                )
            ],
            execution_plan=saved_plan,
        )
        result_raw = json.loads(json.dumps(saved_result.to_dict()))
        loaded_result = SplitResult.from_dict(result_raw)
        assert loaded_result.format is ModelFormat.ONNX
        assert loaded_result.execution_plan.policy is canonical_policy.HYBRID_NPU_IGPU
        assert loaded_result.boundary_tensors[0].shape[1] is None
        assert loaded_result.model_info["original_model"] == "weights"
        result_raw["model_info"]["original_model"] = "other"
        assert loaded_result.model_info["original_model"] == "weights"
        saved_config = SplitConfig(
            model="weights",
            output="out",
            dtype="fp16",
            memory_mode="eager",
            target_device="npu",
            execution_policy="hybrid_npu_igpu",
            quantization={"method": "awq", "weight_dtype": "int4", "group_size": 128},
        )
        loaded_config = SplitConfig.from_dict(json.loads(json.dumps(saved_config.to_dict())))
        assert loaded_config.dtype is TensorDtype.FLOAT16
        assert loaded_config.memory_mode is MemoryMode.EAGER
        assert loaded_config.quantization.weight_dtype is TensorDtype.INT4
        assert loaded_config.quantization.packing_format == "awq"
        assert loaded_config.to_dict() == saved_config.to_dict()
        try:
            ComponentInfo.from_dict(["layer"])
            raise AssertionError("a component list was accepted")
        except ValueError as exc:
            assert "dict" in str(exc)
        try:
            ExecutionPlan.from_dict("hybrid")
            raise AssertionError("a plan string was accepted")
        except ValueError as exc:
            assert "dict" in str(exc)
        try:
            SplitResult.from_dict(["out"])
            raise AssertionError("a result list was accepted")
        except ValueError as exc:
            assert "dict" in str(exc)
        try:
            SplitConfig(model="weights", output="out", target_device="tpu")
            raise AssertionError("an unknown device was accepted")
        except ValueError as exc:
            assert "tpu" in str(exc)
        try:
            SplitConfig(model="weights", output="out", providers="CPUExecutionProvider")
            raise AssertionError("a provider string was accepted")
        except ValueError as exc:
            assert "providers" in str(exc)
        assert providers_for_device(DeviceType.IPU, [
            "DmlExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]) == [
            "DmlExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]
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
    print("  model format: one enum for onnx, safetensors, gguf, pytorch_bin")
    print("  device policy: providers, npu, ipu, hybrid execution")
    print("  execution policy: monolithic, sequential rejected on NPU, hybrid and moe allowed")
    print("  component contract: inputs, shapes, phase, residency")
    print("  boundary tensors: hidden, mask, past and present")
    print("  quantization: awq int4 group, matmul nbits, kv dtype")
    print("  kv cache config: dense float16 bnsd, sliding window, paged blocks")
    print("  execution plan: prefill on npu, decode on igpu")
    print("  validation: files, layers, dtypes, npu policy")
    print("  serialization: component manifest, execution plan, model info")
    print("backend module: OK\n")
    return True


def test_hf_integration():
    """ONNX Runtime dtype strings sit beside the PyTorch map."""
    print("=" * 50)
    print("Testing HuggingFace dtype mapping...")
    print("=" * 50)

    import types
    import warnings

    import numpy as np

    import chunking_utils.common as common
    from chunking_utils.common.hf_integration import (
        HAS_TORCH,
        HAS_TRANSFORMERS,
        ONNX_DTYPE_MAP as direct_map,
        TORCH_DTYPE_MAP,
        check_dependencies,
        _fields_from_hf_config,
        estimate_model_memory,
        estimate_kv_cache_memory,
        get_model_config,
        get_onnx_dtype as direct_get,
        get_split_model_config,
        get_torch_dtype,
        recommend_execution_strategy,
    )
    from chunking_utils.common.types import TensorDtype
    from pipeline_utils.dtypes import get_numpy_dtype

    assert common.get_onnx_dtype is direct_get
    assert common.ONNX_DTYPE_MAP is direct_map
    assert direct_get(TensorDtype.FLOAT32) == "tensor(float)"
    assert direct_get(TensorDtype.FLOAT16) == "tensor(float16)"
    assert direct_get(TensorDtype.BFLOAT16) == "tensor(bfloat16)"
    assert direct_get(TensorDtype.INT8) == "tensor(int8)"
    assert direct_get(TensorDtype.INT4) == "tensor(int4)"
    assert direct_get("float32") == "tensor(float)"
    assert direct_get("int4") == "tensor(int4)"
    assert direct_get(TensorDtype.AUTO) == "tensor(float16)"
    assert direct_get("auto") == "tensor(float16)"
    assert TensorDtype.AUTO not in direct_map
    assert set(direct_map) == {
        TensorDtype.FLOAT32,
        TensorDtype.FLOAT16,
        TensorDtype.BFLOAT16,
        TensorDtype.INT8,
        TensorDtype.INT4,
    }
    assert TORCH_DTYPE_MAP[TensorDtype.INT4] is None
    assert TORCH_DTYPE_MAP[TensorDtype.AUTO] is None
    assert get_numpy_dtype(direct_get(TensorDtype.FLOAT16)) == np.float16
    assert get_numpy_dtype(direct_get(TensorDtype.FLOAT32)) == np.float32
    assert get_numpy_dtype(direct_get(TensorDtype.INT8)) == np.int8
    assert get_numpy_dtype(direct_get(TensorDtype.INT4)) == np.int8
    if HAS_TORCH:
        assert check_dependencies()["torch"] is True
        assert get_torch_dtype(TensorDtype.FLOAT16) is not None
        assert get_torch_dtype(TensorDtype.INT4) is None
        assert get_torch_dtype(TensorDtype.AUTO) is None
    else:
        assert check_dependencies()["torch"] is False
        try:
            get_torch_dtype(TensorDtype.FLOAT16)
            raise AssertionError("torch dtype without torch was accepted")
        except ImportError as exc:
            assert "torch" in str(exc)
    print("  onnx dtype: float, float16, bfloat16, int8, int4")

    assert common.estimate_model_memory is estimate_model_memory
    plain = estimate_model_memory(1000, TensorDtype.FLOAT16)
    assert plain["bytes"] == 2000
    assert plain["dtype"] == "float16"
    assert plain["bytes_per_param"] == 2
    assert plain["overhead_bytes"] == 0
    assert plain["quantization"] is None
    assert plain["megabytes"] == 2000 / (1024 ** 2)
    packed = estimate_model_memory(1000, "int4")
    assert packed["bytes"] == 500
    assert packed["weight_bytes"] == 500
    assert packed["bytes_per_param"] == 0.5
    assert packed["overhead_bytes"] == 0
    odd = estimate_model_memory(3, TensorDtype.INT4)
    assert odd["weight_bytes"] == 2
    assert odd["bytes"] == 2
    awq = estimate_model_memory(256, "fp16", quantization="awq")
    assert awq["quantization"] == "awq"
    assert awq["dtype"] == "int4"
    assert awq["group_size"] == 128
    assert awq["packing"] == "int32"
    assert awq["weight_bytes"] == 128
    assert awq["scale_bytes"] == 4
    assert awq["zero_point_bytes"] == 8
    assert awq["overhead_bytes"] == 12
    assert awq["bytes"] == 140
    gptq = estimate_model_memory(256, TensorDtype.FLOAT16, quantization="gptq", group_size=128)
    assert gptq["quantization"] == "gptq"
    assert gptq["packing"] == "int32"
    assert gptq["weight_bytes"] == awq["weight_bytes"]
    assert gptq["zero_point_bytes"] == 8
    assert gptq["bytes"] == 140
    nbits = estimate_model_memory(256, TensorDtype.INT4, quantization="matmulnbits")
    assert nbits["quantization"] == "matmul_nbits"
    assert nbits["group_size"] == 128
    assert nbits["packing"] == "block"
    assert nbits["weight_bytes"] == 128
    assert nbits["scale_bytes"] == 4
    assert nbits["zero_point_bytes"] == 2
    assert nbits["bytes"] == 134
    partial = estimate_model_memory(100, TensorDtype.FLOAT16, quantization="matmul_nbits", group_size=128)
    assert partial["weight_bytes"] == 64
    assert partial["scale_bytes"] == 2
    assert partial["zero_point_bytes"] == 1
    assert partial["bytes"] == 67
    assert partial["bytes"] > estimate_model_memory(100, TensorDtype.INT4)["bytes"]
    blocked = estimate_model_memory(32, TensorDtype.INT4, quantization="int4_blockwise")
    assert blocked["quantization"] == "blocked"
    assert blocked["group_size"] == 32
    assert blocked["weight_bytes"] == 16
    assert blocked["scale_bytes"] == 2
    assert blocked["zero_point_bytes"] == 1
    assert blocked["bytes"] == 19
    int8_awq = estimate_model_memory(128, TensorDtype.INT8, quantization="awq")
    assert int8_awq["dtype"] == "int8"
    assert int8_awq["bytes_per_param"] == 1
    assert int8_awq["weight_bytes"] == 128
    assert int8_awq["group_size"] == 128
    assert int8_awq["scale_bytes"] == 2
    assert int8_awq["zero_point_bytes"] == 4
    qdq = estimate_model_memory(10, TensorDtype.INT8, quantization="qdq")
    assert qdq["group_size"] is None
    assert qdq["weight_bytes"] == 10
    assert qdq["scale_bytes"] == 4
    assert qdq["zero_point_bytes"] == 1
    assert qdq["bytes"] == 15
    copied = estimate_model_memory(256, TensorDtype.FLOAT16, quantization="awq")
    awq["weight_bytes"] = 0
    assert copied["weight_bytes"] == 128
    try:
        estimate_model_memory(32, TensorDtype.INT4, quantization="sideband")
        raise AssertionError("an unknown quantization method was accepted")
    except ValueError as exc:
        assert "sideband" in str(exc)
    try:
        estimate_model_memory(32, TensorDtype.INT4, quantization="awq", group_size=-1)
        raise AssertionError("per-channel group size was accepted")
    except ValueError as exc:
        assert "group_size" in str(exc)
    try:
        estimate_model_memory(32, TensorDtype.INT4, quantization="gptq", group_size=0)
        raise AssertionError("a zero group size was accepted")
    except ValueError as exc:
        assert "group_size" in str(exc)
    print("  quantized memory: int4 packing, awq and gptq groups, matmul nbits blocks")

    if not HAS_TRANSFORMERS:
        assert get_model_config("unused") == {"error": "transformers not installed"}
    llama = types.SimpleNamespace(
        model_type="llama",
        num_hidden_layers=2,
        hidden_size=32,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_theta=10000,
        torch_dtype="float16",
    )
    planned = _fields_from_hf_config(llama)
    assert planned["model_type"] == "llama"
    assert planned["num_hidden_layers"] == 2
    assert planned["hidden_size"] == 32
    assert planned["vocab_size"] == 128
    assert planned["num_attention_heads"] == 4
    assert planned["torch_dtype"] == "float16"
    assert planned["max_position_embeddings"] == 4096
    assert planned["max_sequence_length"] is None
    assert planned["max_seq_len"] == 4096
    assert planned["num_key_value_heads"] == 2
    assert planned["head_dim"] == 8
    assert planned["rope_theta"] == 10000.0
    assert planned["sliding_window"] is None
    assert planned["use_sliding_window"] is False
    assert planned["is_moe"] is False
    assert planned["num_experts"] == 0
    assert planned["num_experts_per_token"] == 0
    fallen = _fields_from_hf_config(types.SimpleNamespace(
        max_position_embeddings=0,
        max_sequence_length=8192,
        num_attention_heads=4,
        hidden_size=32,
    ))
    assert fallen["max_position_embeddings"] is None
    assert fallen["max_sequence_length"] == 8192
    assert fallen["max_seq_len"] == 8192
    sliding = _fields_from_hf_config(types.SimpleNamespace(
        sliding_window=4096,
        num_attention_heads=4,
        hidden_size=32,
        head_dim=96,
    ))
    assert sliding["sliding_window"] == 4096
    assert sliding["use_sliding_window"] is True
    assert sliding["head_dim"] == 96
    held = _fields_from_hf_config(types.SimpleNamespace(
        sliding_window=4096,
        use_sliding_window=False,
        num_attention_heads=8,
        hidden_size=64,
    ))
    assert held["use_sliding_window"] is False
    assert held["num_key_value_heads"] == 8
    assert held["head_dim"] == 8
    rope_scaling = {"type": "linear", "factor": 2}
    mixtral = _fields_from_hf_config(types.SimpleNamespace(
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        router_aux_loss_coef=0,
        rope_scaling=rope_scaling,
        num_attention_heads=4,
        hidden_size=32,
    ))
    assert mixtral["is_moe"] is True
    assert mixtral["num_experts"] == 8
    assert mixtral["num_experts_per_token"] == 2
    assert mixtral["moe_intermediate_size"] == 64
    assert mixtral["router_aux_loss_coef"] == 0.0
    assert mixtral["rope_scaling"] == {"type": "linear", "factor": 2}
    mixtral["rope_scaling"]["factor"] = 1
    assert rope_scaling["factor"] == 2
    nested = _fields_from_hf_config(types.SimpleNamespace(
        model_type="qwen2_vl",
        text_config=types.SimpleNamespace(
            num_hidden_layers=4,
            hidden_size=16,
            num_attention_heads=2,
            max_position_embeddings=2048,
            rope_theta=1000000,
        ),
    ))
    assert nested["model_type"] == "qwen2_vl"
    assert nested["num_hidden_layers"] == 4
    assert nested["max_seq_len"] == 2048
    assert nested["head_dim"] == 8
    assert nested["rope_theta"] == 1000000.0
    flagged = _fields_from_hf_config(types.SimpleNamespace(is_moe=True))
    assert flagged["is_moe"] is True
    assert flagged["num_experts"] == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        odd_dim = _fields_from_hf_config(types.SimpleNamespace(
            hidden_size=30,
            num_attention_heads=4,
        ))
    assert odd_dim["head_dim"] == 7
    assert any("not divisible" in str(item.message) for item in caught)
    try:
        _fields_from_hf_config(types.SimpleNamespace(
            num_experts=2,
            num_experts_per_token=3,
        ))
        raise AssertionError("more experts per token than experts was accepted")
    except ValueError as exc:
        assert "num_experts_per_token" in str(exc)
    try:
        _fields_from_hf_config(types.SimpleNamespace(max_position_embeddings=True))
        raise AssertionError("a boolean context length was accepted")
    except ValueError as exc:
        assert "max_position_embeddings" in str(exc)
    try:
        _fields_from_hf_config(types.SimpleNamespace(rope_theta=-1))
        raise AssertionError("a negative rope theta was accepted")
    except ValueError as exc:
        assert "rope_theta" in str(exc)
    print("  hf config: context, sliding window, kv heads, head dim, rope, moe")

    import json
    import os
    import tempfile

    assert common.get_split_model_config is get_split_model_config

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

    def _write_json(directory, name, payload):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    try:
        get_split_model_config(os.path.join(tempfile.gettempdir(), "missing-split-dir"))
        raise AssertionError("a missing split directory was accepted")
    except FileNotFoundError as exc:
        assert "model directory not found" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        try:
            get_split_model_config(directory)
            raise AssertionError("an empty directory was accepted")
        except ValueError as exc:
            assert "No ONNX graph" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_past_key_onnx(
            os.path.join(directory, "embeddings.onnx"),
            [("hidden_states", [1, "seq", 32])],
            elem=1,
        )
        _write_past_key_onnx(
            os.path.join(directory, "layer_0.onnx"),
            [
                ("past_key_values.0.key", [1, 2, "seq", 8]),
                ("past_key_values.0.value", [1, 2, "seq", 8]),
            ],
            elem=10,
            external="weights.onnx.data",
        )
        _write_past_key_onnx(
            os.path.join(directory, "layer_1.onnx"),
            [
                ("past_key_values.0.key", [1, 2, "seq", 8]),
                ("past_key_values.0.value", [1, 2, "seq", 8]),
            ],
            elem=10,
        )
        with open(os.path.join(directory, "weights.onnx.data"), "wb") as handle:
            handle.write(b"\x00\x00")
        _write_json(directory, "config.json", {
            "max_position_embeddings": 8192,
            "rope_theta": 10000,
            "model_type": "ignored",
        })
        _write_json(directory, "model_info.json", {
            "num_layers": 2,
            "architecture": "llama",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "producer": "test",
            "ir_version": 8,
            "components": {
                "embedding": {"type": "embedding", "file": "embeddings.onnx"},
                "layer_0": {"type": "layer", "layer_id": 0, "file": "layer_0.onnx"},
                "layer_1": {"type": "layer", "layer_id": 1, "file": "layer_1.onnx"},
            },
            "boundary_tensors": [
                {
                    "name": "hidden_states",
                    "dtype": "float32",
                    "shape": [1, None, 32],
                    "producer_component": "embedding",
                    "consumer_components": ["layer_0"],
                    "is_stateful": False,
                },
                {
                    "name": "past_key_values.0.key",
                    "dtype": "float16",
                    "shape": [1, 2, None, 8],
                    "producer_component": "layer_0",
                    "consumer_components": ["layer_1"],
                    "is_stateful": False,
                },
                {
                    "name": "presentation",
                    "dtype": "float16",
                    "shape": [1],
                    "producer_component": "layer_0",
                    "consumer_components": ["layer_1"],
                    "is_stateful": False,
                },
            ],
        })
        split = get_split_model_config(directory)
        assert split["format"] == "onnx"
        assert split["model_type"] == "llama"
        assert split["num_hidden_layers"] == 2
        assert split["hidden_size"] == 32
        assert split["num_attention_heads"] == 4
        assert split["num_key_value_heads"] == 2
        assert split["head_dim"] == 8
        assert split["max_position_embeddings"] == 4096
        assert split["max_seq_len"] == 4096
        assert split["rope_theta"] == 10000.0
        assert split["producer"] == "test"
        assert split["ir_version"] == 8
        assert [item["name"] for item in split["components"]] == [
            "embedding", "layer_0", "layer_1",
        ]
        assert split["components"][1]["layer_id"] == 0
        assert split["components"][1]["external_data"] == ["weights.onnx.data"]
        assert split["components"][2]["external_data"] == []
        assert "past_key_values.0.key" in split["components"][1]["input_names"]
        past = next(
            item for item in split["components"][1]["inputs"]
            if item["name"] == "past_key_values.0.key"
        )
        assert past["shape"] == [1, 2, None, 8]
        assert past["dtype"] == "tensor(float16)"
        assert split["external_data"] == ["weights.onnx.data"]
        assert split["files"] == ["embeddings.onnx", "layer_0.onnx", "layer_1.onnx"]
        by_name = {item["name"]: item for item in split["boundary_tensors"]}
        assert by_name["past_key_values.0.key"]["is_stateful"] is True
        assert by_name["past_key_values.0.key"]["cache_role"] == "past_key"
        assert by_name["past_key_values.0.key"]["zero_copy"] is True
        assert by_name["past_key_values.0.key"]["device_preference"] == "npu"
        assert by_name["past_key_values.0.key"]["producer"] == "layer_0"
        assert by_name["hidden_states"]["is_stateful"] is False
        assert by_name["hidden_states"]["cache_role"] is None
        assert by_name["hidden_states"]["zero_copy"] is True
        assert by_name["presentation"]["is_stateful"] is False
        assert by_name["presentation"]["cache_role"] is None
        assert by_name["hidden_states"]["producer_component"] == "embedding"
        split["external_data"].append("extra")
        split["boundary_tensors"].append({"name": "extra"})
        again = get_split_model_config(directory)
        assert again["external_data"] == ["weights.onnx.data"]
        assert len(again["boundary_tensors"]) == 3
        via_model = get_model_config(directory)
        assert via_model["max_seq_len"] == 4096
        assert via_model["components"][0]["file"] == "embeddings.onnx"
        os.remove(os.path.join(directory, "weights.onnx.data"))
        try:
            get_split_model_config(directory)
            raise AssertionError("a missing external data file was accepted")
        except ValueError as exc:
            assert "missing external data weights.onnx.data" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_past_key_onnx(
            os.path.join(directory, "layer_0.onnx"),
            [("past_key_values.0.key", [1, 2, "seq", 8])],
            elem=10,
        )
        _write_json(directory, "model_info.json", {
            "num_layers": 1,
            "architecture": "llama",
            "num_key_value_heads": 8,
            "components": {"layer_0": {"type": "layer", "file": "layer_0.onnx"}},
            "boundary_tensors": [{
                "name": "past_key_values.0.key",
                "dtype": "float16",
                "shape": [1, 8, None, 8],
                "producer_component": "layer_0",
                "consumer_components": [],
            }],
        })
        try:
            get_split_model_config(directory)
            raise AssertionError("a KV head mismatch was accepted")
        except ValueError as exc:
            assert "num_key_value_heads 8 != 2" in str(exc)
        info = json.load(open(os.path.join(directory, "model_info.json"), encoding="utf-8"))
        info["num_key_value_heads"] = 2
        info["boundary_tensors"][0]["shape"] = [1, 2, None, 16]
        _write_json(directory, "model_info.json", info)
        try:
            get_split_model_config(directory)
            raise AssertionError("a boundary shape mismatch was accepted")
        except ValueError as exc:
            assert "boundary past_key_values.0.key" in str(exc)
        info["boundary_tensors"][0]["shape"] = [1, 2, None, 8]
        info["boundary_tensors"].append(dict(info["boundary_tensors"][0]))
        _write_json(directory, "model_info.json", info)
        try:
            get_split_model_config(directory)
            raise AssertionError("a duplicate boundary was accepted")
        except ValueError as exc:
            assert "duplicate" in str(exc)
        info["boundary_tensors"] = [{
            "name": "past_key_values.0.key",
            "dtype": "float16",
            "shape": [1, 2, None, 8],
            "producer_component": "missing",
            "consumer_components": [],
        }]
        _write_json(directory, "model_info.json", info)
        try:
            get_split_model_config(directory)
            raise AssertionError("an unknown boundary producer was accepted")
        except ValueError as exc:
            assert "producer missing" in str(exc)
        info["components"]["layer_0"]["input_names"] = ["other"]
        info["boundary_tensors"][0]["producer_component"] = "layer_0"
        _write_json(directory, "model_info.json", info)
        try:
            get_split_model_config(directory)
            raise AssertionError("declared input names were not checked")
        except ValueError as exc:
            assert "input_names" in str(exc)
        info["components"]["layer_1"] = {"type": "layer", "layer_id": 1, "file": "layer_1.onnx"}
        info["num_layers"] = 2
        info["components"]["layer_0"].pop("input_names")
        _write_json(directory, "model_info.json", info)
        try:
            get_split_model_config(directory)
            raise AssertionError("a missing component file was accepted")
        except ValueError as exc:
            assert "missing component file layer_1.onnx" in str(exc)

    with tempfile.TemporaryDirectory() as directory:
        _write_past_key_onnx(os.path.join(directory, "model.onnx"), [
            ("past_key_values.0.key", [1, 4, "seq", 96]),
            ("past_key_values.0.value", [1, 4, "seq", 96]),
            ("past_key_values.1.key", [1, 4, "seq", 96]),
            ("past_key_values.1.value", [1, 4, "seq", 96]),
        ], elem=10)
        monolithic = get_split_model_config(directory)
        assert monolithic["num_hidden_layers"] == 2
        assert monolithic["num_key_value_heads"] == 4
        assert monolithic["head_dim"] == 96
        assert monolithic["torch_dtype"] == "float16"
        assert monolithic["files"] == ["model.onnx"]
        past_boundary = next(
            item for item in monolithic["boundary_tensors"]
            if item["name"] == "past_key_values.0.key"
        )
        assert past_boundary["is_stateful"] is True
        assert past_boundary["producer_component"] == "external"
        assert past_boundary["consumer_components"] == ["model"]
        assert past_boundary["cache_role"] == "past_key"
        assert past_boundary["zero_copy"] is True
        assert past_boundary["device_preference"] == "npu"

    with tempfile.TemporaryDirectory() as directory:
        _write_json(directory, "model_info.json", {
            "num_layers": 1,
            "components": {"layer_0": {"file": "layer_0.onnx"}},
        })
        with open(os.path.join(directory, "layer_0.onnx"), "wb") as handle:
            handle.write(b"not onnx")
        try:
            get_split_model_config(directory)
            raise AssertionError("an unreadable graph was accepted")
        except ValueError as exc:
            assert "could not read ONNX graph layer_0.onnx" in str(exc)

    print("  onnx split: components, boundaries, external data, kv shape")

    assert common.recommend_execution_strategy is recommend_execution_strategy
    amd = ["CPUExecutionProvider", "DmlExecutionProvider", "VitisAIExecutionProvider"]
    small = {"num_params": 1000, "torch_dtype": "float16", "num_hidden_layers": 2}
    hybrid = recommend_execution_strategy(small, amd, 16)
    assert hybrid["provider"] == "VitisAIExecutionProvider"
    assert hybrid["providers"] == ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    assert hybrid["decode_provider"] == "DmlExecutionProvider"
    assert hybrid["device"] == "npu"
    assert hybrid["prefill_device"] == "npu"
    assert hybrid["decode_device"] == "igpu"
    assert hybrid["execution"] == "hybrid"
    assert hybrid["npu_mode"] == "monolithic"
    assert hybrid["fits_npu_memory"] is True
    assert hybrid["fits_memory"] is True
    assert hybrid["dtype"] == "float16"
    assert hybrid["quantization"] is None
    assert hybrid["quantization_needed"] is False
    assert hybrid["npu_safe"] is True
    assert hybrid["sequential_split_safe"] is False
    assert hybrid["num_params"] == 1000
    assert "monolithic" in hybrid["reason"]
    assert "iGPU" in hybrid["reason"]
    hybrid["providers"].append("extra")
    again = recommend_execution_strategy(small, ["vitisai", "directml", "cpu"], 16)
    assert again["providers"] == ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    assert again["provider"] == hybrid["provider"]

    ryzen = recommend_execution_strategy(
        small, ["RyzenAIExecutionProvider", "CPUExecutionProvider"], 16
    )
    assert ryzen["provider"] == "RyzenAIExecutionProvider"
    assert ryzen["providers"] == ["RyzenAIExecutionProvider", "CPUExecutionProvider"]
    assert ryzen["decode_device"] == "cpu"
    assert ryzen["execution"] == "hybrid"
    assert ryzen["sequential_split_safe"] is False

    npu_only = recommend_execution_strategy(small, ["VitisAIExecutionProvider"], 16)
    assert npu_only["execution"] == "monolithic"
    assert npu_only["decode_device"] == "npu"
    assert npu_only["providers"] == ["VitisAIExecutionProvider"]

    wide = recommend_execution_strategy(
        {"num_params": 1000, "torch_dtype": "float32"}, amd, 16
    )
    assert wide["provider"] == "VitisAIExecutionProvider"
    assert wide["dtype"] == "float16"
    assert wide["quantization"] is None
    assert wide["quantization_needed"] is True
    assert wide["execution"] == "hybrid"
    assert "Float16" in wide["reason"]

    packed = estimate_model_memory(2_000_000, TensorDtype.INT4, quantization="matmul_nbits")
    int8_bytes = estimate_model_memory(2_000_000, TensorDtype.INT8)["bytes"]
    budget_mb = (int8_bytes - 1) // (1024 * 1024)
    assert packed["bytes"] <= budget_mb * 1024 * 1024
    assert int8_bytes > budget_mb * 1024 * 1024
    tightened = recommend_execution_strategy(
        {"num_params": 2_000_000, "dtype": "float16"}, amd, budget_mb
    )
    assert tightened["quantization"] == "matmul_nbits"
    assert tightened["dtype"] == "int4"
    assert tightened["quantization_needed"] is True
    assert tightened["fits_npu_memory"] is True
    assert tightened["execution"] == "hybrid"
    assert tightened["weight_bytes"] == packed["bytes"]

    oversized = recommend_execution_strategy(
        {"num_params": 10_000_000, "torch_dtype": "float16"}, amd, 1
    )
    assert oversized["quantization"] == "matmul_nbits"
    assert oversized["fits_npu_memory"] is False
    assert oversized["fits_memory"] is False
    assert oversized["execution"] == "split"
    assert oversized["npu_mode"] == "monolithic"
    assert oversized["sequential_split_safe"] is False
    assert "split" in oversized["reason"]

    cpu_only = recommend_execution_strategy(
        small, ["CPUExecutionProvider"], 16
    )
    assert cpu_only["provider"] == "CPUExecutionProvider"
    assert cpu_only["device"] == "cpu"
    assert cpu_only["execution"] == "monolithic"
    assert cpu_only["sequential_split_safe"] is True
    assert cpu_only["fits_npu_memory"] is False
    assert cpu_only["fits_memory"] is True

    cuda_only = recommend_execution_strategy(small, ["CUDAExecutionProvider"], 16)
    assert cuda_only["provider"] == "CUDAExecutionProvider"
    assert cuda_only["device"] == "gpu"
    assert cuda_only["execution"] == "monolithic"
    assert cuda_only["sequential_split_safe"] is True

    kept = recommend_execution_strategy(
        {"num_params": 256, "torch_dtype": "float16", "quantization": "awq"}, amd, 16
    )
    assert kept["quantization"] == "awq"
    assert kept["quantization_needed"] is False
    assert kept["dtype"] == "int4"
    assert kept["fits_npu_memory"] is True

    hidden = 32
    heads = 4
    kv_heads = 2
    head_dim = 8
    intermediate = 64
    layers = 2
    vocab = 16
    attention = hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim) + (heads * head_dim) * hidden
    mlp = 3 * hidden * intermediate
    embedding = vocab * hidden
    expected_params = embedding + layers * (attention + mlp)
    counted = recommend_execution_strategy({
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "max_seq_len": 4,
    }, amd, 16)
    assert counted["num_params"] == expected_params
    weight = estimate_model_memory(expected_params, TensorDtype.FLOAT16)
    kv_bytes = 2 * layers * kv_heads * 4 * head_dim * 2
    assert counted["kv_bytes"] == kv_bytes
    assert counted["weight_bytes"] == weight["bytes"]
    assert counted["estimated_memory_mb"] == (weight["bytes"] + kv_bytes) / (1024 ** 2)
    assert counted["supports_static_kv_cache"] is True
    assert counted["fits_npu_memory"] is True

    windowed = recommend_execution_strategy({
        "num_params": 1000,
        "torch_dtype": "float16",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_seq_len": 4096,
        "sliding_window": 4,
        "use_sliding_window": True,
    }, amd, 16)
    assert windowed["kv_bytes"] == 2 * 2 * 2 * 4 * 8 * 2

    blocked_window = recommend_execution_strategy({
        "num_params": 1000,
        "torch_dtype": "float16",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_seq_len": 4,
        "sliding_window": 128,
        "use_sliding_window": False,
    }, amd, 16)
    assert blocked_window["kv_bytes"] == 2 * 2 * 2 * 4 * 8 * 2

    heavy_kv = recommend_execution_strategy({
        "num_params": 1000,
        "torch_dtype": "float16",
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_seq_len": 4096,
    }, amd, 1)
    assert heavy_kv["kv_bytes"] == 2 * 2 * 8 * 4096 * 128 * 2
    assert heavy_kv["fits_npu_memory"] is False
    assert heavy_kv["execution"] == "split"
    assert heavy_kv["quantization"] == "matmul_nbits"

    try:
        recommend_execution_strategy(["num_params"], amd, 16)
        raise AssertionError("a list config was accepted")
    except ValueError as exc:
        assert "model_config" in str(exc)
    try:
        recommend_execution_strategy(small, "VitisAIExecutionProvider", 16)
        raise AssertionError("a provider string was accepted")
    except ValueError as exc:
        assert "available_providers" in str(exc)
    try:
        recommend_execution_strategy(small, ["TensorrtExecutionProvider"], 16)
        raise AssertionError("an unknown provider was accepted")
    except ValueError as exc:
        assert "TensorrtExecutionProvider" in str(exc)
    try:
        recommend_execution_strategy(small, [], 16)
        raise AssertionError("an empty provider list was accepted")
    except ValueError as exc:
        assert "available_providers" in str(exc)
    try:
        recommend_execution_strategy(small, amd, -1)
        raise AssertionError("a negative budget was accepted")
    except ValueError as exc:
        assert "memory_budget_mb" in str(exc)
    try:
        recommend_execution_strategy(small, amd, True)
        raise AssertionError("a boolean budget was accepted")
    except ValueError as exc:
        assert "memory_budget_mb" in str(exc)
    try:
        recommend_execution_strategy({"torch_dtype": "float16"}, amd, 16)
        raise AssertionError("a config without a parameter count was accepted")
    except ValueError as exc:
        assert "num_params" in str(exc)

    print("  execution strategy: npu hybrid, int4 budget, split when the cache does not fit")

    assert common.estimate_kv_cache_memory is estimate_kv_cache_memory
    cache = estimate_kv_cache_memory(2, 8, 128, 4096)
    assert cache["bytes"] == heavy_kv["kv_bytes"]
    assert cache["bytes"] == 2 * 2 * 8 * 4096 * 128 * 2
    assert cache["megabytes"] == cache["bytes"] / (1024 ** 2)
    assert cache["gigabytes"] == cache["bytes"] / (1024 ** 3)
    assert cache["bytes_per_token"] == 2 * 2 * 8 * 128 * 2
    assert cache["per_layer_bytes"] == cache["bytes"] / 2
    assert cache["key_bytes"] == cache["value_bytes"] == cache["bytes"] / 2
    assert cache["dtype_bytes"] == 2.0
    assert cache["batch_size"] == 1.0
    assert cache["max_seq_len"] == 4096.0
    cache["bytes"] = 0
    again = estimate_kv_cache_memory(2, 8, 128, 4096)
    assert again["bytes"] == 2 * 2 * 8 * 4096 * 128 * 2

    long_ctx = estimate_kv_cache_memory(32, 8, 128, 8192, 2)
    assert long_ctx["bytes"] == 1_073_741_824
    assert long_ctx["megabytes"] == 1024.0
    assert long_ctx["gigabytes"] == 1.0
    assert long_ctx["bytes_per_token"] == 2 * 32 * 8 * 128 * 2
    assert long_ctx["per_layer_bytes"] == long_ctx["bytes"] / 32
    assert long_ctx["bytes"] > estimate_model_memory(1000, TensorDtype.FLOAT16)["bytes"]
    wider = estimate_kv_cache_memory(32, 8, 128, 8192, dtype_bytes=4)
    assert wider["bytes"] == long_ctx["bytes"] * 2
    assert wider["dtype_bytes"] == 4.0
    narrow = estimate_kv_cache_memory(32, 8, 128, 8192, 1)
    assert narrow["bytes"] == long_ctx["bytes"] // 2
    batched = estimate_kv_cache_memory(2, 8, 128, 4096, batch_size=2)
    assert batched["bytes"] == again["bytes"] * 2
    assert batched["bytes_per_token"] == again["bytes_per_token"] * 2
    assert batched["batch_size"] == 2.0

    for bad_args, label in (
        ((0, 8, 128, 4096), "num_layers"),
        ((-1, 8, 128, 4096), "num_layers"),
        ((True, 8, 128, 4096), "num_layers"),
        ((2, 1.5, 128, 4096), "num_kv_heads"),
        ((2, 8, 0, 4096), "head_dim"),
        ((2, 8, 128, -4), "max_seq_len"),
        ((2, 8, 128, 4096, 0), "dtype_bytes"),
        ((2, 8, 128, 4096, -2), "dtype_bytes"),
        ((2, 8, 128, 4096, True), "dtype_bytes"),
    ):
        try:
            estimate_kv_cache_memory(*bad_args)
            raise AssertionError(f"a bad {label} was accepted")
        except ValueError as exc:
            assert label in str(exc)
    try:
        estimate_kv_cache_memory(2, 8, 128, 4096, batch_size=0)
        raise AssertionError("a zero batch was accepted")
    except ValueError as exc:
        assert "batch_size" in str(exc)
    try:
        estimate_kv_cache_memory(2, 8, 128, 4096, batch_size=False)
        raise AssertionError("a boolean batch was accepted")
    except ValueError as exc:
        assert "batch_size" in str(exc)

    print("  kv cache memory: long context, float16 bytes, per token")
    print("hf integration: OK\n")
    return True


def test_architecture():
    """LLaMA-style architectures share one prefix record per format."""
    print("=" * 50)
    print("Testing architecture prefixes...")
    print("=" * 50)

    import chunking_utils.common as common
    from chunking_utils.common.architecture import (
        ARCHITECTURE_DESCRIPTORS,
        LLAMA_STYLE_ARCHITECTURES,
        ONNX_PREFIXES,
        SAFETENSORS_PREFIXES,
        ArchitectureDescriptor,
        ArchitecturePrefixes,
        ArchitectureType,
        _LLAMA_STYLE,
        _LLAMA_STYLE_ONNX,
        architecture_from_string,
        classify_qwen_family,
        EXACT_MAP,
        get_architecture_descriptor,
        QUANTIZED_OP_PATTERNS,
        get_prefixes,
        quantized_op_scheme,
        get_prefixes_tuple,
    )
    from chunking_utils.common.types import ComponentInfo, ComponentType

    assert common.SAFETENSORS_PREFIXES is SAFETENSORS_PREFIXES
    assert common.ONNX_PREFIXES is ONNX_PREFIXES
    assert set(SAFETENSORS_PREFIXES) == set(ArchitectureType)
    assert set(ONNX_PREFIXES) == set(ArchitectureType)
    assert ArchitectureType.GPT2 not in LLAMA_STYLE_ARCHITECTURES
    assert ArchitectureType.UNKNOWN not in LLAMA_STYLE_ARCHITECTURES
    assert ArchitectureType.QWEN3_MOE not in LLAMA_STYLE_ARCHITECTURES
    assert ArchitectureType.QWEN3_HYBRID not in LLAMA_STYLE_ARCHITECTURES
    assert len(LLAMA_STYLE_ARCHITECTURES) == 7
    assert ArchitectureType.QWEN3 in LLAMA_STYLE_ARCHITECTURES
    assert _LLAMA_STYLE is not _LLAMA_STYLE_ONNX
    assert isinstance(_LLAMA_STYLE, ArchitecturePrefixes)

    for arch in LLAMA_STYLE_ARCHITECTURES:
        assert SAFETENSORS_PREFIXES[arch] is _LLAMA_STYLE
        assert ONNX_PREFIXES[arch] is _LLAMA_STYLE_ONNX
        assert get_prefixes(arch, "safetensors") is _LLAMA_STYLE
        assert get_prefixes(arch, "ONNX") is _LLAMA_STYLE_ONNX

    assert SAFETENSORS_PREFIXES[ArchitectureType.UNKNOWN] is _LLAMA_STYLE
    assert ONNX_PREFIXES[ArchitectureType.UNKNOWN] is _LLAMA_STYLE_ONNX
    assert get_prefixes(ArchitectureType.LLAMA, "SafeTensors") is _LLAMA_STYLE
    assert _LLAMA_STYLE.embed_prefix == "model.embed_tokens."
    assert _LLAMA_STYLE.layer_prefix == "model.layers."
    assert _LLAMA_STYLE.norm_prefix == "model.norm."
    assert _LLAMA_STYLE.lm_head_prefix == "lm_head."
    assert _LLAMA_STYLE_ONNX.embed_prefix == "/model/embed"
    assert _LLAMA_STYLE_ONNX.layer_prefix == "/model/layers."
    assert _LLAMA_STYLE_ONNX.norm_prefix == "/model/norm"
    assert _LLAMA_STYLE_ONNX.lm_head_prefix == "/lm_head"
    assert get_prefixes_tuple(ArchitectureType.MISTRAL, "safetensors") == (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
        "lm_head.",
    )
    assert get_prefixes_tuple(ArchitectureType.QWEN3, "onnx") == (
        "/model/embed",
        "/model/layers.",
        "/model/norm",
        "/lm_head",
    )

    gpt2 = SAFETENSORS_PREFIXES[ArchitectureType.GPT2]
    gpt2_onnx = ONNX_PREFIXES[ArchitectureType.GPT2]
    assert gpt2 is not _LLAMA_STYLE
    assert gpt2_onnx is not _LLAMA_STYLE_ONNX
    assert gpt2.embed_prefix == "transformer.wte."
    assert gpt2.layer_prefix == "transformer.h."
    assert gpt2.norm_prefix == "transformer.ln_f."
    assert gpt2.lm_head_prefix == "lm_head."
    assert gpt2_onnx.embed_prefix == "/transformer/wte"
    assert gpt2_onnx.layer_prefix == "/transformer/h."
    assert gpt2_onnx.norm_prefix == "/transformer/ln_f"
    assert gpt2_onnx.lm_head_prefix == "/lm_head"
    assert get_prefixes(ArchitectureType.GPT2, "safetensors") is gpt2

    try:
        _LLAMA_STYLE.embed_prefix = "other"
        raise AssertionError("a shared prefix record was changed")
    except AttributeError:
        pass

    try:
        get_prefixes(ArchitectureType.LLAMA, "gguf")
        raise AssertionError("an unknown format was accepted")
    except ValueError as exc:
        assert "Unknown format" in str(exc)
        assert "gguf" in str(exc)

    assert architecture_from_string("llama") is ArchitectureType.LLAMA
    assert architecture_from_string("Qwen2") is ArchitectureType.QWEN2
    assert architecture_from_string("qwen3") is ArchitectureType.QWEN3
    assert architecture_from_string("qwen") is ArchitectureType.QWEN
    assert architecture_from_string("") is ArchitectureType.UNKNOWN
    assert get_prefixes(architecture_from_string("mixtral"), "safetensors") is _LLAMA_STYLE

    assert architecture_from_string("qwen3_moe") is ArchitectureType.QWEN3_MOE
    assert architecture_from_string("Qwen3-MoE") is ArchitectureType.QWEN3_MOE
    assert architecture_from_string("qwen3_hybrid") is ArchitectureType.QWEN3_HYBRID
    assert architecture_from_string("Qwen3.8") is ArchitectureType.QWEN3_HYBRID
    assert architecture_from_string("qwen3-8") is ArchitectureType.QWEN3_HYBRID
    assert architecture_from_string("qwen3_8") is ArchitectureType.QWEN3_HYBRID
    assert common.EXACT_MAP is EXACT_MAP
    import chunking_utils
    assert chunking_utils.EXACT_MAP is EXACT_MAP
    assert EXACT_MAP == {
        "qwen3_moe": ArchitectureType.QWEN3_MOE,
        "qwen3": ArchitectureType.QWEN3,
        "qwen2": ArchitectureType.QWEN2,
        "qwen": ArchitectureType.QWEN,
    }
    longest = sorted(EXACT_MAP, key=len, reverse=True)
    assert longest[0] == "qwen3_moe"
    assert longest.index("qwen3_moe") < longest.index("qwen3")
    assert longest.index("qwen2") < longest.index("qwen")
    assert architecture_from_string("model_qwen3_moe") is ArchitectureType.QWEN3_MOE
    assert architecture_from_string("model_qwen3") is ArchitectureType.QWEN3
    assert architecture_from_string("Qwen/Qwen3-MoE") is ArchitectureType.QWEN3_MOE
    assert architecture_from_string("qwen2.5") is ArchitectureType.QWEN2
    assert architecture_from_string("llama-3") is ArchitectureType.LLAMA
    assert architecture_from_string("mistral") is ArchitectureType.MISTRAL
    assert architecture_from_string("gpt2") is ArchitectureType.GPT2
    assert architecture_from_string("  ") is ArchitectureType.UNKNOWN
    assert architecture_from_string(None) is ArchitectureType.UNKNOWN
    assert ArchitectureType.QWEN3.value == "qwen3"
    assert ArchitectureType.QWEN3_MOE.value == "qwen3_moe"
    assert ArchitectureType.QWEN3_HYBRID.value == "qwen3_hybrid"
    hybrid_tuple = get_prefixes_tuple(ArchitectureType.QWEN3_HYBRID, "safetensors")
    assert hybrid_tuple == (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
        "lm_head.",
    )
    assert SAFETENSORS_PREFIXES[ArchitectureType.QWEN3_HYBRID] is not _LLAMA_STYLE
    assert SAFETENSORS_PREFIXES[ArchitectureType.QWEN3_MOE] is not _LLAMA_STYLE
    assert ONNX_PREFIXES[ArchitectureType.QWEN3_HYBRID] is not _LLAMA_STYLE_ONNX
    assert get_prefixes_tuple(ArchitectureType.QWEN3_MOE, "onnx") == (
        "/model/embed",
        "/model/layers.",
        "/model/norm",
        "/lm_head",
    )

    optional_names = (
        "router_prefix",
        "expert_prefix",
        "mtp_head_prefix",
        "vision_prefix",
        "cross_attn_prefix",
        "speculative_prefix",
    )
    for arch in (*LLAMA_STYLE_ARCHITECTURES, ArchitectureType.GPT2, ArchitectureType.UNKNOWN):
        for record in (get_prefixes(arch, "safetensors"), get_prefixes(arch, "onnx")):
            for name in optional_names:
                assert getattr(record, name) is None, (arch, name)

    dense = ArchitecturePrefixes("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
    assert dense.router_prefix is None
    assert dense.vision_prefix is None
    assert dense.speculative_prefix is None

    moe = SAFETENSORS_PREFIXES[ArchitectureType.QWEN3_MOE]
    moe_onnx = ONNX_PREFIXES[ArchitectureType.QWEN3_MOE]
    hybrid = SAFETENSORS_PREFIXES[ArchitectureType.QWEN3_HYBRID]
    hybrid_onnx = ONNX_PREFIXES[ArchitectureType.QWEN3_HYBRID]
    assert moe is not hybrid
    assert moe_onnx is not hybrid_onnx
    assert get_prefixes(ArchitectureType.QWEN3_MOE, "safetensors") is moe
    assert get_prefixes(ArchitectureType.QWEN3_HYBRID, "onnx") is hybrid_onnx
    assert moe.router_prefix == "mlp.gate."
    assert moe.expert_prefix == "mlp.experts."
    assert moe.mtp_head_prefix is None
    assert moe.vision_prefix is None
    assert moe.cross_attn_prefix is None
    assert moe.speculative_prefix is None
    assert "model.layers.0.mlp.gate.weight".find(moe.router_prefix) >= 0
    assert "model.layers.0.mlp.experts.0.gate_proj.weight".find(moe.expert_prefix) >= 0
    assert "mlp.gate_proj.weight".find(moe.router_prefix) < 0
    assert hybrid.router_prefix == moe.router_prefix
    assert hybrid.expert_prefix == moe.expert_prefix
    assert hybrid.mtp_head_prefix == "model.mtp_head."
    assert "model.mtp_head.weight".startswith(hybrid.mtp_head_prefix)
    assert hybrid.vision_prefix is None
    assert hybrid.cross_attn_prefix is None
    assert hybrid.speculative_prefix is None
    assert moe_onnx.router_prefix == "/mlp/gate"
    assert moe_onnx.expert_prefix == "/mlp/experts"
    assert moe_onnx.mtp_head_prefix is None
    assert hybrid_onnx.mtp_head_prefix == "/mtp_head"
    assert "/model/layers.0/mlp/experts/0/gate_proj".find(hybrid_onnx.expert_prefix) >= 0
    assert hybrid_onnx.vision_prefix is None
    assert hybrid_onnx.cross_attn_prefix is None
    assert hybrid_onnx.speculative_prefix is None

    try:
        ArchitecturePrefixes(
            embed_prefix="model.embed_tokens.",
            layer_prefix="model.layers.",
            norm_prefix="model.norm.",
            lm_head_prefix="lm_head.",
            router_prefix=" ",
        )
        raise AssertionError("a blank router prefix was accepted")
    except ValueError as exc:
        assert "router_prefix" in str(exc)
    try:
        ArchitecturePrefixes("", "model.layers.", "model.norm.", "lm_head.")
        raise AssertionError("a blank embed prefix was accepted")
    except ValueError as exc:
        assert "embed_prefix" in str(exc)

    assert classify_qwen_family(["model.layers.0.self_attn.q_norm.weight"]) is ArchitectureType.QWEN3
    assert classify_qwen_family([
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ]) is ArchitectureType.QWEN3_MOE
    assert classify_qwen_family([
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    ]) is ArchitectureType.QWEN3_HYBRID
    assert classify_qwen_family(["model.layers.4.sparse_attention.q_proj.weight"]) is ArchitectureType.QWEN3_HYBRID
    assert classify_qwen_family(["model.mtp_head.weight"]) is ArchitectureType.QWEN3_HYBRID
    assert classify_qwen_family(["model.ngram_embedding.weight"]) is ArchitectureType.QWEN3_HYBRID
    assert classify_qwen_family(["model.layers.0.mlp.experts.0.weight"]) is None
    assert classify_qwen_family(["model.layers.0.self_attn.q_proj.weight"]) is None

    expert = ComponentInfo(component_type="moe_expert", layer_id=3)
    assert expert.component_type is ComponentType.MOE_EXPERT
    assert expert.name == "moe_expert_3"
    expert.validate()
    router = ComponentInfo(component_type=ComponentType.MOE_ROUTER, layer_id=1)
    assert router.name == "moe_router_1"
    gdn = ComponentInfo(component_type="gated_deltanet", layer_id=0)
    assert gdn.name == "gated_deltanet_0"
    gdn.validate()
    sparse = ComponentInfo(component_type="sparse_attention", layer_id=2)
    assert sparse.name == "sparse_attention_2"
    mtp = ComponentInfo(component_type="mtp_head")
    assert mtp.component_type is ComponentType.MTP_HEAD
    assert mtp.name == "mtp_head"
    ngram = ComponentInfo(component_type=ComponentType.NGRAM_EMBEDDING)
    assert ngram.name == "ngram_embedding"
    ngram.validate()
    layer = ComponentInfo(component_type=ComponentType.LAYER, layer_id=4)
    assert layer.name == "layer_4"
    try:
        ComponentInfo(component_type="ngram_embedding", layer_id=1).validate()
        raise AssertionError("an n-gram embedding accepted a layer id")
    except ValueError as exc:
        assert "layer id" in str(exc)
    try:
        ComponentInfo(component_type="mtp_head", layer_id=0).validate()
        raise AssertionError("an MTP head accepted a layer id")
    except ValueError as exc:
        assert "layer id" in str(exc)

    assert common.ArchitectureDescriptor is ArchitectureDescriptor
    assert common.get_architecture_descriptor is get_architecture_descriptor
    assert common.ARCHITECTURE_DESCRIPTORS is ARCHITECTURE_DESCRIPTORS
    assert set(ARCHITECTURE_DESCRIPTORS) == set(ArchitectureType)
    unknown = ARCHITECTURE_DESCRIPTORS[ArchitectureType.UNKNOWN]
    assert unknown.supports_gqa is False
    assert unknown.supports_mqa is False
    assert unknown.is_moe is False
    assert unknown.num_experts == 0
    assert unknown.experts_per_token == 0
    assert unknown.has_sliding_window is False
    assert unknown.has_tied_embeddings is False
    assert unknown.attention_variants == []
    assert unknown.max_seq_len == 0
    assert unknown.has_hybrid_attention is False
    assert unknown.has_mtp_head is False
    assert unknown.has_ngram_embedding is False
    assert unknown.quantization_format == ""
    for record in ARCHITECTURE_DESCRIPTORS.values():
        assert record.supports_mqa is False
        assert record.max_seq_len == 0

    qwen3 = get_architecture_descriptor(ArchitectureType.QWEN3)
    assert qwen3 is ARCHITECTURE_DESCRIPTORS[ArchitectureType.QWEN3]
    assert qwen3.supports_gqa is True
    assert qwen3.is_moe is False
    assert qwen3.has_hybrid_attention is False
    assert get_architecture_descriptor("qwen3") is qwen3
    assert ARCHITECTURE_DESCRIPTORS[ArchitectureType.QWEN].supports_gqa is False
    assert ARCHITECTURE_DESCRIPTORS[ArchitectureType.LLAMA].supports_gqa is True

    moe_desc = get_architecture_descriptor("qwen3_moe")
    assert moe_desc.is_moe is True
    assert moe_desc.supports_gqa is True
    assert moe_desc.num_experts == 0
    assert moe_desc.experts_per_token == 0
    assert moe_desc.has_hybrid_attention is False

    hybrid_desc = get_architecture_descriptor("Qwen3.8")
    assert hybrid_desc.arch_type is ArchitectureType.QWEN3_HYBRID
    assert hybrid_desc is get_architecture_descriptor(ArchitectureType.QWEN3_HYBRID)
    assert hybrid_desc.supports_gqa is True
    assert hybrid_desc.is_moe is True
    assert hybrid_desc.num_experts == 512
    assert hybrid_desc.experts_per_token == 10
    assert hybrid_desc.attention_variants == ["full", "gated_deltanet", "sparse_attention"]
    assert hybrid_desc.has_hybrid_attention is True
    assert hybrid_desc.has_sliding_window is False
    assert hybrid_desc.has_tied_embeddings is False

    mistral = get_architecture_descriptor("mistral")
    assert mistral.supports_gqa is True
    assert mistral.has_sliding_window is True
    assert mistral.attention_variants == ["sliding_window"]
    assert mistral.has_hybrid_attention is False
    gemma = ARCHITECTURE_DESCRIPTORS[ArchitectureType.GEMMA]
    assert gemma.has_tied_embeddings is True
    assert gemma.supports_gqa is True
    gpt2_desc = get_architecture_descriptor("gpt2")
    assert gpt2_desc.has_tied_embeddings is True
    assert gpt2_desc.supports_gqa is False
    assert get_architecture_descriptor("bert") is unknown

    supplied = ["Full", "full", " gated_deltanet "]
    custom = ArchitectureDescriptor(
        "qwen3_hybrid",
        is_moe=True,
        num_experts=512,
        experts_per_token=10,
        attention_variants=supplied,
        max_seq_len=8192,
    )
    assert custom.arch_type is ArchitectureType.QWEN3_HYBRID
    assert custom.attention_variants == ["full", "gated_deltanet"]
    assert custom.attention_variants is not supplied
    assert custom.max_seq_len == 8192
    supplied.append("sparse_attention")
    assert custom.attention_variants == ["full", "gated_deltanet"]
    first = ArchitectureDescriptor(ArchitectureType.LLAMA)
    second = ArchitectureDescriptor(ArchitectureType.LLAMA)
    assert first.attention_variants is not second.attention_variants
    try:
        qwen3.supports_gqa = False
        raise AssertionError("a descriptor field was changed")
    except AttributeError:
        pass
    try:
        ArchitectureDescriptor(ArchitectureType.LLAMA, supports_gqa=1)
        raise AssertionError("an int was accepted as a bool")
    except ValueError as exc:
        assert "supports_gqa" in str(exc)
    try:
        ArchitectureDescriptor(ArchitectureType.LLAMA, num_experts=8)
        raise AssertionError("expert counts were accepted without MoE")
    except ValueError as exc:
        assert "is_moe" in str(exc)
    try:
        ArchitectureDescriptor(ArchitectureType.QWEN3_MOE, is_moe=True, experts_per_token=4)
        raise AssertionError("a top-k was accepted without an expert count")
    except ValueError as exc:
        assert "num_experts" in str(exc)
    try:
        ArchitectureDescriptor(
            ArchitectureType.QWEN3_HYBRID,
            is_moe=True,
            num_experts=8,
            experts_per_token=10,
        )
        raise AssertionError("a top-k above the expert count was accepted")
    except ValueError as exc:
        assert "exceeds" in str(exc)
    try:
        ArchitectureDescriptor(ArchitectureType.LLAMA, max_seq_len=-1)
        raise AssertionError("a negative context length was accepted")
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    try:
        ArchitectureDescriptor(ArchitectureType.LLAMA, attention_variants="full")
        raise AssertionError("a string was accepted as attention variants")
    except ValueError as exc:
        assert "attention_variants" in str(exc)
    try:
        get_architecture_descriptor(None)
        raise AssertionError("a missing architecture was accepted")
    except ValueError as exc:
        assert "arch" in str(exc)

    assert common.QUANTIZED_OP_PATTERNS is QUANTIZED_OP_PATTERNS
    assert common.quantized_op_scheme is quantized_op_scheme
    assert QUANTIZED_OP_PATTERNS == {
        "MatMulNBits": "int4_weight",
        "QLinearMatMul": "int8_quantized",
        "AWQMatMul": "awq_int4",
    }
    q_proj = "/model/layers.0/self_attn/q_proj/MatMulNBits"
    gate_proj = "/model/layers.0/mlp/gate_proj/MatMulNBits"
    layer_prefix = get_prefixes(ArchitectureType.LLAMA, "onnx").layer_prefix
    assert layer_prefix == "/model/layers."
    assert layer_prefix in q_proj
    assert layer_prefix in gate_proj
    assert quantized_op_scheme(q_proj) == "int4_weight"
    assert quantized_op_scheme(gate_proj) == "int4_weight"
    assert quantized_op_scheme("MatMulNBits") == "int4_weight"
    assert quantized_op_scheme("matmulnbits") == "int4_weight"
    assert quantized_op_scheme("MatMulNBits_k_6_0") == "int4_weight"
    assert quantized_op_scheme("/model/layers.1/mlp/down_proj/QLinearMatMul") == "int8_quantized"
    assert quantized_op_scheme("QLinearMatMul") == "int8_quantized"
    assert quantized_op_scheme("model.layers.2.mlp.gate_proj.AWQMatMul") == "awq_int4"
    assert quantized_op_scheme("AWQMatMul") == "awq_int4"
    assert quantized_op_scheme("/model/layers.0/self_attn/q_proj/MatMul") is None
    assert quantized_op_scheme("MatMul") is None
    try:
        quantized_op_scheme("  ")
        raise AssertionError("a blank node name was accepted")
    except ValueError as exc:
        assert "node name" in str(exc)
    try:
        quantized_op_scheme(None)
        raise AssertionError("a missing node name was accepted")
    except ValueError as exc:
        assert "node name" in str(exc)

    import sys
    import types
    import chunking_utils
    from chunking_utils.common.architecture_detector import detect_architecture

    try:
        from chunking_utils.onnx.splitter import detect_architecture_from_onnx
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import detect_architecture_from_onnx

    assert common.detect_architecture is detect_architecture
    assert chunking_utils.detect_architecture is detect_architecture

    class _Node:
        def __init__(self, name, op_type="MatMul"):
            self.name = name
            self.op_type = op_type

    class _Init:
        def __init__(self, name):
            self.name = name

    class _Graph:
        def __init__(self, nodes, inits=()):
            self.node = nodes
            self.initializer = list(inits)

    hybrid_graph = _Graph([
        _Node("/model/layers.0/linear_attn/in_proj"),
        _Node("/model/layers.1/sparse_attn/q_proj"),
        _Node("/model/layers.0/mlp/experts/0/gate_proj/MatMulNBits", "MatMulNBits"),
        _Node("/mtp_head"),
        _Node("/model/ngram_embedding"),
    ])
    hybrid_found = detect_architecture(hybrid_graph)
    assert hybrid_found.arch_type is ArchitectureType.QWEN3_HYBRID
    assert hybrid_found.is_moe is True
    assert hybrid_found.attention_variants == ["gdn", "qsa"]
    assert hybrid_found.has_mtp_head is True
    assert hybrid_found.has_ngram_embedding is True
    assert hybrid_found.quantization_format == "matmulnbits"
    assert detect_architecture_from_onnx(hybrid_graph) is ArchitectureType.QWEN3_HYBRID

    dense_qwen = detect_architecture(_Graph([
        _Node("/model/layers.0/self_attn/q_norm"),
        _Node("/model/layers.0/mlp/gate_proj/MatMul"),
    ]))
    assert dense_qwen.arch_type is ArchitectureType.QWEN3
    assert dense_qwen.is_moe is False
    assert dense_qwen.attention_variants == []
    assert dense_qwen.quantization_format == ""

    moe_found = detect_architecture(_Graph([
        _Node("/model/layers.0/self_attn/q_norm"),
        _Node("/model/layers.0/mlp/experts/0/up_proj"),
        _Node("/model/layers.0/mlp/gate/MatMul"),
    ]))
    assert moe_found.arch_type is ArchitectureType.QWEN3_MOE
    assert moe_found.is_moe is True
    assert moe_found.has_mtp_head is False

    ryzen = detect_architecture(_Graph([
        _Node("gqo_8_1_0"),
        _Node("ssmlp_3_0_0"),
        _Node("MatMulNBits_k_6_0", "MatMulNBits"),
    ]))
    assert ryzen.arch_type is ArchitectureType.LLAMA
    assert ryzen.is_moe is False
    assert ryzen.attention_variants == []
    assert ryzen.quantization_format == "matmulnbits"

    configured = detect_architecture(
        _Graph([
            _Node("gqo_8_1_0"),
            _Node("ssmlp_3_0_0"),
            _Node("MatMulNBits_k_6_0", "MatMulNBits"),
        ]),
        config_json={
            "model_type": "qwen3.8",
            "layer_types": ["linear_attention", "sparse_attention"],
            "num_experts": 512,
            "num_experts_per_tok": 10,
        },
    )
    assert configured.arch_type is ArchitectureType.QWEN3_HYBRID
    assert configured.is_moe is True
    assert configured.num_experts == 512
    assert configured.experts_per_token == 10
    assert configured.attention_variants == ["gdn", "qsa"]
    assert configured.quantization_format == "matmulnbits"
    assert configured.supports_gqa is True

    from_info = detect_architecture(
        _Graph([_Node("/model/layers.0/self_attn/q_proj")]),
        config_json={"model_type": "llama"},
        model_info={
            "model_type": "qwen3_moe",
            "num_experts": 64,
            "num_experts_per_token": 4,
        },
    )
    assert from_info.arch_type is ArchitectureType.QWEN3_MOE
    assert from_info.num_experts == 64
    assert from_info.experts_per_token == 4

    from_genai = detect_architecture(
        _Graph([_Node("/model/layers.0/self_attn/q_norm/MatMulNBits", "MatMulNBits")]),
        config_json='{"quantization_config": {"quant_method": "awq"}}',
        genai_config={"model": {"decoder": {"model_type": "llama"}}},
    )
    assert from_genai.arch_type is ArchitectureType.QWEN3
    assert from_genai.quantization_format == "awq"

    ngram_only = detect_architecture(_Graph([], [_Init("model.ngram_embedding.weight")]))
    assert ngram_only.arch_type is ArchitectureType.QWEN3_HYBRID
    assert ngram_only.has_ngram_embedding is True

    flagged = detect_architecture(
        None,
        config_json={
            "model_type": "qwen3_hybrid",
            "has_mtp_head": True,
            "has_ngram_embedding": True,
        },
    )
    assert flagged.arch_type is ArchitectureType.QWEN3_HYBRID
    assert flagged.is_moe is True
    assert flagged.has_mtp_head is True
    assert flagged.has_ngram_embedding is True

    assert detect_architecture(None).arch_type is ArchitectureType.UNKNOWN
    assert detect_architecture(_Graph([])).arch_type is ArchitectureType.LLAMA
    assert detect_architecture(_Graph([_Node("/transformer/h.0/attn")])).arch_type is ArchitectureType.GPT2
    assert detect_architecture_from_onnx(_Graph([])) is ArchitectureType.LLAMA

    try:
        detect_architecture(_Graph([]), config_json="{")
        raise AssertionError("broken config json was accepted")
    except ValueError as exc:
        assert "config_json" in str(exc)
    try:
        detect_architecture(1)
        raise AssertionError("a non-graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        detect_architecture(
            None,
            model_info={"model_type": "qwen3_moe", "num_experts": 4, "num_experts_per_tok": 8},
        )
        raise AssertionError("a top-k above the expert count was accepted")
    except ValueError as exc:
        assert "exceeds" in str(exc)
    try:
        ArchitectureDescriptor(ArchitectureType.LLAMA, quantization_format="int4/weight")
        raise AssertionError("a quantization path was accepted")
    except ValueError as exc:
        assert "quantization_format" in str(exc)
    stored = ArchitectureDescriptor(ArchitectureType.LLAMA, quantization_format="matmul_nbits")
    assert stored.quantization_format == "matmulnbits"
    copied = get_architecture_descriptor(ArchitectureType.LLAMA).to_dict()
    copied["attention_variants"].append("full")
    assert get_architecture_descriptor(ArchitectureType.LLAMA).attention_variants == []
    assert copied["arch_type"] == "llama"
    assert copied["supports_gqa"] is True
    assert copied["has_sliding_window"] is False
    assert copied["has_tied_embeddings"] is False
    assert copied["is_moe"] is False
    assert copied["has_hybrid_attention"] is False

    grouped = detect_architecture(
        None,
        config_json={
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
    )
    assert grouped.supports_gqa is True
    assert grouped.supports_mqa is False
    multi_query = detect_architecture(
        None,
        config_json={
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
        },
    )
    assert multi_query.supports_gqa is False
    assert multi_query.supports_mqa is True
    matched_heads = detect_architecture(
        None,
        config_json={
            "model_type": "llama",
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
        },
    )
    assert matched_heads.supports_gqa is False
    assert matched_heads.supports_mqa is False
    window = detect_architecture(
        None,
        config_json={"model_type": "llama", "sliding_window": 4096},
    )
    assert window.has_sliding_window is True
    closed_window = detect_architecture(
        None,
        config_json={"model_type": "mistral", "use_sliding_window": False},
    )
    assert closed_window.has_sliding_window is False
    tied = detect_architecture(
        None,
        config_json={"model_type": "llama", "tie_word_embeddings": True},
    )
    assert tied.has_tied_embeddings is True
    untied = detect_architecture(
        None,
        config_json={"model_type": "gemma", "tie_word_embeddings": False},
    )
    assert untied.has_tied_embeddings is False
    silent = detect_architecture(None, config_json={"model_type": "gemma", "tie_word_embeddings": 1})
    assert silent.has_tied_embeddings is True
    try:
        detect_architecture(
            None,
            config_json={
                "model_type": "llama",
                "num_attention_heads": 4,
                "num_key_value_heads": 8,
            },
        )
        raise AssertionError("more kv heads than query heads were accepted")
    except ValueError as exc:
        assert "num_key_value_heads" in str(exc)

    print("  architecture prefixes: dense llama record, moe router and expert, hybrid mtp head")
    print("  architecture names: longest qwen3_moe, dense qwen3, qwen2.5")
    print("  quantized onnx nodes: MatMulNBits int4, QLinearMatMul int8, AWQMatMul awq")
    print("  architecture descriptor: gqa, moe counts, sliding window, tied embeddings, hybrid attention")
    print("  checkpoint head counts, sliding window, and tied embeddings replace the family default")
    print("  architecture detector: hybrid gdn qsa, mtp, ngram, matmulnbits, ryzenai config")
    print("architecture prefixes: OK\n")
    return True


def test_graph_topology():
    """Topology maps, dependency closures, and config-aware component grouping."""
    print("=" * 50)
    print("Testing graph topology...")
    print("=" * 50)

    import sys
    import types
    import chunking_utils
    import chunking_utils.common as common
    from chunking_utils.common.graph_topology import (
        analyze_graph_topology,
        GraphTopology,
        NodeGroup,
    )

    assert common.analyze_graph_topology is analyze_graph_topology
    assert chunking_utils.analyze_graph_topology is analyze_graph_topology
    assert common.GraphTopology is GraphTopology
    assert common.NodeGroup is NodeGroup

    try:
        from chunking_utils.onnx.splitter import extract_components
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import extract_components

    class _Node:
        def __init__(self, name, inputs, outputs, op_type="MatMul"):
            self.name = name
            self.input = list(inputs)
            self.output = list(outputs)
            self.op_type = op_type

    class _Init:
        def __init__(self, name):
            self.name = name

    class _Out:
        def __init__(self, name):
            self.name = name

    class _Graph:
        def __init__(self, nodes, inits=(), outputs=()):
            self.node = list(nodes)
            self.initializer = list(inits)
            self.output = list(outputs)

    class _Model:
        def __init__(self, graph):
            self.graph = graph

    def names_of(topology, nodes, group_name):
        group = next(item for item in topology.groups if item.name == group_name)
        return [nodes[index].name for index in group.node_indices]

    embed = _Node("/model/embed_tokens/Gather", ["input_ids", "embed_weight"], ["hidden"])
    layer0 = _Node("/model/layers.0/self_attn/MatMul", ["hidden", "w0"], ["h0"])
    layer1 = _Node("/model/layers.1/mlp/MatMul", ["h0", "w1"], ["h1"])
    norm = _Node("/model/norm/Mul", ["h1", "norm_weight"], ["hn"])
    lm_head = _Node("/lm_head/MatMul", ["hn", "embed_weight"], ["logits"])
    topology_nodes = [embed, layer0, layer1, norm, lm_head]
    clean = _Graph(
        topology_nodes,
        [_Init("embed_weight"), _Init("w0"), _Init("w1"), _Init("norm_weight"), _Init("unused_weight")],
        [_Out("logits")],
    )
    clean_topology = analyze_graph_topology(clean)
    assert clean_topology.producer_map["hidden"] == 0
    assert clean_topology.producer_map["h0"] == 1
    assert clean_topology.consumer_map["hidden"] == [1]
    assert clean_topology.consumer_map["h0"] == [2]
    assert clean_topology.initializer_ownership["w0"] == [1]
    assert clean_topology.initializer_ownership["embed_weight"] == [0, 4]
    assert clean_topology.initializer_ownership["unused_weight"] == []
    assert clean_topology.topological_order == [0, 1, 2, 3, 4]
    assert [group.name for group in clean_topology.groups] == [
        "embedding",
        "layer_0",
        "layer_1",
        "norm",
        "lm_head",
    ]
    assert names_of(clean_topology, topology_nodes, "layer_0") == ["/model/layers.0/self_attn/MatMul"]
    assert names_of(clean_topology, topology_nodes, "lm_head") == ["/lm_head/MatMul"]

    genai_norm = _Graph([
        _Node("/model/embed_tokens/Gather", ["input_ids"], ["hidden"]),
        _Node("/model/layers.0/self_attn/MatMul", ["hidden"], ["h0"]),
        _Node("/model/layers.1/final_norm_layernorm/SkipLayerNorm", ["h0"], ["hn"]),
        _Node("/lm_head/MatMul", ["hn"], ["logits"]),
    ], outputs=[_Out("logits")])
    genai_norm_topology = analyze_graph_topology(genai_norm)
    assert [group.name for group in genai_norm_topology.groups] == [
        "embedding",
        "layer_0",
        "norm",
        "lm_head",
    ]

    reversed_nodes = [lm_head, norm, layer1, layer0, embed]
    reversed_graph = _Graph(
        reversed_nodes,
        [_Init("embed_weight"), _Init("w0"), _Init("w1"), _Init("norm_weight")],
        [_Out("logits")],
    )
    reversed_topology = analyze_graph_topology(reversed_graph)
    assert reversed_topology.topological_order == [4, 3, 2, 1, 0]
    assert [reversed_nodes[index].name for index in reversed_topology.topological_order] == [
        "/model/embed_tokens/Gather",
        "/model/layers.0/self_attn/MatMul",
        "/model/layers.1/mlp/MatMul",
        "/model/norm/Mul",
        "/lm_head/MatMul",
    ]

    prologue = _Node("pos_add", ["input_ids"], ["positions"])
    fused = _Node("fused_attn_kernel", ["a0"], ["a1"])
    start = _Node("/model/layers.0/self_attn/q_proj", ["hidden"], ["a0"])
    end = _Node("/model/layers.0/mlp/down_proj", ["a1"], ["h0"])
    tail = _Node("tail_matmul", ["h0"], ["logits"])
    mtp = _Node("/mtp_head", ["logits"], ["mtp_out"])
    ngram = _Node("/model/ngram_embedding", ["input_ids"], ["ng"])
    fused_graph = _Graph(
        [
            prologue,
            _Node("/model/embed_tokens/Gather", ["positions", "embed_weight"], ["hidden"]),
            start,
            fused,
            end,
            _Node("/model/norm/Mul", ["h0"], ["hn"]),
            _Node("/lm_head/MatMul", ["hn"], ["logits"]),
            tail,
            mtp,
            ngram,
        ],
        [_Init("embed_weight")],
        [_Out("logits"), _Out("mtp_out")],
    )
    fused_topology = analyze_graph_topology(fused_graph)
    assert names_of(fused_topology, fused_graph.node, "embedding") == [
        "pos_add",
        "/model/embed_tokens/Gather",
    ]
    assert names_of(fused_topology, fused_graph.node, "layer_0") == [
        "/model/layers.0/self_attn/q_proj",
        "fused_attn_kernel",
        "/model/layers.0/mlp/down_proj",
    ]
    assert names_of(fused_topology, fused_graph.node, "other") == [
        "/model/ngram_embedding",
        "tail_matmul",
        "/mtp_head",
    ]

    sparse_side = _Node("Gather", ["q"], ["idx"])
    sparse = _Graph([
        _Node("/model/layers.1/sparse_attn/q_proj", ["hidden"], ["q"]),
        sparse_side,
        _Node("/model/layers.1/sparse_attn/out_proj", ["q"], ["h1"]),
    ])
    sparse_topology = analyze_graph_topology(sparse)
    assert names_of(sparse_topology, sparse.node, "layer_1") == [
        "/model/layers.1/sparse_attn/q_proj",
        "Gather",
        "/model/layers.1/sparse_attn/out_proj",
    ]
    assert sparse_topology.consumer_map["q"] == [1, 2]

    expert = _Graph([
        _Node("/model/layers.0/mlp/experts/0/up_proj", ["hidden"], ["e0"]),
        _Node("fused_expert_gemm", ["e0"], ["e1"]),
        _Node("/model/layers.0/mlp/experts/0/down_proj", ["e1"], ["h0"]),
        _Node("/model/layers.0/mlp/gate/MatMul", ["hidden"], ["gate"]),
    ])
    expert_topology = analyze_graph_topology(expert)
    assert names_of(expert_topology, expert.node, "layer_0") == [
        "/model/layers.0/mlp/experts/0/up_proj",
        "/model/layers.0/mlp/gate/MatMul",
        "fused_expert_gemm",
        "/model/layers.0/mlp/experts/0/down_proj",
    ]

    quant = _Graph(
        [_Node("MatMulNBits", ["model.layers.2.mlp.experts.0.up_proj.weight", "hidden"], ["y"], "MatMulNBits")],
        [_Init("model.layers.2.mlp.experts.0.up_proj.weight")],
    )
    quant_topology = analyze_graph_topology(quant)
    assert quant_topology.groups[0].name == "layer_2"
    assert quant_topology.groups[0].component_type == "layer"
    assert quant_topology.groups[0].layer_id == 2
    assert quant_topology.initializer_ownership["model.layers.2.mlp.experts.0.up_proj.weight"] == [0]

    mixed = _Graph(
        [_Node("fused", ["model.layers.0.w", "model.layers.1.w"], ["y"])],
        [_Init("model.layers.0.w"), _Init("model.layers.1.w")],
    )
    mixed_topology = analyze_graph_topology(mixed)
    assert mixed_topology.groups[0].name == "other"

    ryzen = _Graph([
        _Node("gqo_8_1_0", ["h"], ["a"]),
        _Node("com.amd.qsa", ["a"], ["b"], "com.amd.SparseAttention"),
        _Node("ssmlp_3_0_0", ["b"], ["c"]),
        _Node("MatMulNBits_k_6_1", ["c"], ["d"], "MatMulNBits"),
    ])
    ryzen_topology = analyze_graph_topology(ryzen)
    assert names_of(ryzen_topology, ryzen.node, "layer_0") == [
        "gqo_8_1_0",
        "com.amd.qsa",
        "ssmlp_3_0_0",
    ]
    assert names_of(ryzen_topology, ryzen.node, "layer_1") == ["MatMulNBits_k_6_1"]

    def anonymous(count, side=False):
        nodes = [_Node("/model/embed_tokens/Gather", ["input_ids"], ["h"])]
        previous = "h"
        for index in range(count):
            output = f"f{index}"
            nodes.append(_Node(f"fused_{index}", [previous, f"w{index}"], [output]))
            previous = output
        if side:
            nodes.append(_Node("side_gather", ["f0"], ["side"]))
        nodes.append(_Node("/model/norm/Mul", [previous], ["hn"]))
        nodes.append(_Node("/lm_head/MatMul", ["hn"], ["logits"]))
        return _Graph(nodes, outputs=[_Out("logits")])

    configured_graph = anonymous(4, side=True)
    configured = analyze_graph_topology(
        configured_graph,
        config_json={"num_hidden_layers": 4},
        model_info='{"num_hidden_layers": 2}',
    )
    assert names_of(configured, configured_graph.node, "layer_0") == [
        "fused_0",
        "fused_1",
        "side_gather",
    ]
    assert names_of(configured, configured_graph.node, "layer_1") == ["fused_2", "fused_3"]

    uneven_graph = anonymous(3)
    uneven = analyze_graph_topology(uneven_graph, config_json={"num_hidden_layers": 2})
    assert [group.name for group in uneven.groups] == ["embedding", "norm", "lm_head", "other"]
    assert names_of(uneven, uneven_graph.node, "other") == ["fused_0", "fused_1", "fused_2"]

    typed_graph = anonymous(2)
    by_types = analyze_graph_topology(
        typed_graph,
        genai_config={"model": {"decoder": {"layer_types": ["linear_attention", "sparse_attention"]}}},
    )
    assert names_of(by_types, typed_graph.node, "layer_0") == ["fused_0"]
    assert names_of(by_types, typed_graph.node, "layer_1") == ["fused_1"]
    assert by_types.groups[1].component_type == "layer"

    cycle = _Graph([
        _Node("/model/layers.0/a", ["t1"], ["t0"]),
        _Node("/model/layers.1/b", ["t0"], ["t1"]),
    ])
    cycle_topology = analyze_graph_topology(cycle)
    assert cycle_topology.topological_order == [0, 1]
    assert [group.name for group in cycle_topology.groups] == ["layer_0", "layer_1"]

    boundary_graph = _Graph([
        _Node("/model/embed_tokens/Gather", ["input_ids"], ["hidden"]),
        _Node("boundary_fused", ["hidden"], ["mid"]),
        _Node("/model/norm/Mul", ["mid"], ["hn"]),
        _Node("/lm_head/MatMul", ["hn"], ["logits"]),
    ], outputs=[_Out("logits")])
    boundary_topology = analyze_graph_topology(boundary_graph)
    assert names_of(boundary_topology, boundary_graph.node, "other") == ["boundary_fused"]
    one_layer = analyze_graph_topology(boundary_graph, config_json={"num_hidden_layers": 1})
    assert names_of(one_layer, boundary_graph.node, "layer_0") == ["boundary_fused"]

    extracted = extract_components(_Model(fused_graph))
    assert extracted["layer_0"].component_type == "layer"
    assert extracted["layer_0"].layer_id == 0
    assert "fused_attn_kernel" in extracted["layer_0"].node_names
    assert extracted["embedding"].initializer_names == {"embed_weight"}
    assert "pos_add" in extracted["embedding"].node_names
    assert "/mtp_head" in extracted["other"].node_names

    try:
        analyze_graph_topology(1)
        raise AssertionError("a non-graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        analyze_graph_topology(None)
        raise AssertionError("a missing graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        analyze_graph_topology(_Graph([]), config_json="{")
        raise AssertionError("broken config json was accepted")
    except ValueError as exc:
        assert "config_json" in str(exc)
    try:
        analyze_graph_topology(_Graph([]), genai_config="  ")
        raise AssertionError("a blank genai config was accepted")
    except ValueError as exc:
        assert "genai_config" in str(exc)
    try:
        extract_components(object())
        raise AssertionError("a model without a graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)

    print("  topology maps: producers, consumers, initializer ownership, topo order")
    print("  dependency closure: fused kernel, moe expert, sparse branch, ryzenai contrib")
    print("  config split: renamed backbone uses num_hidden_layers, side branch stays on its layer")
    print("graph topology: OK\n")
    return True


def test_boundary_contract():
    """Boundary tensors carry dtype, shape, role, device, and zero-copy."""
    print("=" * 50)
    print("Testing boundary contract...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types
    import chunking_utils
    from chunking_utils.common.types import BoundaryTensor, SplitResult

    try:
        from chunking_utils.onnx.splitter import collect_boundary_tensors
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import collect_boundary_tensors

    assert chunking_utils.collect_boundary_tensors is collect_boundary_tensors

    hidden = BoundaryTensor(
        name="hidden_states",
        dtype="float16",
        shape=[1, None, 16],
        producer="embedding",
        consumers=["layer_0"],
    )
    assert hidden.producer == "embedding"
    assert hidden.producer_component == "embedding"
    assert hidden.consumers == ["layer_0"]
    assert hidden.cache_role is None
    assert hidden.device_preference is None
    assert hidden.zero_copy is True
    recorded = hidden.to_dict()
    restored = BoundaryTensor.from_dict(json.loads(json.dumps(recorded)))
    recorded["consumers"].append("norm")
    recorded["consumer_components"].append("lm_head")
    assert hidden.consumer_components == ["layer_0"]
    assert restored.producer_component == "embedding"
    assert restored.shape[1] is None
    assert restored.zero_copy is True
    alias_only = BoundaryTensor.from_dict({
        "name": "attention_mask",
        "dtype": "int64",
        "shape": [1, None],
        "producer": "external",
        "consumers": ["layer_0", "layer_1"],
    })
    assert alias_only.producer_component == "external"
    assert alias_only.consumer_components == ["layer_0", "layer_1"]
    assert alias_only.zero_copy is False
    past = BoundaryTensor(
        name="past_key_values.0.key",
        dtype="float16",
        shape=[1, 8, None, 128],
        producer_component="external",
        consumer_components=["layer_0"],
        is_stateful=False,
        device_preference=None,
    )
    assert past.is_stateful is True
    assert past.cache_role == "past_key"
    assert past.device_preference == "npu"
    assert past.zero_copy is True
    present = BoundaryTensor(
        name="present.0.value",
        dtype="fp16",
        shape=[1, 8, None, 128],
        producer_component="layer_0",
        consumer_components=[],
    )
    assert present.cache_role == "present_value"
    assert present.zero_copy is True
    kept = BoundaryTensor(
        name="past_key_values.0.value",
        dtype="float16",
        shape=[1, 8, None, 128],
        producer_component="external",
        consumer_components=["layer_0"],
        zero_copy=False,
        device_preference="cpu",
    )
    assert kept.cache_role == "past_value"
    assert kept.zero_copy is False
    assert kept.device_preference == "cpu"
    try:
        BoundaryTensor(
            name="hidden_states",
            dtype="float16",
            shape=[1, None, 16],
            producer="embedding",
            producer_component="norm",
            consumer_components=["layer_0"],
        )
        raise AssertionError("disagreeing producers were accepted")
    except ValueError as exc:
        assert "producer" in str(exc)
    try:
        BoundaryTensor(
            name="hidden_states",
            dtype="float16",
            shape=[1, None, 16],
            producer_component="embedding",
            consumers=["norm"],
            consumer_components=["layer_0"],
        )
        raise AssertionError("disagreeing consumers were accepted")
    except ValueError as exc:
        assert "consumers" in str(exc)
    try:
        BoundaryTensor(
            name="past_key_values.0.key",
            dtype="float16",
            shape=[1],
            producer_component="external",
            consumer_components=["layer_0"],
            cache_role="present_key",
        )
        raise AssertionError("a mismatched cache role was accepted")
    except ValueError as exc:
        assert "cache_role" in str(exc)
    try:
        BoundaryTensor(
            name="hidden_states",
            dtype="float16",
            shape=[1],
            producer_component="embedding",
            consumer_components=["layer_0"],
            cache_role="sideband",
        )
        raise AssertionError("an unknown cache role was accepted")
    except ValueError as exc:
        assert "cache_role" in str(exc)
    try:
        BoundaryTensor(
            name="hidden_states",
            dtype="float16",
            shape=[1],
            producer_component="embedding",
            consumer_components=["layer_0"],
            zero_copy=1,
        )
        raise AssertionError("a non-bool zero_copy was accepted")
    except ValueError as exc:
        assert "zero_copy" in str(exc)
    try:
        BoundaryTensor(
            name="  ",
            dtype="float16",
            shape=[1],
            producer_component="embedding",
            consumer_components=["layer_0"],
        )
        raise AssertionError("a blank boundary name was accepted")
    except ValueError as exc:
        assert "name" in str(exc)

    class _Node:
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.input = list(inputs)
            self.output = list(outputs)
            self.op_type = "MatMul"

    class _Init:
        def __init__(self, name):
            self.name = name

    class _Dim:
        def __init__(self, value=None, param=""):
            self.dim_value = 0 if value is None else value
            self.dim_param = param

    class _Value:
        def __init__(self, name, elem, dims):
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(
                    elem_type=elem,
                    shape=types.SimpleNamespace(dim=dims),
                )
            )

    class _Graph:
        def __init__(self):
            self.node = [
                _Node("embed", ["input_ids", "embed_weight"], ["hidden"]),
                _Node("q_proj", ["hidden"], ["q"]),
                _Node(
                    "out_proj",
                    ["q", "past_key_values.0.key", "attention_mask"],
                    ["hidden_1", "present.0.key"],
                ),
                _Node(
                    "layer1",
                    ["hidden_1", "past_key_values.1.key", "attention_mask"],
                    ["hidden_2", "present.1.key"],
                ),
                _Node("final_norm", ["hidden_2"], ["norm_out"]),
                _Node("head", ["norm_out"], ["logits"]),
            ]
            self.initializer = [_Init("embed_weight")]
            self.input = [
                _Value("input_ids", 7, [_Dim(1), _Dim(param="seq")]),
                _Value("past_key_values.0.key", 10, [_Dim(1), _Dim(8), _Dim(param="seq"), _Dim(128)]),
                _Value("past_key_values.1.key", 10, [_Dim(1), _Dim(8), _Dim(param="seq"), _Dim(128)]),
                _Value("attention_mask", 7, [_Dim(1), _Dim(param="seq")]),
            ]
            self.output = [
                _Value("logits", 10, [_Dim(1), _Dim(param="seq"), _Dim(16)]),
                _Value("present.0.key", 10, [_Dim(1), _Dim(8), _Dim(param="seq"), _Dim(128)]),
                _Value("present.1.key", 10, [_Dim(1), _Dim(8), _Dim(param="seq"), _Dim(128)]),
            ]
            self.value_info = [
                _Value("hidden", 10, [_Dim(1), _Dim(param="seq"), _Dim(16)]),
                _Value("hidden_1", 10, [_Dim(1), _Dim(param="seq"), _Dim(16)]),
                _Value("q", 10, [_Dim(1), _Dim(param="seq"), _Dim(16)]),
            ]

    class _Part:
        def __init__(self, indices):
            self.node_indices = list(indices)

    graph = _Graph()
    components = {
        "embedding": _Part([0]),
        "layer_0": _Part([1, 2]),
        "layer_1": _Part([3]),
        "norm": _Part([4]),
        "lm_head": _Part([5]),
    }
    found = collect_boundary_tensors(graph, components)
    by_name = {item.name: item for item in found}
    assert "q" not in by_name
    assert "embed_weight" not in by_name
    assert by_name["hidden"].producer == "embedding"
    assert by_name["hidden"].consumers == ["layer_0"]
    assert by_name["hidden"].dtype == "float16"
    assert by_name["hidden"].shape == [1, None, 16]
    assert by_name["hidden"].zero_copy is True
    assert by_name["hidden"].cache_role is None
    assert by_name["past_key_values.0.key"].producer == "external"
    assert by_name["past_key_values.0.key"].consumers == ["layer_0"]
    assert by_name["past_key_values.0.key"].cache_role == "past_key"
    assert by_name["past_key_values.0.key"].zero_copy is True
    assert by_name["past_key_values.0.key"].device_preference == "npu"
    assert by_name["past_key_values.0.key"].shape[2] is None
    assert by_name["present.0.key"].producer == "layer_0"
    assert by_name["present.0.key"].consumers == []
    assert by_name["present.0.key"].cache_role == "present_key"
    assert by_name["present.0.key"].zero_copy is True
    assert by_name["attention_mask"].producer == "external"
    assert by_name["attention_mask"].consumers == ["layer_0", "layer_1"]
    assert by_name["attention_mask"].dtype == "int64"
    assert by_name["attention_mask"].zero_copy is False
    assert by_name["logits"].producer == "lm_head"
    assert by_name["logits"].consumers == []
    assert by_name["logits"].zero_copy is False
    assert by_name["logits"].cache_role is None
    assert by_name["input_ids"].dtype == "int64"
    names = [item.name for item in found]
    assert names.index("input_ids") < names.index("hidden")
    document = [item.to_dict() for item in found]
    assert document[0]["producer"] == document[0]["producer_component"]
    assert "cache_role" in document[0]
    assert "zero_copy" in document[0]
    BoundaryTensor.from_dict(document[0])
    try:
        collect_boundary_tensors(None, {})
        raise AssertionError("a missing graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        collect_boundary_tensors(1, {})
        raise AssertionError("a non-graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        collect_boundary_tensors(graph, {"embedding": _Part([0])})
        raise AssertionError("an unassigned node was accepted")
    except ValueError as exc:
        assert "node 1" in str(exc)
    try:
        collect_boundary_tensors(graph, {
            "embedding": _Part([0]),
            "layer_0": _Part([0, 1, 2, 3, 4, 5]),
        })
        raise AssertionError("a shared node was accepted")
    except ValueError as exc:
        assert "belongs" in str(exc)
    graph.value_info.append(_Value("logits", 99, [_Dim(1)]))
    try:
        collect_boundary_tensors(graph, components)
        raise AssertionError("an unknown element type was accepted")
    except ValueError as exc:
        assert "logits" in str(exc)
    graph.value_info.pop()

    with tempfile.TemporaryDirectory() as directory:
        embed_path = os.path.join(directory, "embedding.onnx")
        layer_path = os.path.join(directory, "layer_0.onnx")
        for path in (embed_path, layer_path):
            with open(path, "wb") as handle:
                handle.write(b"onnx")
        result = SplitResult(
            output_dir=directory,
            num_layers=1,
            component_files={"embedding": embed_path, "layer_0": layer_path},
            model_info={"boundary_tensors": [past.to_dict()]},
            boundary_tensors=[past],
        )
        result.validate()
        assert result.model_info["boundary_tensors"][0]["cache_role"] == "past_key"
        assert result.boundary_tensors[0].zero_copy is True

    print("boundary contract: OK")
    print("  dtype, shape, producer, and consumers")
    print("  KV cache role prefers the NPU and stays zero-copy")
    print("  model_info stores the boundary list")
    return True


def test_output_detection():
    """Component outputs come from one consumer map."""
    print("=" * 50)
    print("Testing output detection...")
    print("=" * 50)

    import sys
    import types
    import chunking_utils

    try:
        from chunking_utils.onnx.splitter import create_component_onnx
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace(FLOAT16=10, INT64=7, INT32=6)
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import create_component_onnx
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.create_component_onnx is create_component_onnx

    class _Inputs(list):
        iterations = 0
        contains_calls = 0

        def __iter__(self):
            type(self).iterations += 1
            return super().__iter__()

        def __contains__(self, item):
            type(self).contains_calls += 1
            return super().__contains__(item)

    class _Node:
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.input = _Inputs(inputs)
            self.output = list(outputs)
            self.op_type = "MatMul"

    class _Init:
        def __init__(self, name):
            self.name = name

    class _Value:
        def __init__(self, name, elem=10):
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(elem_type=elem)
            )

    class _Graph:
        def __init__(self, nodes, outputs, inputs=None, initializers=None, value_info=None):
            self.node = list(nodes)
            self.output = list(outputs)
            self.input = list(inputs or [])
            self.initializer = list(initializers or [])
            self.value_info = list(value_info or [])

    class _Model:
        def __init__(self, graph):
            self.graph = graph
            self.opset_import = []
            self.ir_version = 8

    captured = {}

    def make_tensor_value_info(name, elem_type, shape):
        return _Value(name, elem_type)

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["outputs"] = [item.name for item in outputs]
        captured["inputs"] = [item.name for item in inputs]
        return types.SimpleNamespace(
            node=nodes,
            name=name,
            input=inputs,
            output=outputs,
            initializer=initializer,
        )

    def make_model(graph, producer_name, opset_imports):
        model = types.SimpleNamespace(
            graph=graph,
            producer_name=producer_name,
            opset_import=opset_imports,
            ir_version=0,
        )
        return model

    original_helper = splitter.helper
    splitter.helper = types.SimpleNamespace(
        make_tensor_value_info=make_tensor_value_info,
        make_graph=make_graph,
        make_model=make_model,
    )
    try:
        _Inputs.iterations = 0
        _Inputs.contains_calls = 0
        embed = _Node("embed", ["input_ids", ""], ["hidden"])
        q_proj = _Node("q_proj", ["hidden", "wq"], ["q"])
        readers = [
            _Node(f"read_{index}", ["q"], [f"side_{index}"])
            for index in range(40)
        ]
        out_proj = _Node("out_proj", ["q"], ["hidden_1", "present.0.key"])
        nxt = _Node("next", ["hidden_1"], ["logits"])
        nodes = [embed, q_proj, *readers, out_proj, nxt]
        out_index = len(nodes) - 2
        next_index = len(nodes) - 1
        values = [
            _Value(name)
            for name in (
                "input_ids",
                "hidden",
                "q",
                "hidden_1",
                "present.0.key",
                "logits",
                "scratch",
                "second",
                "first",
            )
        ]
        graph = _Graph(
            nodes,
            outputs=[_Value("logits"), _Value("present.0.key")],
            inputs=[_Value("input_ids", 7)],
            initializers=[_Init("wq")],
            value_info=values,
        )
        model = _Model(graph)
        layer = ComponentInfo(
            component_type="layer",
            layer_id=0,
            node_indices=[1, *range(2, out_index), out_index],
        )
        created = create_component_onnx(model, layer, "layer_0")
        assert captured["outputs"] == ["hidden_1", "present.0.key"]
        assert created.graph.output[0].name == "hidden_1"
        assert "q" not in captured["outputs"]
        assert all(not name.startswith("side_") for name in captured["outputs"])
        assert captured["inputs"] == ["hidden"]
        assert _Inputs.contains_calls == 0
        # required-input scan plus one consumer-map pass
        assert _Inputs.iterations == len(layer.node_indices) + len(nodes)

        embedding = ComponentInfo(component_type="embedding", node_indices=[0])
        create_component_onnx(model, embedding, "embedding")
        assert captured["outputs"] == ["hidden"]

        head = ComponentInfo(component_type="lm_head", node_indices=[next_index])
        create_component_onnx(model, head, "lm_head")
        assert captured["outputs"] == ["logits"]
        assert captured["inputs"] == ["hidden_1"]

        closed = ComponentInfo(
            component_type="layer",
            layer_id=1,
            node_indices=[out_index, next_index],
        )
        graph.output = [_Value("present.0.key")]
        create_component_onnx(model, closed, "layer_1")
        assert captured["outputs"] == ["present.0.key"]

        solo = _Node("solo", ["hidden"], ["scratch"])
        dead = _Model(_Graph(
            [solo],
            outputs=[],
            inputs=[_Value("hidden")],
            value_info=values,
        ))
        create_component_onnx(
            dead,
            ComponentInfo(component_type="other", node_indices=[0]),
            "other",
        )
        assert captured["outputs"] == ["scratch"]

        later = _Node("later", [], ["second"])
        earlier = _Node("earlier", [], ["first"])
        ordered = _Model(_Graph(
            [earlier, later],
            outputs=[_Value("first"), _Value("second")],
            value_info=values,
        ))
        create_component_onnx(
            ordered,
            ComponentInfo(component_type="other", node_indices=[1, 0]),
            "ordered",
        )
        assert captured["outputs"] == ["second", "first"]
        assert _Inputs.contains_calls == 0
    finally:
        splitter.helper = original_helper

    print("output detection: OK")
    print("  one consumer map selects tensors read outside the component")
    print("  a graph output stays an output")
    print("  an internal tensor stays inside the component")
    return True


def test_dtype_resolution():
    """Dtype and shape use value info, then an initializer, then a warned heuristic."""
    print("=" * 50)
    print("Testing dtype resolution...")
    print("=" * 50)

    import sys
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.splitter import (
            collect_boundary_tensors,
            create_component_onnx,
            resolve_tensor_type,
        )
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import (
            collect_boundary_tensors,
            create_component_onnx,
            resolve_tensor_type,
        )
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.resolve_tensor_type is resolve_tensor_type

    class _Dim:
        def __init__(self, value=None, param=""):
            self.dim_value = 0 if value is None else value
            self.dim_param = param

    class _Value:
        def __init__(self, name, elem, dims):
            shape = None if dims is None else types.SimpleNamespace(dim=dims)
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(elem_type=elem, shape=shape)
            )

    class _Init:
        def __init__(self, data_type, dims):
            self.data_type = data_type
            self.dims = dims

    def _quiet(name, value=None, initializer=None, role="boundary"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_tensor_type(name, value, initializer, role=role)
        return resolved, caught

    declared = _Value("attention_mask", 1, [_Dim(1), _Dim(8)])
    resolved, caught = _quiet(
        "attention_mask",
        declared,
        _Init(3, [4]),
    )
    assert resolved.dtype == "float32"
    assert resolved.elem_type == 1
    assert resolved.shape == [1, 8]
    assert resolved.dtype_source == "value_info"
    assert resolved.shape_source == "value_info"
    assert caught == []
    resolved.shape.append(9)
    again, _ = _quiet("attention_mask", declared, _Init(3, [4]))
    assert again.shape == [1, 8]

    resolved, caught = _quiet("attention_mask", None, _Init(3, [4, 4]))
    assert resolved.dtype == "int8"
    assert resolved.elem_type == 3
    assert resolved.shape == [4, 4]
    assert resolved.dtype_source == "initializer"
    assert resolved.shape_source == "initializer"
    assert caught == []

    resolved, caught = _quiet(
        "hidden",
        _Value("hidden", 1, None),
        _Init(0, [2, 2]),
    )
    assert resolved.dtype == "float32"
    assert resolved.elem_type == 1
    assert resolved.shape == [2, 2]
    assert resolved.dtype_source == "value_info"
    assert resolved.shape_source == "initializer"
    assert caught == []

    resolved, caught = _quiet("hidden", _Value("hidden", 1, None))
    assert resolved.dtype == "float32"
    assert resolved.shape == [None]
    assert resolved.dtype_source == "value_info"
    assert resolved.shape_source == "heuristic"
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "heuristic" in str(caught[0].message)
    assert "shape" in str(caught[0].message)
    assert "dtype" not in str(caught[0].message)

    resolved, caught = _quiet("hidden_4D")
    assert resolved.dtype == "float16"
    assert resolved.elem_type == 10
    assert resolved.shape == [None, None, None, None]
    assert resolved.dtype_source == "heuristic"
    assert resolved.shape_source == "heuristic"
    assert "dtype float16" in str(caught[0].message)
    assert "shape" in str(caught[0].message)

    resolved, caught = _quiet("attention_mask", role="output")
    assert resolved.dtype == "int64"
    assert resolved.elem_type == 7
    assert resolved.shape == [None]
    assert "output tensor attention_mask" in str(caught[0].message)
    resolved, _ = _quiet("token_count", role="input")
    assert resolved.dtype == "int64"
    resolved, _ = _quiet("cache_2D")
    assert resolved.shape == [None, None]
    assert resolved.dtype == "float16"

    try:
        resolve_tensor_type("  ")
        raise AssertionError("a blank tensor name was accepted")
    except ValueError as exc:
        assert "name" in str(exc)
    try:
        resolve_tensor_type("logits", _Value("logits", 99, [_Dim(1)]))
        raise AssertionError("an unknown element type was accepted")
    except ValueError as exc:
        assert "logits" in str(exc)
    try:
        resolve_tensor_type("weight", None, _Init(True, [1]))
        raise AssertionError("a bool data_type was accepted")
    except ValueError as exc:
        assert "data_type" in str(exc)
    try:
        resolve_tensor_type("weight", None, _Init(1, [-1]))
        raise AssertionError("a negative initializer dim was accepted")
    except ValueError as exc:
        assert "negative" in str(exc)
    try:
        resolve_tensor_type("hidden", role="side")
        raise AssertionError("an unknown role was accepted")
    except ValueError as exc:
        assert "role" in str(exc)

    class _Node:
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.input = list(inputs)
            self.output = list(outputs)

    class _Graph:
        def __init__(self, nodes, outputs, inputs=None, initializers=None, value_info=None):
            self.node = list(nodes)
            self.output = list(outputs)
            self.input = list(inputs or [])
            self.initializer = list(initializers or [])
            self.value_info = list(value_info or [])

    class _Model:
        def __init__(self, graph):
            self.graph = graph
            self.opset_import = []
            self.ir_version = 8

    made = []

    def make_tensor_value_info(name, elem_type, shape):
        made.append((name, elem_type, list(shape)))
        return _Value(name, elem_type, [_Dim(dim) for dim in shape if dim is not None] or [])

    def make_graph(nodes, name, inputs, outputs, initializer):
        return types.SimpleNamespace(
            node=nodes, name=name, input=inputs, output=outputs, initializer=initializer
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(
            graph=graph, producer_name=producer_name, opset_import=opset_imports, ir_version=0
        )

    original_helper = splitter.helper
    splitter.helper = types.SimpleNamespace(
        make_tensor_value_info=make_tensor_value_info,
        make_graph=make_graph,
        make_model=make_model,
    )
    try:
        known = _Value("hidden", 1, [_Dim(1), _Dim(4)])
        graph = _Graph(
            [_Node("proj", ["hidden"], ["hidden_2D"])],
            outputs=[_Value("hidden_2D", 0, None)],
            inputs=[known],
            value_info=[known],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            created = create_component_onnx(
                _Model(graph),
                ComponentInfo(component_type="other", node_indices=[0]),
                "proj",
            )
        assert created.graph.input[0] is known
        assert ("hidden", 1, [1, 4]) not in made
        assert ("hidden_2D", 10, [None, None]) in made
        assert any("hidden_2D" in str(item.message) and "heuristic" in str(item.message) for item in caught)
        assert all("hidden uses a name heuristic" not in str(item.message) for item in caught)

        mask_graph = _Graph(
            [_Node("mask", ["hidden"], ["attention_mask"])],
            outputs=[],
            inputs=[known],
            value_info=[known],
        )
        made.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_component_onnx(
                _Model(mask_graph),
                ComponentInfo(component_type="other", node_indices=[0]),
                "mask",
            )
        assert ("attention_mask", 7, [None]) in made
        assert any("output tensor attention_mask" in str(item.message) for item in caught)

        class _Part:
            def __init__(self, indices):
                self.node_indices = list(indices)

        bare = _Graph(
            [
                _Node("embed", ["input_ids"], ["hidden"]),
                _Node("head", ["hidden"], ["logits"]),
            ],
            outputs=[_Value("logits", 10, [_Dim(1), _Dim(4)])],
            inputs=[_Value("input_ids", 7, [_Dim(1), _Dim(param="seq")])],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = collect_boundary_tensors(bare, {
                "embedding": _Part([0]),
                "lm_head": _Part([1]),
            })
        by_name = {item.name: item for item in found}
        assert by_name["hidden"].dtype == "float16"
        assert by_name["hidden"].shape == [None]
        assert by_name["input_ids"].dtype == "int64"
        assert by_name["logits"].dtype == "float16"
        hidden_warnings = [item for item in caught if "boundary tensor hidden " in str(item.message)]
        assert len(hidden_warnings) == 1
        assert all("input_ids" not in str(item.message) for item in caught)
        assert all("logits" not in str(item.message) for item in caught)
    finally:
        splitter.helper = original_helper

    print("dtype resolution: OK")
    print("  value info, then initializer, then a name heuristic")
    print("  a heuristic dtype or shape warns")
    print("  a mask input and a mask output stay int64")
    return True


def test_quantization_metadata():
    """Quantization metadata is read from the graph and saved in model_info."""
    print("=" * 50)
    print("Testing quantization metadata...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.splitter import extract_quantization_info
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import extract_quantization_info
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.extract_quantization_info is extract_quantization_info

    class _Attr:
        def __init__(self, name, value, type_code=2):
            self.name = name
            self.type = type_code
            self.i = value

    class _Dim:
        def __init__(self, value):
            self.dim_value = value
            self.dim_param = ""

    class _Value:
        def __init__(self, name, elem, dims):
            shape = None if dims is None else types.SimpleNamespace(
                dim=[_Dim(dim) for dim in dims]
            )
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(elem_type=elem, shape=shape)
            )

    class _Init:
        def __init__(self, name, data_type, dims):
            self.name = name
            self.data_type = data_type
            self.dims = dims

    class _Node:
        def __init__(self, name, op_type, inputs, attributes=()):
            self.name = name
            self.op_type = op_type
            self.domain = "com.microsoft"
            self.input = list(inputs)
            self.attribute = list(attributes)

    class _Graph:
        def __init__(self, nodes, inputs=(), initializers=(), value_info=()):
            self.node = list(nodes)
            self.input = list(inputs)
            self.initializer = list(initializers)
            self.output = []
            self.value_info = list(value_info)

    def _nbits(name="q_proj", bits=4, block=32, inputs=None, op="MatMulNBits"):
        if inputs is None:
            inputs = ["hidden", "q_weight", "q_scale", "q_zp"]
        attributes = []
        if bits is not None:
            attributes.append(_Attr("bits", bits))
        if block is not None:
            attributes.append(_Attr("block_size", block))
        return _Node(name, op, inputs, attributes)

    def _quiet(graph):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = extract_quantization_info(graph)
        return info, caught

    values = [
        _Value("hidden", 10, [1, 8]),
        _Value("q_weight", 2, [8, 4, 16]),
        _Value("q_scale", 10, [8, 4]),
        _Value("q_zp", 2, [8, 4]),
    ]
    second = _nbits("k_proj", block=None)
    second.attribute.append(_Attr("group_size", 32))
    info, caught = _quiet(_Graph([_nbits(), second], value_info=values))
    assert caught == []
    assert info == {
        "method": "MatMulNBits",
        "weight_dtype": "int4",
        "activation_dtype": "float16",
        "group_size": 32,
        "has_zero_point": True,
        "packing_format": "packed_uint8",
        "bits": 4,
        "scale_dtype": "float16",
        "zero_point_dtype": "uint8",
    }
    info["group_size"] = 1
    again, _ = _quiet(_Graph([_nbits()], value_info=values))
    assert again["group_size"] == 32
    assert again is not info
    json.dumps(again)

    dense, caught = _quiet(_Graph([_Node("proj", "MatMul", ["hidden"])]))
    assert dense["method"] is None
    assert dense["weight_dtype"] is None
    assert dense["activation_dtype"] is None
    assert dense["group_size"] is None
    assert dense["has_zero_point"] is False
    assert dense["packing_format"] is None
    assert dense["bits"] is None
    assert dense["scale_dtype"] is None
    assert dense["zero_point_dtype"] is None
    assert caught == []
    other_dense, _ = _quiet(types.SimpleNamespace(node=[_Node("proj", "MatMul", ["hidden"])]))
    assert other_dense is not dense
    dense["method"] = "MatMulNBits"
    assert other_dense["method"] is None

    gptq, caught = _quiet(_Graph([
        _nbits("q_proj", inputs=["hidden", "q_weight", "q_scale", "", "g_idx"]),
    ], value_info=values))
    assert caught == []
    assert gptq["method"] == "GPTQ"
    assert gptq["has_zero_point"] is False
    assert gptq["zero_point_dtype"] is None
    assert gptq["packing_format"] == "packed_uint8"
    assert gptq["group_size"] == 32

    awq, caught = _quiet(_Graph([
        _nbits("gate_awq", op="AWQMatMul"),
    ], value_info=values))
    assert caught == []
    assert awq["method"] == "AWQ"
    assert awq["weight_dtype"] == "int4"
    assert awq["packing_format"] == "packed_uint8"
    named, _ = _quiet(_Graph([_nbits("layers.0.awq.q_proj")], value_info=values))
    assert named["method"] == "AWQ"

    qvalues = [
        _Value("hidden", 2, [1, 8]),
        _Value("b_weight", 3, [8, 8]),
        _Value("b_scale", 1, [1]),
        _Value("b_zp", 3, [1]),
    ]
    qlinear, caught = _quiet(_Graph([
        _Node(
            "down",
            "QLinearMatMul",
            ["hidden", "a_scale", "a_zp", "b_weight", "b_scale", "b_zp"],
            [_Attr("bits", 8)],
        ),
    ], value_info=qvalues))
    assert caught == []
    assert qlinear["method"] == "QLinearMatMul"
    assert qlinear["bits"] == 8
    assert qlinear["weight_dtype"] == "int8"
    assert qlinear["activation_dtype"] == "uint8"
    assert qlinear["group_size"] is None
    assert qlinear["has_zero_point"] is True
    assert qlinear["packing_format"] == "qdq"
    assert qlinear["scale_dtype"] == "float32"
    assert qlinear["zero_point_dtype"] == "int8"

    bare, caught = _quiet(_Graph([
        _Node("q_proj", "MatMulNBits", ["attention_mask", "", "", ""]),
    ]))
    assert bare["method"] == "MatMulNBits"
    assert bare["bits"] == 4
    assert bare["weight_dtype"] == "int4"
    assert bare["activation_dtype"] == "float16"
    assert bare["group_size"] == 128
    assert bare["packing_format"] == "packed_uint8"
    assert bare["has_zero_point"] is False
    messages = [str(item.message) for item in caught]
    assert all(issubclass(item.category, UserWarning) for item in caught)
    assert any("default bits 4" in message for message in messages)
    assert any("default group_size 128" in message for message in messages)
    assert any("default activation_dtype float16" in message for message in messages)
    assert any("default packing_format packed_uint8" in message for message in messages)
    assert all("heuristic" not in message for message in messages)
    assert bare["activation_dtype"] != "int64"

    shaped = _nbits(block=None)
    inferred, caught = _quiet(_Graph([shaped], value_info=values))
    assert inferred["group_size"] == 32
    assert all("group_size" not in str(item.message) for item in caught)

    from_k = _nbits(block=None)
    from_k.attribute.append(_Attr("K", 128))
    k_values = [
        _Value("hidden", 10, [1, 8]),
        _Value("q_weight", 2, [8, 16]),
        _Value("q_scale", 10, [8, 4]),
        _Value("q_zp", 2, [8, 4]),
    ]
    k_info, caught = _quiet(_Graph([from_k], value_info=k_values))
    assert k_info["group_size"] == 32
    assert caught == []

    init_values = [
        _Value("hidden", 1, [1, 4]),
        _Value("q_weight", 6, [8, 4, 16]),
        _Value("q_zp", 2, [8, 4]),
    ]
    stored, caught = _quiet(_Graph(
        [_nbits()],
        value_info=init_values,
        initializers=[
            _Init("hidden", 10, [1, 4]),
            _Init("q_scale", 10, [8, 4]),
        ],
    ))
    assert caught == []
    assert stored["activation_dtype"] == "float32"
    assert stored["scale_dtype"] == "float16"
    assert stored["packing_format"] == "packed_int32"
    assert stored["weight_dtype"] == "int4"

    try:
        extract_quantization_info(_Graph([
            _nbits(block=32),
            _nbits("other", block=128),
        ], value_info=values))
        raise AssertionError("disagreeing group sizes were accepted")
    except ValueError as exc:
        assert "group_size" in str(exc)
        assert "32" in str(exc) and "128" in str(exc)
    try:
        extract_quantization_info(None)
        raise AssertionError("a missing graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        bad_bits = _nbits()
        bad_bits.attribute = [_Attr("bits", True)]
        extract_quantization_info(_Graph([bad_bits], value_info=values))
        raise AssertionError("a bool bits attribute was accepted")
    except ValueError as exc:
        assert "bits" in str(exc)
    try:
        extract_quantization_info(_Graph([_nbits(bits=7)], value_info=values))
        raise AssertionError("a 7-bit weight was accepted")
    except ValueError as exc:
        assert "bits" in str(exc)
    try:
        extract_quantization_info(_Graph([_nbits(block=0)], value_info=values))
        raise AssertionError("a zero group size was accepted")
    except ValueError as exc:
        assert "group_size" in str(exc)
    try:
        extract_quantization_info(_Graph([_nbits(block=3)], value_info=values))
        raise AssertionError("a group that does not fill a byte was accepted")
    except ValueError as exc:
        assert "bytes" in str(exc)
    try:
        broken = _nbits()
        broken.attribute = "bits"
        extract_quantization_info(_Graph([broken]))
        raise AssertionError("a string attribute list was accepted")
    except ValueError as exc:
        assert "list" in str(exc)

    kept = {}

    def make_tensor_value_info(name, elem_type, shape):
        return _Value(name, elem_type, shape)

    def make_graph(nodes, name, inputs, outputs, initializer):
        kept["nodes"] = nodes
        kept["initializer"] = initializer
        return types.SimpleNamespace(
            node=list(nodes),
            name=name,
            input=list(inputs),
            output=list(outputs),
            initializer=list(initializer),
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(
            graph=graph,
            producer_name=producer_name,
            opset_import=opset_imports,
            ir_version=0,
        )

    original_helper = splitter.helper
    splitter.helper = types.SimpleNamespace(
        make_tensor_value_info=make_tensor_value_info,
        make_graph=make_graph,
        make_model=make_model,
    )
    try:
        survived = _nbits("/model/layers.0/self_attn/q_proj/MatMulNBits")
        survived.output = ["hidden_out"]
        weight = _Init("q_weight", 2, [8, 4, 16])
        scale = _Init("q_scale", 10, [8, 4])
        zero = _Init("q_zp", 2, [8, 4])
        component_graph = _Graph(
            [survived],
            value_info=list(values) + [_Value("hidden_out", 10, [1, 8])],
            initializers=[weight, scale, zero],
        )
        component_model = types.SimpleNamespace(
            graph=component_graph,
            opset_import=[],
            ir_version=8,
        )
        created = splitter.create_component_onnx(
            component_model,
            ComponentInfo(
                component_type="layer",
                layer_id=0,
                node_indices=[0],
                initializer_names={"q_weight", "q_scale", "q_zp"},
            ),
            "layer_0",
        )
        assert created.graph.node[0] is survived
        assert created.graph.node[0].attribute[0].name == "bits"
        assert created.graph.node[0].attribute[0].i == 4
        assert created.graph.node[0].attribute[1].name == "block_size"
        assert created.graph.node[0].attribute[1].i == 32
        assert {item.name for item in created.graph.initializer} == {
            "q_weight",
            "q_scale",
            "q_zp",
        }
    finally:
        splitter.helper = original_helper

    class _SplitModel:
        def __init__(self, graph):
            self.graph = graph
            self.ir_version = 8
            self.producer_name = "test"

    split_graph = _Graph(
        [_nbits("/model/layers.0/self_attn/q_proj/MatMulNBits")],
        value_info=values,
    )
    original_load = splitter.load_onnx_model
    original_extract = splitter.extract_components
    original_create = splitter.create_component_onnx
    original_boundaries = splitter.collect_boundary_tensors
    original_save = getattr(splitter.onnx, "save", None)
    original_infer = getattr(splitter.shape_inference, "infer_shapes", None)

    def fake_load(path):
        return _SplitModel(split_graph)

    def fake_infer(model):
        return model

    def fake_extract(model, **kwargs):
        return {
            "layer_0": ComponentInfo(
                component_type="layer",
                layer_id=0,
                node_indices=[0],
            )
        }

    def fake_create(model, component, name):
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[types.SimpleNamespace(name=f"{name}_in")],
                output=[types.SimpleNamespace(name=f"{name}_out")],
            )
        )

    def fake_boundaries(model, components):
        return []

    def fake_save(*args, **kwargs):
        return None

    splitter.load_onnx_model = fake_load
    splitter.extract_components = fake_extract
    splitter.create_component_onnx = fake_create
    splitter.collect_boundary_tensors = fake_boundaries
    splitter.onnx.save = fake_save
    splitter.shape_inference.infer_shapes = fake_infer
    try:
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.onnx")
            output_path = os.path.join(directory, "out")
            os.makedirs(output_path)
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "model_type": "qwen3",
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "sliding_window": 2048,
                    "tie_word_embeddings": True,
                }, handle)
            result = splitter.split_onnx(
                model=model_path,
                output=output_path,
                split_only=True,
            )
            info_path = os.path.join(result.output_dir, "model_info.json")
            saved = json.load(open(info_path, encoding="utf-8"))
            assert saved["architecture"] == "qwen3"
            descriptor = saved["architecture_descriptor"]
            assert descriptor["arch_type"] == "qwen3"
            assert descriptor["supports_gqa"] is True
            assert descriptor["supports_mqa"] is False
            assert descriptor["is_moe"] is False
            assert descriptor["has_sliding_window"] is True
            assert descriptor["has_tied_embeddings"] is True
            assert descriptor["quantization_format"] == "matmulnbits"
            assert saved["quantization_info"]["method"] == "MatMulNBits"
            assert saved["quantization_info"]["weight_dtype"] == "int4"
            assert saved["quantization_info"]["activation_dtype"] == "float16"
            assert saved["quantization_info"]["group_size"] == 32
            assert saved["quantization_info"]["has_zero_point"] is True
            assert saved["quantization_info"]["packing_format"] == "packed_uint8"
            assert saved["quantization_info"]["bits"] == 4
            assert saved["quantization_info"]["scale_dtype"] == "float16"
            assert saved["quantization_info"]["zero_point_dtype"] == "uint8"
            assert result.model_info["quantization_info"] == saved["quantization_info"]
            assert result.model_info["architecture_descriptor"] == descriptor
            bad_model = os.path.join(directory, "broken.onnx")
            with open(bad_model, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{")
            try:
                splitter.split_onnx(model=bad_model, output=output_path, split_only=True)
                raise AssertionError("broken config json was accepted")
            except ValueError as exc:
                assert "config.json" in str(exc)
            with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("[]")
            try:
                splitter.split_onnx(model=bad_model, output=output_path, split_only=True)
                raise AssertionError("a config list was accepted")
            except ValueError as exc:
                assert "JSON object" in str(exc)
    finally:
        splitter.load_onnx_model = original_load
        splitter.extract_components = original_extract
        splitter.create_component_onnx = original_create
        splitter.collect_boundary_tensors = original_boundaries
        if original_save is None:
            del splitter.onnx.save
        else:
            splitter.onnx.save = original_save
        if original_infer is None:
            if hasattr(splitter.shape_inference, "infer_shapes"):
                del splitter.shape_inference.infer_shapes
        else:
            splitter.shape_inference.infer_shapes = original_infer

    print("quantization metadata: OK")
    print("  matmul nbits: int4 group 32 packed uint8")
    print("  formats: AWQ, GPTQ, scale and zero point")
    print("  model_info: quantization_info and architecture capabilities saved")
    print("  matmul nbits params stay on the component graph")
    return True


def test_kv_cache_metadata():
    """KV cache metadata is read from past and present tensors and saved."""
    print("=" * 50)
    print("Testing KV cache metadata...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.splitter import extract_kv_cache_info
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import extract_kv_cache_info
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.extract_kv_cache_info is extract_kv_cache_info

    class _Dim:
        def __init__(self, value=0, name=""):
            self.dim_value = value
            self.dim_param = name

    class _Value:
        def __init__(self, name, elem, dims):
            shape = None if dims is None else types.SimpleNamespace(
                dim=[dim if isinstance(dim, _Dim) else _Dim(dim) for dim in dims]
            )
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(elem_type=elem, shape=shape)
            )

    class _Init:
        def __init__(self, name, data_type, dims):
            self.name = name
            self.data_type = data_type
            self.dims = dims

    class _Node:
        def __init__(self, name, op_type, inputs, outputs=()):
            self.name = name
            self.op_type = op_type
            self.input = list(inputs)
            self.output = list(outputs)

    class _Graph:
        def __init__(self, nodes, inputs=(), outputs=(), initializers=(), value_info=()):
            self.node = list(nodes)
            self.input = list(inputs)
            self.initializer = list(initializers)
            self.output = list(outputs)
            self.value_info = list(value_info)

    def _quiet(graph):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = extract_kv_cache_info(graph)
        return info, caught

    def _named(name, elem=10, dims=None):
        if dims is None:
            dims = [
                _Dim(1, "batch"),
                _Dim(8, "num_kv_heads"),
                _Dim(128, "past_sequence_length"),
                _Dim(64, "head_dim"),
            ]
        return _Value(name, elem, dims)

    past_names = []
    present_names = []
    values = []
    for layer in (0, 1):
        for kind in ("key", "value"):
            past = f"past_key_values.{layer}.{kind}"
            present = f"present.{layer}.{kind}"
            past_names.append(past)
            present_names.append(present)
            values.append(_named(past))
            values.append(_named(present))
    info, caught = _quiet(_Graph(
        [_Node("/model/layers.0/self_attn/MatMul", "MatMul", ["hidden"], ["hidden_out"])],
        inputs=[_named(name) for name in past_names],
        outputs=[_named(name) for name in present_names],
        value_info=values,
    ))
    assert caught == []
    assert info == {
        "num_layers": 2,
        "num_kv_heads": 8,
        "head_dim": 64,
        "layout": "BNSD",
        "dtype": "float16",
        "past_key_pattern": "past_key_values.{layer}.key",
        "past_value_pattern": "past_key_values.{layer}.value",
        "present_key_pattern": "present.{layer}.key",
        "present_value_pattern": "present.{layer}.value",
        "max_seq_len": 128,
        "stateful": True,
        "stateful_contract": {
            "roles": ["past_key", "past_value", "present_key", "present_value"],
            "zero_copy": True,
        },
    }
    info["num_kv_heads"] = 1
    info["stateful_contract"]["roles"].append("other")
    again, _ = _quiet(_Graph(
        [_Node("attn", "MatMul", past_names, present_names)],
        inputs=[_named(name) for name in past_names],
        outputs=[_named(name) for name in present_names],
    ))
    assert again["num_kv_heads"] == 8
    assert again["stateful_contract"]["roles"] == [
        "past_key", "past_value", "present_key", "present_value",
    ]
    assert again is not info
    json.dumps(again)

    dense, caught = _quiet(_Graph([_Node("proj", "MatMul", ["hidden"], ["out"])]))
    assert dense["num_layers"] == 0
    assert dense["num_kv_heads"] is None
    assert dense["head_dim"] is None
    assert dense["layout"] is None
    assert dense["dtype"] is None
    assert dense["past_key_pattern"] is None
    assert dense["past_value_pattern"] is None
    assert dense["present_key_pattern"] is None
    assert dense["present_value_pattern"] is None
    assert dense["max_seq_len"] is None
    assert dense["stateful"] is False
    assert dense["stateful_contract"] is None
    assert caught == []
    other_dense, _ = _quiet(types.SimpleNamespace(
        node=[_Node("proj", "MatMul", ["hidden"], ["out"])],
        input=[],
        output=[],
    ))
    assert other_dense is not dense
    dense["dtype"] = "float16"
    assert other_dense["dtype"] is None

    bare_name = "past_key_values.0.key"
    bare_value = _Value(bare_name, 10, [1, 8, 32, 64])
    assumed, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", [bare_name], ["present.0.key"])],
        inputs=[bare_value],
        outputs=[_Value("present.0.key", 10, [1, 8, 32, 64])],
    ))
    assert assumed["layout"] == "BNSD"
    assert assumed["num_kv_heads"] == 8
    assert assumed["head_dim"] == 64
    assert assumed["max_seq_len"] == 32
    assert assumed["dtype"] == "float16"
    assert assumed["num_layers"] == 1
    messages = [str(item.message) for item in caught]
    assert any("default layout BNSD" in message for message in messages)
    assert all(issubclass(item.category, UserWarning) for item in caught)

    bsnd_dims = [
        _Dim(1, "batch"),
        _Dim(128, "sequence"),
        _Dim(8, "num_kv_heads"),
        _Dim(64, "head_dim"),
    ]
    bsnd, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", ["past_key_values.0.key"])],
        inputs=[_Value("past_key_values.0.key", 10, bsnd_dims)],
        outputs=[_Value("present.0.key", 10, bsnd_dims)],
    ))
    assert caught == []
    assert bsnd["layout"] == "BSND"
    assert bsnd["num_kv_heads"] == 8
    assert bsnd["head_dim"] == 64
    assert bsnd["max_seq_len"] == 128
    assert bsnd["past_value_pattern"] is None
    assert bsnd["present_value_pattern"] is None
    assert bsnd["stateful_contract"]["roles"] == ["past_key", "present_key"]

    symbolic_dims = [
        _Dim(1, "batch"),
        _Dim(8, "num_kv_heads"),
        _Dim(0, "past_sequence_length"),
        _Dim(64, "head_dim"),
    ]
    symbolic, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", ["past_key_values.0.key"])],
        inputs=[_Value("past_key_values.0.key", 10, symbolic_dims)],
    ))
    assert symbolic["layout"] == "BNSD"
    assert symbolic["num_kv_heads"] == 8
    assert symbolic["head_dim"] == 64
    assert symbolic["max_seq_len"] is None
    assert symbolic["dtype"] == "float16"
    assert any("does not record max_seq_len" in str(item.message) for item in caught)
    assert all("default layout" not in str(item.message) for item in caught)

    masked = "past_key_values.0.attention_mask.key"
    masked_info, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", [masked])],
        inputs=[_Value(masked, 0, [1, 4, 16, 32])],
    ))
    assert masked_info["dtype"] == "float16"
    assert masked_info["dtype"] != "int64"
    assert masked_info["past_key_pattern"] == "past_key_values.{layer}.attention_mask.key"
    masked_messages = [str(item.message) for item in caught]
    assert any("default dtype float16" in message for message in masked_messages)
    assert any("default layout BNSD" in message for message in masked_messages)
    assert all("heuristic" not in message for message in masked_messages)

    typed = _Value("past_key_values.0.key", 1, None)
    stored, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", ["past_key_values.0.key"])],
        inputs=[typed],
        initializers=[_Init("past_key_values.0.key", 10, [1, 8, 32, 64])],
    ))
    assert stored["dtype"] == "float32"
    assert stored["num_kv_heads"] == 8
    assert stored["head_dim"] == 64
    assert stored["max_seq_len"] == 32
    assert any("default layout BNSD" in str(item.message) for item in caught)
    assert all("dtype" not in str(item.message) for item in caught)

    rank3, caught = _quiet(_Graph(
        [_Node("attn", "MatMul", ["past_key_values.0.key"])],
        inputs=[_Value("past_key_values.0.key", 10, [1, 8, 64])],
    ))
    assert rank3["layout"] is None
    assert rank3["num_kv_heads"] is None
    assert rank3["head_dim"] is None
    assert rank3["max_seq_len"] is None
    assert rank3["num_layers"] == 1
    assert rank3["stateful"] is True
    assert any("rank 3" in str(item.message) for item in caught)

    plain = _Graph(
        [_Node("attn", "MatMul", ["past_key"])],
        inputs=[_Value("past_key", 10, [
            _Dim(1, "batch"),
            _Dim(2, "num_kv_heads"),
            _Dim(4, "seq"),
            _Dim(8, "head_dim"),
        ])],
    )
    plain_info, caught = _quiet(plain)
    assert plain_info["num_layers"] == 1
    assert plain_info["past_key_pattern"] == "past_key"
    assert plain_info["num_kv_heads"] == 2
    assert any("default layer index 0" in str(item.message) for item in caught)

    try:
        extract_kv_cache_info(_Graph(
            [_Node("attn", "MatMul", ["past_key_values.0.key", "past_key_values.1.key"])],
            inputs=[
                _Value("past_key_values.0.key", 10, [1, 8, 4, 64]),
                _Value("past_key_values.1.key", 10, [1, 4, 4, 64]),
            ],
        ))
        raise AssertionError("disagreeing kv heads were accepted")
    except ValueError as exc:
        assert "num_kv_heads" in str(exc)
        assert "8" in str(exc) and "4" in str(exc)
    try:
        extract_kv_cache_info(_Graph(
            [_Node("attn", "MatMul", ["past_key_values.0.key", "past_key_1"])],
            inputs=[
                _named("past_key_values.0.key"),
                _named("past_key_1"),
            ],
        ))
        raise AssertionError("an ambiguous layer index was accepted")
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    try:
        extract_kv_cache_info(_Graph(
            [_Node("attn", "MatMul", ["past_key_values.0.key", "past_key_values.2.key"])],
            inputs=[_named("past_key_values.0.key"), _named("past_key_values.2.key")],
        ))
        raise AssertionError("a gap in kv layers was accepted")
    except ValueError as exc:
        assert "not contiguous" in str(exc)
    try:
        extract_kv_cache_info(_Graph(
            [_Node("attn", "MatMul", ["past_key_values.0.key", "past_key_values.1.key"], ["present.0.key"])],
            inputs=[_named("past_key_values.0.key"), _named("past_key_values.1.key")],
            outputs=[_named("present.0.key")],
        ))
        raise AssertionError("a partial present layer set was accepted")
    except ValueError as exc:
        assert "layers disagree" in str(exc)
    try:
        extract_kv_cache_info(None)
        raise AssertionError("a missing graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        extract_kv_cache_info(_Graph(
            [_Node("attn", "MatMul", ["past_key_values.0.key"])],
            inputs=[_Value("past_key_values.0.key", 10, [_Dim(True)])],
        ))
        raise AssertionError("a bool cache dimension was accepted")
    except ValueError as exc:
        assert "dimension" in str(exc)

    class _SplitModel:
        def __init__(self, graph):
            self.graph = graph
            self.ir_version = 8
            self.producer_name = "test"

    split_values = [
        _named("past_key_values.0.key"),
        _named("past_key_values.0.value"),
        _named("present.0.key"),
        _named("present.0.value"),
    ]
    split_graph = _Graph(
        [_Node("/model/layers.0/self_attn/MatMul", "MatMul", ["past_key_values.0.key"], ["present.0.key"])],
        inputs=split_values[:2],
        outputs=split_values[2:],
    )
    original_load = splitter.load_onnx_model
    original_extract = splitter.extract_components
    original_create = splitter.create_component_onnx
    original_boundaries = splitter.collect_boundary_tensors
    original_save = getattr(splitter.onnx, "save", None)
    original_infer = getattr(splitter.shape_inference, "infer_shapes", None)

    def fake_load(path):
        return _SplitModel(split_graph)

    def fake_infer(model):
        return model

    def fake_extract(model, **kwargs):
        return {
            "layer_0": ComponentInfo(
                component_type="layer",
                layer_id=0,
                node_indices=[0],
            )
        }

    def fake_create(model, component, name):
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[types.SimpleNamespace(name="past_key_values.0.key")],
                output=[types.SimpleNamespace(name="present.0.key")],
            )
        )

    def fake_boundaries(model, components):
        return []

    def fake_save(*args, **kwargs):
        return None

    splitter.load_onnx_model = fake_load
    splitter.extract_components = fake_extract
    splitter.create_component_onnx = fake_create
    splitter.collect_boundary_tensors = fake_boundaries
    splitter.onnx.save = fake_save
    splitter.shape_inference.infer_shapes = fake_infer
    try:
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.onnx")
            output_path = os.path.join(directory, "out")
            os.makedirs(output_path)
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            result = splitter.split_onnx(
                model=model_path,
                output=output_path,
                split_only=True,
            )
            info_path = os.path.join(result.output_dir, "model_info.json")
            saved = json.load(open(info_path, encoding="utf-8"))
            assert saved["kv_cache_info"]["num_layers"] == 1
            assert saved["kv_cache_info"]["num_kv_heads"] == 8
            assert saved["kv_cache_info"]["head_dim"] == 64
            assert saved["kv_cache_info"]["layout"] == "BNSD"
            assert saved["kv_cache_info"]["dtype"] == "float16"
            assert saved["kv_cache_info"]["past_key_pattern"] == "past_key_values.{layer}.key"
            assert saved["kv_cache_info"]["past_value_pattern"] == "past_key_values.{layer}.value"
            assert saved["kv_cache_info"]["present_key_pattern"] == "present.{layer}.key"
            assert saved["kv_cache_info"]["present_value_pattern"] == "present.{layer}.value"
            assert saved["kv_cache_info"]["max_seq_len"] == 128
            assert saved["kv_cache_info"]["stateful"] is True
            assert result.model_info["kv_cache_info"] == saved["kv_cache_info"]
            assert saved["architecture"] == "llama"
            assert saved["architecture_descriptor"]["arch_type"] == "llama"
            assert saved["architecture_descriptor"]["supports_gqa"] is True
            assert saved["architecture_descriptor"]["is_moe"] is False
            assert saved["architecture_descriptor"]["has_sliding_window"] is False
            assert saved["architecture_descriptor"]["has_tied_embeddings"] is False
            assert saved["kv_cache_info"]["layout"] == "BNSD"
            assert saved["kv_cache_info"]["dtype"] == "float16"
            assert saved["kv_cache_info"]["max_seq_len"] == 128
    finally:
        splitter.load_onnx_model = original_load
        splitter.extract_components = original_extract
        splitter.create_component_onnx = original_create
        splitter.collect_boundary_tensors = original_boundaries
        if original_save is None:
            del splitter.onnx.save
        else:
            splitter.onnx.save = original_save
        if original_infer is None:
            if hasattr(splitter.shape_inference, "infer_shapes"):
                del splitter.shape_inference.infer_shapes
        else:
            splitter.shape_inference.infer_shapes = original_infer

    print("kv cache metadata: OK")
    print("  past_key_values and present: 8 heads, head dim 64, BNSD")
    print("  max sequence length, dtype, and stateful contract")
    print("  model_info: kv_cache_info saved")
    return True


def test_execution_policy():
    """AMD execution policy is read from the split and saved."""
    print("=" * 50)
    print("Testing execution policy...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.splitter import extract_execution_policy
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.splitter import extract_execution_policy
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.extract_execution_policy is extract_execution_policy

    class _Init:
        def __init__(self, name, data_type):
            self.name = name
            self.data_type = data_type

    class _Node:
        def __init__(self, name, op_type="MatMul"):
            self.name = name
            self.op_type = op_type
            self.input = ["hidden"]
            self.output = ["hidden_out"]

    class _Graph:
        def __init__(self, nodes=None, initializers=()):
            self.node = list(nodes or [_Node("/model/layers.0/MatMul")])
            self.initializer = list(initializers)
            self.input = []
            self.output = []
            self.value_info = []

    def _parts(*kinds):
        parts = {}
        for index, kind in enumerate(kinds):
            parts[f"{kind}_{index}"] = ComponentInfo(
                component_type=kind,
                layer_id=0 if kind == "layer" else None,
                node_indices=[index],
            )
        return parts

    def _read(graph=None, components=None, quantization_info=None):
        if graph is None:
            graph = _Graph()
        if components is None:
            components = _parts("embedding", "layer", "norm", "lm_head")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = extract_execution_policy(
                graph,
                components,
                quantization_info=quantization_info,
            )
        return info, caught

    example = {
        "npu_safe": False,
        "recommended_mode": "hybrid_phase_split",
        "prefill_device": "npu",
        "decode_device": "igpu",
        "resident_components": ["embedding", "norm", "lm_head"],
        "streamed_components": ["layer_*"],
        "graph_form": "split",
        "layer_schedule": "persistent",
        "moe_routing": None,
    }
    info, caught = _read()
    assert info == example
    messages = [str(item.message) for item in caught]
    assert any("default npu_safe False" in message for message in messages)
    assert all("heuristic" not in message for message in messages)
    assert all(issubclass(item.category, UserWarning) for item in caught)
    info["resident_components"].append("other")
    info["streamed_components"].append("other_*")
    info["npu_safe"] = True
    again, _ = _read(components=_parts("lm_head", "embedding", "layer", "norm"))
    assert again == example
    assert again is not info
    assert again["resident_components"] is not info["resident_components"]
    assert again["streamed_components"] is not info["streamed_components"]
    json.dumps(again)

    masked, caught = _read(_Graph(initializers=[_Init("attention_mask", 0)]))
    assert masked == example
    masked_messages = [str(item.message) for item in caught]
    assert any("default npu_safe False" in message for message in masked_messages)
    assert all("heuristic" not in message for message in masked_messages)
    assert all("int64" not in message for message in masked_messages)

    for method in ("MatMulNBits", "AWQ", "GPTQ", "QLinearMatMul"):
        legal, caught = _read(quantization_info={"method": method})
        assert legal["npu_safe"] is True
        assert legal["recommended_mode"] == "hybrid_phase_split"
        assert legal["prefill_device"] == "npu"
        assert legal["decode_device"] == "igpu"
        assert legal["graph_form"] == "split"
        assert legal["layer_schedule"] == "persistent"
        assert all("default npu_safe" not in str(item.message) for item in caught)

    float16, caught = _read(_Graph(initializers=[_Init("weight", 10)]))
    assert float16["npu_safe"] is True
    assert float16["recommended_mode"] == "hybrid_phase_split"
    assert caught == []
    int4, caught = _read(_Graph(initializers=[_Init("weight", 21)]))
    assert int4["npu_safe"] is True
    assert caught == []

    float32, caught = _read(_Graph(initializers=[_Init("weight", 1)]))
    assert float32["npu_safe"] is False
    assert float32["recommended_mode"] == "sequential"
    assert float32["prefill_device"] == "igpu"
    assert float32["decode_device"] == "igpu"
    assert float32["layer_schedule"] == "sequential"
    assert float32["graph_form"] == "split"
    assert float32["resident_components"] == ["embedding", "norm", "lm_head"]
    assert float32["streamed_components"] == ["layer_*"]
    assert all("default npu_safe" not in str(item.message) for item in caught)
    float64, _ = _read(_Graph(initializers=[_Init("weight", 11)]))
    assert float64["recommended_mode"] == "sequential"
    assert float64["prefill_device"] == "igpu"

    mixed_safe, caught = _read(_Graph(initializers=[
        _Init("weight", 1),
        _Init("packed", 10),
    ]))
    assert mixed_safe["npu_safe"] is True
    assert mixed_safe["recommended_mode"] == "hybrid_phase_split"
    assert any("majority" in str(item.message) and "True" in str(item.message) for item in caught)
    mixed_unsafe, caught = _read(_Graph(initializers=[
        _Init("a", 1),
        _Init("b", 1),
        _Init("packed", 10),
    ]))
    assert mixed_unsafe["npu_safe"] is False
    assert mixed_unsafe["recommended_mode"] == "hybrid_phase_split"
    assert any("majority" in str(item.message) and "False" in str(item.message) for item in caught)

    alone, caught = _read(components=_parts("layer"))
    assert alone == {
        "npu_safe": False,
        "recommended_mode": "monolithic",
        "prefill_device": "igpu",
        "decode_device": "igpu",
        "resident_components": ["layer"],
        "streamed_components": [],
        "graph_form": "monolithic",
        "layer_schedule": "persistent",
        "moe_routing": None,
    }
    assert any("default npu_safe False" in str(item.message) for item in caught)
    alone_npu, caught = _read(
        _Graph(initializers=[_Init("weight", 10)]),
        components=_parts("lm_head"),
    )
    assert alone_npu["npu_safe"] is True
    assert alone_npu["recommended_mode"] == "monolithic"
    assert alone_npu["prefill_device"] == "npu"
    assert alone_npu["decode_device"] == "npu"
    assert alone_npu["resident_components"] == ["lm_head"]
    assert alone_npu["streamed_components"] == []
    assert caught == []

    two_embed, _ = _read(components=_parts("embedding", "embedding"))
    assert two_embed["graph_form"] == "split"
    assert two_embed["resident_components"] == ["embedding"]
    assert two_embed["streamed_components"] == []
    assert two_embed["recommended_mode"] == "hybrid_phase_split"

    moe, _ = _read(components=_parts(
        "moe_expert",
        "lm_head",
        "gated_deltanet",
        "embedding",
        "sparse_attention",
        "moe_router",
        "other",
        "mtp_head",
        "layer",
        "ngram_embedding",
        "norm",
    ))
    assert moe["resident_components"] == [
        "embedding", "norm", "lm_head", "moe_router", "mtp_head", "ngram_embedding",
    ]
    assert moe["streamed_components"] == [
        "layer_*",
        "moe_expert_*",
        "gated_deltanet_*",
        "sparse_attention_*",
        "other",
    ]
    assert moe["moe_routing"] == "moe_routing"
    assert moe["recommended_mode"] == "hybrid_phase_split"
    expert_only, _ = _read(components=_parts("moe_expert", "layer"))
    assert expert_only["moe_routing"] == "moe_routing"
    assert expert_only["streamed_components"] == ["layer_*", "moe_expert_*"]
    assert expert_only["resident_components"] == []

    hyphen, _ = _read(components={
        "head": {"component_type": "lm-head"},
        "block": {"component_type": "Gated DeltaNet"},
    })
    assert hyphen["resident_components"] == ["lm_head"]
    assert hyphen["streamed_components"] == ["gated_deltanet_*"]

    try:
        extract_execution_policy(None, _parts("layer"))
        raise AssertionError("a missing graph was accepted")
    except ValueError as exc:
        assert "graph" in str(exc)
    try:
        extract_execution_policy(_Graph(), ["layer"])
        raise AssertionError("a component list was accepted")
    except ValueError as exc:
        assert "dict" in str(exc)
    try:
        extract_execution_policy(_Graph(), {})
        raise AssertionError("an empty component map was accepted")
    except ValueError as exc:
        assert "at least one" in str(exc)
    try:
        extract_execution_policy(_Graph(), {" ": ComponentInfo(component_type="layer")})
        raise AssertionError("a blank component name was accepted")
    except ValueError as exc:
        assert "name" in str(exc)
    try:
        extract_execution_policy(_Graph(), {"layer_0": ComponentInfo(component_type="  ")})
        raise AssertionError("a blank component type was accepted")
    except ValueError as exc:
        assert "type" in str(exc)
    try:
        extract_execution_policy(_Graph(), {"layer_0": {"component_type": "banana"}})
        raise AssertionError("an unknown component type was accepted")
    except ValueError as exc:
        assert "Unknown component type" in str(exc)
        assert "banana" in str(exc)
    try:
        extract_execution_policy(_Graph(initializers=[_Init("weight", True)]), _parts("layer"))
        raise AssertionError("a bool data_type was accepted")
    except ValueError as exc:
        assert "data_type" in str(exc)
    try:
        extract_execution_policy(_Graph(), _parts("layer"), quantization_info=["MatMulNBits"])
        raise AssertionError("a quantization list was accepted")
    except ValueError as exc:
        assert "dict" in str(exc)
    try:
        extract_execution_policy(_Graph(), _parts("layer"), quantization_info={"method": True})
        raise AssertionError("a bool quantization method was accepted")
    except ValueError as exc:
        assert "method" in str(exc)

    class _SplitModel:
        def __init__(self, graph):
            self.graph = graph
            self.ir_version = 8
            self.producer_name = "test"

    split_graph = _Graph([_Node("/model/layers.0/self_attn/MatMul")])
    original_load = splitter.load_onnx_model
    original_extract = splitter.extract_components
    original_create = splitter.create_component_onnx
    original_boundaries = splitter.collect_boundary_tensors
    original_save = getattr(splitter.onnx, "save", None)
    original_infer = getattr(splitter.shape_inference, "infer_shapes", None)

    def fake_load(path):
        return _SplitModel(split_graph)

    def fake_infer(model):
        return model

    def fake_extract(model, **kwargs):
        return _parts("embedding", "layer", "norm", "lm_head")

    def fake_create(model, component, name):
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[types.SimpleNamespace(name=f"{name}_in")],
                output=[types.SimpleNamespace(name=f"{name}_out")],
            )
        )

    def fake_boundaries(model, components):
        return []

    def fake_save(*args, **kwargs):
        return None

    splitter.load_onnx_model = fake_load
    splitter.extract_components = fake_extract
    splitter.create_component_onnx = fake_create
    splitter.collect_boundary_tensors = fake_boundaries
    splitter.onnx.save = fake_save
    splitter.shape_inference.infer_shapes = fake_infer
    try:
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.onnx")
            output_path = os.path.join(directory, "out")
            os.makedirs(output_path)
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = splitter.split_onnx(
                    model=model_path,
                    output=output_path,
                    split_only=True,
                )
            info_path = os.path.join(result.output_dir, "model_info.json")
            saved = json.load(open(info_path, encoding="utf-8"))
            assert saved["execution_policy"] == example
            assert result.model_info["execution_policy"] == saved["execution_policy"]
            assert any("default npu_safe False" in str(item.message) for item in caught)

            def fake_extract_bad(model, **kwargs):
                return {"bad": ComponentInfo(component_type="not_a_type", node_indices=[0])}

            splitter.extract_components = fake_extract_bad
            bad_output = os.path.join(directory, "bad")
            os.makedirs(bad_output)
            try:
                splitter.split_onnx(model=model_path, output=bad_output, split_only=True)
                raise AssertionError("an unknown component was written")
            except ValueError as exc:
                assert "Unknown component type" in str(exc)
            assert not os.path.exists(os.path.join(bad_output, "model.onnx"))
    finally:
        splitter.load_onnx_model = original_load
        splitter.extract_components = original_extract
        splitter.create_component_onnx = original_create
        splitter.collect_boundary_tensors = original_boundaries
        if original_save is None:
            del splitter.onnx.save
        else:
            splitter.onnx.save = original_save
        if original_infer is None:
            if hasattr(splitter.shape_inference, "infer_shapes"):
                del splitter.shape_inference.infer_shapes
        else:
            splitter.shape_inference.infer_shapes = original_infer

    print("execution policy: OK")
    print("  hybrid phase split: npu prefill, igpu decode")
    print("  resident embedding, norm, lm_head; streamed layer")
    print("  model_info: execution_policy saved")
    return True


def test_split_validation():
    """PSNR, boundary contracts, and component IO are validated before files are written."""
    print("=" * 50)
    print("Testing split validation...")
    print("=" * 50)

    import json
    import math
    import os
    import sys
    import tempfile
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.validator import (
            compute_psnr,
            validate_boundary_contracts,
            validate_component_io,
            verify_split_correctness,
        )
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.validator import (
            compute_psnr,
            validate_boundary_contracts,
            validate_component_io,
            verify_split_correctness,
        )
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.compute_psnr is compute_psnr
    assert chunking_utils.verify_split_correctness is verify_split_correctness
    assert chunking_utils.validate_boundary_contracts is validate_boundary_contracts
    assert chunking_utils.validate_component_io is validate_component_io

    reference = [0.0, 1.0]
    assert math.isinf(compute_psnr(reference, [0.0, 1.0]))
    assert math.isinf(compute_psnr([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]))
    changed = [0.0, 1.1]
    measured = compute_psnr(reference, changed, data_range=1.0)
    mse = ((0.1 ** 2) / 2.0)
    expected = 20.0 * math.log10(1.0 / math.sqrt(mse))
    assert math.isclose(measured, expected)
    assert compute_psnr([0.0, 0.0], [0.0, 1.0]) == 0.0
    ranged = compute_psnr([0.0, 0.0], [0.0, 1.0], data_range=1.0)
    assert ranged > 0.0

    def _check(ref, cand, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = verify_split_correctness(ref, cand, **kwargs)
        return info, caught

    exact, caught = _check(reference, list(reference))
    assert caught == []
    assert exact["exact"] is True
    assert exact["psnr_db"] is None
    assert exact["mse"] == 0.0
    assert exact["quality"] == "production"
    assert exact["passed"] is True
    assert exact["production_ready"] is True
    assert exact["min_psnr_db"] == 35.0
    assert exact["production_psnr_db"] == 40.0
    exact["outputs"]["output"]["quality"] = "failed"
    again, _ = _check(reference, list(reference))
    assert again["outputs"]["output"]["quality"] == "production"
    assert again is not exact

    equal, caught = _check(reference, changed, min_psnr=measured, production_psnr=measured + 1.0)
    assert equal["passed"] is False
    assert equal["quality"] == "failed"
    assert any("PSNR verification failed" in str(item.message) for item in caught)
    good, caught = _check(
        reference,
        changed,
        min_psnr=measured - 0.1,
        production_psnr=measured + 1.0,
    )
    assert good["passed"] is True
    assert good["production_ready"] is False
    assert good["quality"] == "good"
    assert caught == []
    production, _ = _check(
        reference,
        changed,
        min_psnr=measured - 1.0,
        production_psnr=measured - 0.1,
    )
    assert production["production_ready"] is True
    assert production["quality"] == "production"
    json.dumps(production)

    worst, caught = _check(
        {"logits": reference, "hidden": reference},
        {"hidden": reference, "logits": changed},
        min_psnr=measured - 0.1,
        production_psnr=measured + 1.0,
    )
    assert worst["passed"] is True
    assert worst["production_ready"] is False
    assert worst["outputs"]["hidden"]["exact"] is True
    assert worst["outputs"]["logits"]["quality"] == "good"
    assert caught == []

    import numpy as np
    many, _ = _check(
        [np.array([1.0, 2.0]), np.array([3.0])],
        [np.array([1.0, 2.0]), np.array([3.0])],
    )
    assert set(many["outputs"]) == {"output_0", "output_1"}
    assert many["exact"] is True

    try:
        verify_split_correctness(reference, [0.0])
        raise AssertionError("a shape mismatch was accepted")
    except ValueError as exc:
        assert "shape" in str(exc)
    try:
        compute_psnr([True, False], [True, False])
        raise AssertionError("a bool array was accepted")
    except ValueError as exc:
        assert "numeric" in str(exc)
    try:
        compute_psnr([0.0, float("nan")], [0.0, 1.0])
        raise AssertionError("a NaN was accepted")
    except ValueError as exc:
        assert "finite" in str(exc)
    try:
        compute_psnr([], [])
        raise AssertionError("an empty array was accepted")
    except ValueError as exc:
        assert "at least one" in str(exc)
    try:
        compute_psnr(reference, changed, data_range=0)
        raise AssertionError("a zero data range was accepted")
    except ValueError as exc:
        assert "data_range" in str(exc)
    try:
        compute_psnr(reference, changed, data_range=True)
        raise AssertionError("a bool data range was accepted")
    except ValueError as exc:
        assert "data_range" in str(exc)
    try:
        verify_split_correctness(reference, changed, min_psnr=True)
        raise AssertionError("a bool threshold was accepted")
    except ValueError as exc:
        assert "min_psnr" in str(exc)
    try:
        verify_split_correctness(reference, changed, min_psnr=40.0, production_psnr=35.0)
        raise AssertionError("a production threshold below the minimum was accepted")
    except ValueError as exc:
        assert "production_psnr" in str(exc)
    try:
        verify_split_correctness({"logits": reference}, {"hidden": reference})
        raise AssertionError("disagreeing output names were accepted")
    except ValueError as exc:
        assert "output names" in str(exc)
    try:
        verify_split_correctness({}, {})
        raise AssertionError("an empty output map was accepted")
    except ValueError as exc:
        assert "at least one" in str(exc)

    components = {
        "embedding": {"input_names": ["input_ids"], "output_names": ["hidden"]},
        "layer_0": {
            "input_names": ["hidden", "past_key_values.0.key"],
            "output_names": ["logits"],
        },
    }
    hidden = {
        "name": "hidden",
        "dtype": "float16",
        "shape": [1, None, 8],
        "producer": "embedding",
        "consumers": ["layer_0"],
        "is_stateful": False,
        "zero_copy": True,
        "device_preference": None,
        "cache_role": None,
    }
    past = {
        "name": "past_key_values.0.key",
        "dtype": "float16",
        "shape": [1, 8, 128, 64],
        "producer": "external",
        "consumers": ["layer_0"],
        "is_stateful": True,
        "zero_copy": True,
        "device_preference": "npu",
        "cache_role": "past_key",
    }
    contracts = validate_boundary_contracts([hidden, past], components)
    assert contracts[0]["name"] == "hidden"
    assert contracts[0]["shape"] == [1, None, 8]
    assert contracts[1]["cache_role"] == "past_key"
    assert contracts[1]["device_preference"] == "npu"
    assert contracts[1]["zero_copy"] is True
    contracts[0]["shape"].append(1)
    again_contracts = validate_boundary_contracts([hidden, past], components)
    assert again_contracts[0]["shape"] == [1, None, 8]
    assert again_contracts is not contracts
    io = validate_component_io(components, [hidden, past])
    assert io["crossings"] == [
        {"name": "hidden", "producer": "embedding", "consumers": ["layer_0"]},
    ]
    io["components"]["embedding"]["output_names"].append("extra")
    fresh_io = validate_component_io(components, [hidden, past])
    assert fresh_io["components"]["embedding"]["output_names"] == ["hidden"]
    json.dumps(fresh_io)
    mask = dict(hidden)
    mask["name"] = "attention_mask"
    mask_components = {
        "embedding": {"input_names": ["input_ids"], "output_names": ["attention_mask"]},
        "layer_0": {"input_names": ["attention_mask"], "output_names": ["logits"]},
    }
    mask_contracts = validate_boundary_contracts([mask], mask_components)
    assert mask_contracts[0]["dtype"] == "float16"
    assert mask_contracts[0]["dtype"] != "int64"

    try:
        validate_boundary_contracts([hidden, hidden], components)
        raise AssertionError("a duplicate boundary was accepted")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    missing_producer = dict(hidden)
    missing_producer["producer"] = "norm"
    try:
        validate_boundary_contracts([missing_producer], components)
        raise AssertionError("an unknown producer was accepted")
    except ValueError as exc:
        assert "producer" in str(exc)
    quiet_past = dict(past)
    quiet_past["is_stateful"] = False
    try:
        validate_boundary_contracts([quiet_past], components)
        raise AssertionError("a past tensor was accepted as stateless")
    except ValueError as exc:
        assert "stateful" in str(exc)
    igpu = dict(past)
    igpu["device_preference"] = "igpu"
    try:
        validate_boundary_contracts([igpu], components)
        raise AssertionError("a stateful tensor on the igpu was accepted")
    except ValueError as exc:
        assert "npu" in str(exc)
    copied = dict(hidden)
    copied["zero_copy"] = False
    try:
        validate_boundary_contracts([copied], components)
        raise AssertionError("a handoff without zero-copy was accepted")
    except ValueError as exc:
        assert "zero_copy" in str(exc)
    broken_shape = dict(hidden)
    broken_shape["shape"] = [True]
    try:
        validate_boundary_contracts([broken_shape], components)
        raise AssertionError("a bool shape was accepted")
    except ValueError as exc:
        assert "dimension" in str(exc)
    try:
        validate_boundary_contracts("hidden", components)
        raise AssertionError("a boundary string was accepted")
    except ValueError as exc:
        assert "list" in str(exc)
    try:
        validate_component_io(["embedding"], [hidden])
        raise AssertionError("a component list was accepted")
    except ValueError as exc:
        assert "dict" in str(exc)
    blank = {
        "embedding": {"input_names": [" "], "output_names": ["hidden"]},
        "layer_0": {"input_names": ["hidden"], "output_names": ["logits"]},
    }
    try:
        validate_component_io(blank, [hidden])
        raise AssertionError("a blank input name was accepted")
    except ValueError as exc:
        assert "input_names" in str(exc)
    try:
        validate_component_io(components, [])
        raise AssertionError("a missing crossing was accepted")
    except ValueError as exc:
        assert "not a boundary" in str(exc)
    absent = dict(hidden)
    absent["name"] = "other"
    try:
        validate_boundary_contracts([absent], components)
        raise AssertionError("a boundary missing from the producer outputs was accepted")
    except ValueError as exc:
        assert "not an output" in str(exc)

    class _SplitModel:
        def __init__(self, graph):
            self.graph = graph
            self.ir_version = 8
            self.producer_name = "test"

    class _Node:
        def __init__(self, name):
            self.name = name
            self.op_type = "MatMul"
            self.input = ["hidden"]
            self.output = ["hidden_out"]

    split_graph = types.SimpleNamespace(
        node=[_Node("/model/layers.0/MatMul")],
        initializer=[],
        input=[],
        output=[],
        value_info=[],
    )
    original_load = splitter.load_onnx_model
    original_extract = splitter.extract_components
    original_create = splitter.create_component_onnx
    original_boundaries = splitter.collect_boundary_tensors
    original_save = getattr(splitter.onnx, "save", None)
    original_infer = getattr(splitter.shape_inference, "infer_shapes", None)

    def fake_load(path):
        return _SplitModel(split_graph)

    def fake_infer(model):
        return model

    def fake_extract(model, **kwargs):
        return {
            "embedding": ComponentInfo(component_type="embedding", node_indices=[0]),
            "layer_0": ComponentInfo(component_type="layer", layer_id=0, node_indices=[0]),
        }

    def fake_create(model, component, name):
        if name == "embedding":
            inputs, outputs = ["input_ids"], ["hidden"]
        else:
            inputs, outputs = ["hidden"], ["logits"]
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[types.SimpleNamespace(name=item) for item in inputs],
                output=[types.SimpleNamespace(name=item) for item in outputs],
            )
        )

    def fake_boundaries(model, components):
        return []

    saved_files = []

    def fake_save(model, filepath, **kwargs):
        saved_files.append(filepath)

    splitter.load_onnx_model = fake_load
    splitter.extract_components = fake_extract
    splitter.create_component_onnx = fake_create
    splitter.collect_boundary_tensors = fake_boundaries
    splitter.onnx.save = fake_save
    splitter.shape_inference.infer_shapes = fake_infer
    try:
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.onnx")
            output_path = os.path.join(directory, "out")
            os.makedirs(output_path)
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            saved_files.clear()
            try:
                splitter.split_onnx(model=model_path, output=output_path, split_only=True)
                raise AssertionError("a missing crossing was written")
            except ValueError as exc:
                assert "not a boundary" in str(exc)
            assert saved_files == []
            assert not os.path.exists(os.path.join(output_path, "model.onnx", "embeddings.onnx"))

            from chunking_utils.common.types import BoundaryTensor

            def fake_boundaries_ok(model, components):
                return [BoundaryTensor(
                    name="hidden",
                    dtype="float16",
                    shape=[1, 4],
                    producer_component="embedding",
                    consumer_components=["layer_0"],
                    is_stateful=False,
                    zero_copy=True,
                )]

            splitter.collect_boundary_tensors = fake_boundaries_ok
            saved_files.clear()
            result = splitter.split_onnx(model=model_path, output=output_path, split_only=True)
            info_path = os.path.join(result.output_dir, "model_info.json")
            saved = json.load(open(info_path, encoding="utf-8"))
            assert saved["boundary_tensors"][0]["name"] == "hidden"
            assert saved["components"]["embedding"]["output_names"] == ["hidden"]
            assert saved["components"]["layer_0"]["input_names"] == ["hidden"]
            assert len(saved_files) == 2
    finally:
        splitter.load_onnx_model = original_load
        splitter.extract_components = original_extract
        splitter.create_component_onnx = original_create
        splitter.collect_boundary_tensors = original_boundaries
        if original_save is None:
            del splitter.onnx.save
        else:
            splitter.onnx.save = original_save
        if original_infer is None:
            if hasattr(splitter.shape_inference, "infer_shapes"):
                del splitter.shape_inference.infer_shapes
        else:
            splitter.shape_inference.infer_shapes = original_infer

    print("split validation: OK")
    print("  psnr above 35 dB is good and above 40 dB is production")
    print("  boundary contract: producer, consumers, stateful NPU zero-copy")
    print("  component IO is checked before a file is written")
    return True


def test_weight_storage():
    """Shared, hard-linked, and streamed weights are deduplicated before files are written."""
    print("=" * 50)
    print("Testing weight storage...")
    print("=" * 50)

    import hashlib
    import json
    import os
    import sys
    import tempfile
    import types
    import warnings
    import chunking_utils

    try:
        from chunking_utils.onnx.weight_store import (
            WeightStrategy,
            coerce_weight_strategy,
            plan_weight_storage,
            store_component_weights,
        )
        import chunking_utils.onnx.weight_store as weight_store
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.weight_store import (
            WeightStrategy,
            coerce_weight_strategy,
            plan_weight_storage,
            store_component_weights,
        )
        import chunking_utils.onnx.weight_store as weight_store
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx.types import ComponentInfo

    assert chunking_utils.WeightStrategy is WeightStrategy
    assert chunking_utils.coerce_weight_strategy is coerce_weight_strategy
    assert chunking_utils.plan_weight_storage is plan_weight_storage
    assert chunking_utils.store_component_weights is store_component_weights
    assert WeightStrategy.COPY.value == "copy"
    assert WeightStrategy.HARDLINK.value == "hardlink"
    assert WeightStrategy.MANIFEST.value == "manifest"
    assert WeightStrategy.STREAMING.value == "streaming"
    assert coerce_weight_strategy(WeightStrategy.COPY) is WeightStrategy.COPY
    assert coerce_weight_strategy("hard-link") is WeightStrategy.HARDLINK
    assert coerce_weight_strategy("HARD LINK") is WeightStrategy.HARDLINK

    payload = b"shared-weight"
    other = b"other-bytes"
    mask = b"\x00\x01"

    def _components(first, second=None, first_name="embed.weight", second_name="head.weight"):
        items = [{
            "name": "embedding",
            "file": "embeddings.onnx",
            "initializers": [{"name": first_name, "data": first}],
        }]
        if second is not None:
            items.append({
                "name": "lm_head",
                "file": "lm_head.onnx",
                "initializers": [{"name": second_name, "data": second}],
            })
        return items

    def _store(directory, components, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = store_component_weights(components, directory, **kwargs)
        return record, caught

    with tempfile.TemporaryDirectory() as directory:
        missing = os.path.join(directory, "not-yet")
        planned_components = _components(payload, payload)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            planned = plan_weight_storage(
                planned_components,
                missing,
                strategy=WeightStrategy.COPY,
                size_threshold=1,
            )
        assert not os.path.exists(missing)
        assert planned_components[0]["initializers"][0]["data"] == payload
        assert any("weight strategy copy duplicates 1 initializer payload" in str(item.message) for item in caught)
        assert planned["deduplicated"] is False
        assert planned["shared_file"] is None
        assert planned["manifest"] is None
        assert planned["files"][0]["bytes"] == len(payload)
        assert "_data" not in planned["files"][0]
        planned["components"][0]["slices"].append({"name": "extra"})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            again = plan_weight_storage(
                _components(payload, payload),
                missing,
                strategy="copy",
                size_threshold=1,
            )
        assert len(again["components"][0]["slices"]) == 1
        assert again is not planned
        json.dumps(again)

        copy_dir = os.path.join(directory, "copy")
        copied, caught = _store(copy_dir, _components(payload, payload), strategy="copy", size_threshold=1)
        assert copied["strategy"] == "copy"
        assert copied["deduplicated"] is False
        assert any("duplicates 1 initializer payload" in str(item.message) for item in caught)
        embedding_data = os.path.join(copy_dir, "embeddings.onnx.data")
        head_data = os.path.join(copy_dir, "lm_head.onnx.data")
        assert open(embedding_data, "rb").read() == payload
        assert open(head_data, "rb").read() == payload
        assert not os.path.samefile(embedding_data, head_data)
        assert not os.path.exists(os.path.join(copy_dir, "weights"))
        assert copied["components"][0]["slices"][0]["location"] == "embeddings.onnx.data"
        assert copied["components"][0]["slices"][0]["offset"] == 0
        distinct, caught = _store(
            os.path.join(directory, "distinct"),
            _components(payload, other),
            strategy=WeightStrategy.COPY,
            size_threshold=1,
        )
        assert caught == []
        assert distinct["files"][0]["bytes"] == len(payload)
        assert distinct["files"][1]["bytes"] == len(other)
        same_component = [{
            "name": "embedding",
            "file": "embeddings.onnx",
            "initializers": [
                {"name": "a", "data": payload},
                {"name": "b", "data": payload},
            ],
        }]
        once, caught = _store(os.path.join(directory, "once"), same_component, strategy="copy", size_threshold=1)
        assert caught == []
        assert once["files"][0]["bytes"] == len(payload)
        assert once["components"][0]["slices"][0]["offset"] == 0
        assert once["components"][0]["slices"][1]["offset"] == 0

        kept = _components(b"abcd", None)
        inline, _ = _store(os.path.join(directory, "inline"), kept, strategy="copy", size_threshold=5)
        assert inline["components"][0]["slices"] == []
        assert kept[0]["initializers"][0]["data"] == b"abcd"
        external, _ = _store(
            os.path.join(directory, "threshold"),
            _components(b"abcd"),
            strategy="copy",
            size_threshold=4,
        )
        assert external["files"][0]["bytes"] == 4

        link_dir = os.path.join(directory, "link")
        linked, caught = _store(
            link_dir,
            _components(payload, payload, first_name="attention_mask"),
            strategy="hardlink",
            size_threshold=1,
        )
        assert caught == []
        assert linked["deduplicated"] is True
        assert linked["shared_file"] == "weights/shared.bin"
        assert linked["manifest"] == "weights/manifest.json"
        shared = os.path.join(link_dir, "weights", "shared.bin")
        assert os.path.getsize(shared) == len(payload)
        assert os.path.samefile(shared, os.path.join(link_dir, "embeddings.onnx.data"))
        assert os.path.samefile(shared, os.path.join(link_dir, "lm_head.onnx.data"))
        assert os.stat(shared).st_nlink >= 3
        assert linked["components"][0]["slices"][0]["name"] == "attention_mask"
        assert linked["components"][0]["slices"][0]["digest"] == hashlib.sha256(payload).hexdigest()
        assert linked["components"][1]["slices"][0]["offset"] == 0
        assert linked["components"][0]["slices"][0]["location"] == "embeddings.onnx.data"
        manifest = json.load(open(os.path.join(link_dir, "weights", "manifest.json"), encoding="utf-8"))
        assert manifest["strategy"] == "hardlink"
        assert manifest["tensors"][0]["name"] == "attention_mask"
        assert link_dir and linked["components"][0]["slices"][0]["digest"] != "int64"

        manifest_dir = os.path.join(directory, "manifest")
        shared_record, caught = _store(
            manifest_dir,
            _components(b"aaaa", b"bb"),
            strategy="manifest",
            size_threshold=1,
            chunk_size=4,
        )
        assert caught == []
        assert shared_record["chunk_size"] is None
        assert shared_record["deduplicated"] is True
        assert shared_record["components"][0]["slices"][0]["offset"] == 0
        assert shared_record["components"][1]["slices"][0]["offset"] == 4
        assert shared_record["components"][0]["slices"][0]["location"] == "weights/shared.bin"
        manifest_file = os.path.join(manifest_dir, "weights", "shared.bin")
        assert os.path.getsize(manifest_file) == 6
        assert not os.path.exists(os.path.join(manifest_dir, "embeddings.onnx.data"))
        written = json.load(open(os.path.join(manifest_dir, "weights", "manifest.json"), encoding="utf-8"))
        assert written["file"] == "weights/shared.bin"
        assert len(written["tensors"]) == 2
        assert shared_record["components"][0]["data_file"] == "weights/shared.bin"

        stream_dir = os.path.join(directory, "stream")
        streamed, caught = _store(
            stream_dir,
            _components(b"aaaa", b"bbbb"),
            strategy=WeightStrategy.STREAMING,
            size_threshold=1,
            chunk_size=6,
        )
        assert caught == []
        assert streamed["chunk_size"] == 6
        assert streamed["deduplicated"] is True
        assert len(streamed["files"]) == 2
        assert streamed["components"][0]["slices"][0]["location"] == "weights/chunks/chunk_000000.bin"
        assert streamed["components"][1]["slices"][0]["location"] == "weights/chunks/chunk_000001.bin"
        stream_doc = json.load(open(os.path.join(stream_dir, "weights", "stream.json"), encoding="utf-8"))
        assert stream_doc["chunks"][0]["bytes"] == 4
        packed, _ = _store(
            os.path.join(directory, "packed"),
            _components(b"aa", b"bb"),
            strategy="streaming",
            size_threshold=1,
            chunk_size=6,
        )
        assert len(packed["files"]) == 1
        assert packed["files"][0]["bytes"] == 4
        assert packed["components"][0]["slices"][0]["location"] == packed["components"][1]["slices"][0]["location"]
        oversized, caught = _store(
            os.path.join(directory, "over"),
            _components(b"c" * 10),
            strategy="streaming",
            size_threshold=1,
            chunk_size=4,
        )
        assert oversized["files"][0]["bytes"] == 10
        assert any("own chunk" in str(item.message) and "embed.weight" in str(item.message) for item in caught)

        viewed = [{
            "name": "embedding",
            "file": "embeddings.onnx",
            "initializers": [{"name": "embed.weight", "data": memoryview(payload)}],
        }]
        viewed_record, _ = _store(os.path.join(directory, "view"), viewed, strategy="manifest", size_threshold=1)
        assert viewed_record["files"][0]["bytes"] == len(payload)
        assert viewed[0]["initializers"][0]["data"] == b""
        assert viewed[0]["initializers"][0]["external_data"][0]["key"] == "location"
        assert viewed[0]["initializers"][0]["external_data"][0]["value"] == "weights/shared.bin"

        class _Entry:
            def __init__(self):
                self.key = ""
                self.value = ""

        class _External(list):
            def add(self):
                item = _Entry()
                self.append(item)
                return item

        class _Tensor:
            def __init__(self):
                self.name = "embed.weight"
                self.raw_data = payload
                self.external_data = _External()
                self.cleared = False

            def ClearField(self, field):
                assert field == "raw_data"
                self.cleared = True
                self.raw_data = b""

        tensor = _Tensor()
        proto_components = [{
            "name": "embedding",
            "file": "embeddings.onnx",
            "initializers": [tensor],
        }]
        proto, _ = _store(os.path.join(directory, "proto"), proto_components, strategy="copy", size_threshold=1)
        assert tensor.cleared is True
        assert tensor.raw_data == b""
        assert tensor.data_location == 1
        assert [(item.key, item.value) for item in tensor.external_data] == [
            ("location", "embeddings.onnx.data"),
            ("offset", "0"),
            ("length", str(len(payload))),
        ]
        assert proto["components"][0]["slices"][0]["length"] == len(payload)

        original_link = weight_store.os.link

        def _fail_link(*_args, **_kwargs):
            raise OSError("exdev")

        weight_store.os.link = _fail_link
        try:
            failed, caught = _store(
                os.path.join(directory, "fallback"),
                _components(payload, payload),
                strategy="hardlink",
                size_threshold=1,
            )
        finally:
            weight_store.os.link = original_link
        assert any("weight hardlink failed" in str(item.message) for item in caught)
        assert failed["files"][1]["hardlink_of"] is None
        fallback_shared = os.path.join(directory, "fallback", "weights", "shared.bin")
        fallback_copy = os.path.join(directory, "fallback", "embeddings.onnx.data")
        assert open(fallback_shared, "rb").read() == open(fallback_copy, "rb").read() == payload
        assert not os.path.samefile(fallback_shared, fallback_copy)

        named = store_component_weights(
            {"embedding": {"file": "embeddings.onnx", "initializers": [{"name": "embed.weight", "data": mask}]}},
            os.path.join(directory, "named"),
            strategy="manifest",
            size_threshold=1,
        )
        assert named["components"][0]["slices"][0]["name"] == "embed.weight"
        assert named["components"][0]["slices"][0]["length"] == len(mask)

        duplicate_init = [{
            "name": "embedding",
            "file": "embeddings.onnx",
            "initializers": [
                {"name": "embed.weight", "data": payload},
                {"name": "embed.weight", "data": other},
            ],
        }]
        try:
            store_component_weights(duplicate_init, os.path.join(directory, "dup-init"), size_threshold=1)
            raise AssertionError("a duplicate initializer was accepted")
        except ValueError as exc:
            assert "duplicate" in str(exc)
        try:
            store_component_weights(_components(payload) + _components(other), directory, size_threshold=1)
            raise AssertionError("a duplicate component was accepted")
        except ValueError as exc:
            assert "duplicate" in str(exc)
        try:
            store_component_weights(
                [{"name": " ", "file": "embeddings.onnx", "initializers": []}],
                directory,
            )
            raise AssertionError("a blank component name was accepted")
        except ValueError as exc:
            assert "component name" in str(exc)
        try:
            store_component_weights(
                [{"name": "embedding", "file": "embeddings.onnx", "initializers": [{"name": " ", "data": payload}]}],
                directory,
                size_threshold=1,
            )
            raise AssertionError("a blank initializer name was accepted")
        except ValueError as exc:
            assert "initializer name" in str(exc)
        try:
            store_component_weights(
                [{"name": "embedding", "file": "embeddings.onnx", "initializers": [{"name": "w", "data": "text"}]}],
                directory,
                size_threshold=1,
            )
            raise AssertionError("a text payload was accepted")
        except ValueError as exc:
            assert "bytes" in str(exc)
        try:
            store_component_weights([], directory)
            raise AssertionError("an empty component list was accepted")
        except ValueError as exc:
            assert "at least one" in str(exc)
        try:
            store_component_weights("embedding", directory)
            raise AssertionError("a component string was accepted")
        except ValueError as exc:
            assert "dict" in str(exc)
        try:
            coerce_weight_strategy("banana")
            raise AssertionError("an unknown strategy was accepted")
        except ValueError as exc:
            assert "hardlink" in str(exc)
        try:
            coerce_weight_strategy(True)
            raise AssertionError("a bool strategy was accepted")
        except ValueError as exc:
            assert "WeightStrategy" in str(exc)
        try:
            coerce_weight_strategy(" ")
            raise AssertionError("a blank strategy was accepted")
        except ValueError as exc:
            assert "WeightStrategy" in str(exc)
        try:
            store_component_weights(_components(payload), directory, size_threshold=True)
            raise AssertionError("a bool threshold was accepted")
        except ValueError as exc:
            assert "integer" in str(exc)
        try:
            store_component_weights(_components(payload), directory, size_threshold=-1)
            raise AssertionError("a negative threshold was accepted")
        except ValueError as exc:
            assert "non-negative" in str(exc)
        try:
            store_component_weights(_components(payload), directory, strategy="streaming", chunk_size=0)
            raise AssertionError("a zero chunk size was accepted")
        except ValueError as exc:
            assert "positive" in str(exc)
        try:
            store_component_weights(_components(payload), directory, strategy="copy", chunk_size=True)
            raise AssertionError("a bool chunk size was accepted")
        except ValueError as exc:
            assert "integer" in str(exc)
        try:
            store_component_weights(_components(payload), " ")
            raise AssertionError("a blank output directory was accepted")
        except ValueError as exc:
            assert "directory" in str(exc)
        occupied = os.path.join(directory, "occupied")
        with open(occupied, "w", encoding="utf-8") as handle:
            handle.write("file")
        try:
            store_component_weights(_components(payload), occupied, size_threshold=1)
            raise AssertionError("a file was accepted as the output directory")
        except ValueError as exc:
            assert "directory" in str(exc)
        try:
            store_component_weights(
                [{"name": "embedding", "file": "../embeddings.onnx", "initializers": []}],
                directory,
            )
            raise AssertionError("a parent path was accepted")
        except ValueError as exc:
            assert "base name" in str(exc)
        try:
            store_component_weights(
                [{"name": "embedding", "file": "nested/embeddings.onnx", "initializers": []}],
                directory,
            )
            raise AssertionError("a nested file name was accepted")
        except ValueError as exc:
            assert "base name" in str(exc)
        try:
            store_component_weights(
                [{"name": "embedding", "model": types.SimpleNamespace()}],
                directory,
            )
            raise AssertionError("a model without a graph was accepted")
        except ValueError as exc:
            assert "graph" in str(exc)
        try:
            store_component_weights(
                {"embedding": {"name": "layer_0", "initializers": []}},
                directory,
            )
            raise AssertionError("a mismatched component name was accepted")
        except ValueError as exc:
            assert "component name" in str(exc)
        empty_model, caught = _store(
            os.path.join(directory, "empty-model"),
            [{"name": "embedding", "file": "embeddings.onnx", "model": types.SimpleNamespace(graph=types.SimpleNamespace())}],
            strategy="copy",
            size_threshold=1,
        )
        assert caught == []
        assert empty_model["components"][0]["slices"] == []

    class _SplitModel:
        def __init__(self, graph):
            self.graph = graph
            self.ir_version = 8
            self.producer_name = "test"

    class _Node:
        def __init__(self, name):
            self.name = name
            self.op_type = "MatMul"
            self.input = ["hidden"]
            self.output = ["hidden_out"]

    class _Weight:
        def __init__(self, name, data):
            self.name = name
            self.raw_data = data
            self.external_data = None

    split_graph = types.SimpleNamespace(
        node=[_Node("/model/layers.0/MatMul")],
        initializer=[],
        input=[],
        output=[],
        value_info=[],
    )
    original_load = splitter.load_onnx_model
    original_extract = splitter.extract_components
    original_create = splitter.create_component_onnx
    original_boundaries = splitter.collect_boundary_tensors
    original_save = getattr(splitter.onnx, "save", None)
    original_infer = getattr(splitter.shape_inference, "infer_shapes", None)

    def fake_load(path):
        return _SplitModel(split_graph)

    def fake_infer(model):
        return model

    def fake_extract(model, **kwargs):
        return {
            "embedding": ComponentInfo(component_type="embedding", node_indices=[0]),
            "layer_0": ComponentInfo(component_type="layer", layer_id=0, node_indices=[0]),
        }

    def fake_create(model, component, name):
        if name == "embedding":
            inputs, outputs = ["input_ids"], ["hidden"]
        else:
            inputs, outputs = ["hidden"], ["logits"]
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[types.SimpleNamespace(name=item) for item in inputs],
                output=[types.SimpleNamespace(name=item) for item in outputs],
                initializer=[_Weight("attention_mask", payload)],
            )
        )

    from chunking_utils.common.types import BoundaryTensor

    def fake_boundaries(model, components):
        return [BoundaryTensor(
            name="hidden",
            dtype="float16",
            shape=[1, 4],
            producer_component="embedding",
            consumer_components=["layer_0"],
            is_stateful=False,
            zero_copy=True,
        )]

    saved_files = []

    def fake_save(model, filepath, **kwargs):
        saved_files.append(filepath)

    splitter.load_onnx_model = fake_load
    splitter.extract_components = fake_extract
    splitter.create_component_onnx = fake_create
    splitter.collect_boundary_tensors = fake_boundaries
    splitter.onnx.save = fake_save
    splitter.shape_inference.infer_shapes = fake_infer
    try:
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.onnx")
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("onnx")
            bad_output = os.path.join(directory, "bad")
            try:
                splitter.split_onnx(
                    model=model_path,
                    output=bad_output,
                    split_only=True,
                    weight_strategy="banana",
                )
                raise AssertionError("an unknown strategy was written")
            except ValueError as exc:
                assert "Unknown weight strategy" in str(exc)
            assert saved_files == []
            assert not os.path.exists(os.path.join(bad_output, "model.onnx", "embeddings.onnx"))

            output_path = os.path.join(directory, "out")
            result = splitter.split_onnx(
                model=model_path,
                output=output_path,
                split_only=True,
                weight_strategy="manifest",
                weight_size_threshold=1,
            )
            info_path = os.path.join(result.output_dir, "model_info.json")
            saved = json.load(open(info_path, encoding="utf-8"))
            assert saved["weight_storage"]["strategy"] == "manifest"
            assert saved["weight_storage"]["deduplicated"] is True
            assert saved["weight_storage"]["shared_file"] == "weights/shared.bin"
            shared_path = os.path.join(result.output_dir, "weights", "shared.bin")
            assert os.path.getsize(shared_path) == len(payload)
            assert not os.path.exists(os.path.join(result.output_dir, "embeddings.onnx.data"))
            slices = saved["weight_storage"]["components"]
            assert slices[0]["slices"][0]["name"] == "attention_mask"
            assert slices[0]["slices"][0]["digest"] == slices[1]["slices"][0]["digest"]
            assert slices[0]["slices"][0]["offset"] == 0
            assert len(saved_files) == 2
            json.dumps(saved["weight_storage"])
    finally:
        splitter.load_onnx_model = original_load
        splitter.extract_components = original_extract
        splitter.create_component_onnx = original_create
        splitter.collect_boundary_tensors = original_boundaries
        if original_save is None:
            del splitter.onnx.save
        else:
            splitter.onnx.save = original_save
        if original_infer is None:
            if hasattr(splitter.shape_inference, "infer_shapes"):
                del splitter.shape_inference.infer_shapes
        else:
            splitter.shape_inference.infer_shapes = original_infer

    print("weight storage: OK")
    print("  hardlink, manifest, and streaming store a shared payload once")
    print("  copy warns when a payload is duplicated")
    print("  model_info.json keeps the weight storage record")
    return True


def test_package_structure():
    """The ONNX package is the splitter, and splitter.py orchestrates it."""
    print("=" * 50)
    print("Testing package structure...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types
    import chunking_utils
    from chunking_utils.common.architecture import ArchitectureType

    try:
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.splitter as splitter

    from chunking_utils.onnx import boundary_contract, component_builder, graph_analysis
    from chunking_utils.onnx import kv_contract, metadata_writer, npu_policy, quantization
    from chunking_utils.onnx.splitter import (
        build_boundary_contract,
        extract_components,
        extract_kv_cache_info,
        save_components,
        split_onnx,
        write_execution_contract,
    )

    assert chunking_utils.extract_components is splitter.extract_components
    assert chunking_utils.extract_components is component_builder.extract_components
    assert chunking_utils.create_component_onnx is splitter.create_component_onnx
    assert chunking_utils.collect_boundary_tensors is splitter.collect_boundary_tensors
    assert chunking_utils.collect_boundary_tensors is boundary_contract.collect_boundary_tensors
    assert chunking_utils.resolve_tensor_type is splitter.resolve_tensor_type
    assert chunking_utils.resolve_tensor_type is graph_analysis.resolve_tensor_type
    assert chunking_utils.extract_quantization_info is quantization.extract_quantization_info
    assert chunking_utils.extract_kv_cache_info is kv_contract.extract_kv_cache_info
    assert chunking_utils.extract_execution_policy is npu_policy.extract_execution_policy
    assert chunking_utils.detect_architecture_from_onnx is graph_analysis.detect_architecture_from_onnx
    assert chunking_utils.build_boundary_contract is build_boundary_contract
    assert chunking_utils.save_components is save_components
    assert chunking_utils.write_execution_contract is write_execution_contract
    assert write_execution_contract is metadata_writer.write_execution_contract
    assert build_boundary_contract.__module__ == "chunking_utils.onnx.splitter"
    assert save_components.__module__ == "chunking_utils.onnx.splitter"
    assert split_onnx.__module__ == "chunking_utils.onnx.splitter"
    assert "splitter" not in component_builder.__dict__
    names = split_onnx.__code__.co_names
    for required in (
        "load_onnx_model",
        "detect_architecture",
        "extract_components",
        "build_boundary_contract",
        "extract_quantization_info",
        "extract_kv_cache_info",
        "save_components",
        "write_execution_contract",
    ):
        assert required in names

    class _Node:
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.input = list(inputs)
            self.output = list(outputs)
            self.op_type = "Gather"

    class _Graph:
        def __init__(self, nodes):
            self.node = list(nodes)
            self.initializer = []
            self.output = []
            self.input = []
            self.value_info = []

    class _Model:
        def __init__(self, graph):
            self.graph = graph

    llama = _Model(_Graph([
        _Node("/model/embed/Gather", ["input_ids"], ["hidden"]),
    ]))
    detected = extract_components(llama)
    forced = extract_components(llama, ArchitectureType.GPT2)
    described = extract_components(llama, arch=type("Desc", (), {"arch_type": ArchitectureType.LLAMA})())
    configured = extract_components(llama, {"num_hidden_layers": 1})
    assert "embedding" in detected
    assert "embedding" in described
    assert "embedding" in configured
    assert "embedding" not in forced
    try:
        extract_components(llama, arch="gpt2")
        raise AssertionError("a string architecture was accepted")
    except ValueError as exc:
        assert "ArchitectureType" in str(exc)
    try:
        extract_components(llama, ArchitectureType.LLAMA, arch=ArchitectureType.GPT2)
        raise AssertionError("two architectures were accepted")
    except ValueError as exc:
        assert "disagree" in str(exc)

    empty = extract_kv_cache_info(_Graph([]))
    with_components = extract_kv_cache_info(_Graph([]), {"layer_0": object()})
    assert empty == with_components
    assert empty["num_layers"] == 0
    try:
        extract_kv_cache_info(_Graph([]), ["layer_0"])
        raise AssertionError("a list of components was accepted")
    except ValueError as exc:
        assert "dict" in str(exc)

    original_boundaries = splitter.collect_boundary_tensors
    def fake_boundaries(model, components):
        return ["from-patch"]
    splitter.collect_boundary_tensors = fake_boundaries
    try:
        assert build_boundary_contract(llama, {}) == ["from-patch"]
    finally:
        splitter.collect_boundary_tensors = original_boundaries

    with tempfile.TemporaryDirectory() as directory:
        info_path = write_execution_contract(directory, {"architecture": "llama"})
        with open(info_path, encoding="utf-8") as handle:
            assert json.load(handle)["architecture"] == "llama"
        file_path = os.path.join(directory, "not_a_dir")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("x")
        try:
            write_execution_contract(file_path, {})
            raise AssertionError("a file was accepted as a directory")
        except ValueError as exc:
            assert "directory" in str(exc)
        try:
            write_execution_contract("  ", {})
            raise AssertionError("a blank directory was accepted")
        except ValueError as exc:
            assert "directory" in str(exc)
        try:
            write_execution_contract(directory, ["nope"])
            raise AssertionError("a list model_info was accepted")
        except ValueError as exc:
            assert "dict" in str(exc)
        try:
            write_execution_contract(directory, {"bad": {1}})
            raise AssertionError("a non-JSON model_info was accepted")
        except ValueError as exc:
            assert "JSON" in str(exc)

    print("package structure: OK")
    print("  splitter.py orchestrates the onnx modules")
    print("  an architecture argument selects component prefixes")
    print("  kv cache metadata accepts the component map")
    return True


def test_merge_order():
    """An execution plan selects merge order for MoE, hybrid, and MTP graphs."""
    print("=" * 50)
    print("Testing merge order...")
    print("=" * 50)

    import inspect
    import json
    import os
    import sys
    import tempfile
    import types
    import chunking_utils

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    from chunking_utils.common.types import ExecutionPlan

    assert chunking_utils.merge_onnx is merger.merge_onnx
    signature = inspect.signature(merger.merge_onnx)
    assert list(signature.parameters) == ["output_dir", "save_path", "execution_plan"]

    loaded = []
    captured = {}

    def fake_load(filepath, load_external_data=True):
        name = os.path.splitext(os.path.basename(filepath))[0]
        loaded.append(name)
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                node=[types.SimpleNamespace(name=name)],
                initializer=[types.SimpleNamespace(name=name + "_weight")],
                input=[types.SimpleNamespace(name=name + "_in")],
                output=[types.SimpleNamespace(name=name + "_out")],
            ),
            opset_import=[],
            ir_version=8,
        )

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["nodes"] = [node.name for node in nodes]
        captured["outputs"] = [item.name for item in outputs]
        return types.SimpleNamespace(
            node=list(nodes),
            input=list(inputs),
            output=list(outputs),
            initializer=list(initializer),
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(
            graph=graph,
            producer_name=producer_name,
            ir_version=None,
        )

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    if not hasattr(merger.onnx, "save"):
        merger.onnx.save = None
    original_load = merger.onnx.load
    original_helper = merger.helper
    merger.onnx.load = fake_load
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)

    def write_components(directory, names, num_layers=1, include_layers=True, files=True):
        payload = {"components": {name: {"file": name + ".onnx"} for name in names}}
        if include_layers:
            payload["num_layers"] = num_layers
        with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        if files:
            for name in names:
                with open(os.path.join(directory, name + ".onnx"), "w", encoding="utf-8") as handle:
                    handle.write("onnx")

    try:
        with tempfile.TemporaryDirectory() as directory:
            write_components(
                directory,
                ["embedding", "layer_0", "norm", "lm_head", "other", "moe_router"],
                num_layers=1,
            )
            loaded.clear()
            merged = merger.merge_onnx(directory)
            assert loaded == ["embedding", "layer_0", "norm", "lm_head", "other"]
            assert captured["outputs"] == ["lm_head_out"]
            assert merged.producer_name == "chunking_utils_merged"
            assert "moe_router" not in loaded

            moe_names = [
                "embedding",
                "moe_router",
                "moe_expert_0",
                "layer_0",
                "norm",
                "lm_head",
                "mtp_head",
            ]
            write_components(directory, moe_names, include_layers=False)
            plan = {
                "steps": [
                    {"component_name": "embedding"},
                    {"component_name": "moe_router", "depends_on": ["embedding"], "phase": "router"},
                    {"component_name": "moe_expert_0", "depends_on": ["moe_router"], "phase": "expert"},
                    {"component_name": "layer_0", "depends_on": ["moe_expert_0"]},
                    {"component_name": "norm", "depends_on": ["layer_0"]},
                    {"component_name": "lm_head", "depends_on": ["norm"]},
                    {"component_name": "mtp_head", "depends_on": ["lm_head"]},
                ],
                "policy": "moe_routing",
            }
            loaded.clear()
            merger.merge_onnx(directory, execution_plan=plan)
            assert loaded == moe_names
            assert captured["outputs"] == ["mtp_head_out"]

            hybrid_names = ["gated_deltanet_0", "sparse_attention_0"]
            write_components(directory, hybrid_names, include_layers=False)
            loaded.clear()
            merger.merge_onnx(
                directory,
                execution_plan=ExecutionPlan(steps=[
                    {"component_name": "gated_deltanet_0", "phase": "prefill"},
                    {"component_name": "sparse_attention_0", "depends_on": ["gated_deltanet_0"], "phase": "decode"},
                ]),
            )
            assert loaded == hybrid_names
            assert captured["outputs"] == ["sparse_attention_0_out"]

            write_components(directory, ["embedding", "mtp_head"], include_layers=False)
            os.remove(os.path.join(directory, "mtp_head.onnx"))
            try:
                merger.merge_onnx(directory, execution_plan={
                    "steps": [
                        {"component_name": "embedding"},
                        {"component_name": "mtp_head", "depends_on": ["embedding"]},
                    ],
                })
                raise AssertionError("a missing planned file was skipped")
            except ValueError as exc:
                assert "not found" in str(exc)

            write_components(directory, ["embedding", "moe_expert_0"], include_layers=False)
            try:
                merger.merge_onnx(directory, execution_plan={
                    "steps": [{"component_name": "embedding"}],
                })
                raise AssertionError("an omitted expert was accepted")
            except ValueError as exc:
                assert "omits" in str(exc)

            write_components(directory, ["embedding"], include_layers=False)
            try:
                merger.merge_onnx(directory, execution_plan={
                    "steps": [
                        {"component_name": "embedding"},
                        {"component_name": "moe_expert_0"},
                    ],
                })
                raise AssertionError("an unknown component was accepted")
            except ValueError as exc:
                assert "unknown" in str(exc)

            write_components(directory, ["embedding", "mtp_head"], include_layers=False)
            try:
                merger.merge_onnx(directory, execution_plan={
                    "steps": [
                        {"component_name": "mtp_head", "depends_on": ["embedding"]},
                        {"component_name": "embedding"},
                    ],
                })
                raise AssertionError("a later dependency was accepted")
            except ValueError as exc:
                assert "merged later" in str(exc)

            write_components(directory, ["encoder", "decoder"], include_layers=False)
            try:
                merger.merge_onnx(directory, execution_plan={
                    "steps": [
                        {"component_name": "encoder", "depends_on": ["decoder"]},
                        {"component_name": "decoder", "depends_on": ["encoder"]},
                    ],
                })
                raise AssertionError("a dependency cycle was accepted")
            except ValueError as exc:
                assert "cycle" in str(exc)

            try:
                merger.merge_onnx(directory, execution_plan=["embedding"])
                raise AssertionError("a list plan was accepted")
            except ValueError as exc:
                assert "dict" in str(exc)

            result = merger.merge_onnx_with_result(directory, execution_plan=True)
            assert result.success is False
            assert "dict" in result.error

            with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
                handle.write("{")
            try:
                merger.merge_onnx(directory)
                raise AssertionError("broken JSON was accepted")
            except ValueError as exc:
                assert "JSON" in str(exc)

            with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
                handle.write("[]")
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a list model_info was accepted")
            except ValueError as exc:
                assert "dict" in str(exc)

        try:
            merger.merge_onnx("  ")
            raise AssertionError("a blank directory was accepted")
        except ValueError as exc:
            assert "directory" in str(exc)

        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "not_a_dir")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("x")
            try:
                merger.merge_onnx(file_path)
                raise AssertionError("a file was accepted as a directory")
            except ValueError as exc:
                assert "directory" in str(exc)
    finally:
        merger.onnx.load = original_load
        merger.helper = original_helper

    print("merge order: OK")
    print("  a missing plan keeps the linear chain")
    print("  a plan merges MoE, hybrid, and MTP components in step order")
    return True


def test_other_dependencies():
    """other is inserted where its inputs are produced."""
    print("=" * 50)
    print("Testing other dependencies...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    loaded = []

    def fake_load(filepath, load_external_data=True):
        name = os.path.splitext(os.path.basename(filepath))[0]
        loaded.append(name)
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                node=[types.SimpleNamespace(name=name)],
                initializer=[],
                input=[],
                output=[types.SimpleNamespace(name=name + "_out")],
            ),
            opset_import=[],
            ir_version=8,
        )

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["outputs"] = [item.name for item in outputs]
        return types.SimpleNamespace(node=list(nodes), output=list(outputs), initializer=list(initializer), input=list(inputs))

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(graph=graph, producer_name=producer_name, ir_version=None)

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    original_load = merger.onnx.load
    original_helper = merger.helper
    captured = {}
    merger.onnx.load = fake_load
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)

    def component(filename, **fields):
        record = {"file": filename}
        record.update(fields)
        return record

    def write_info(directory, components, boundary=None):
        payload = {"num_layers": 1, "components": components}
        if boundary is not None:
            payload["boundary_tensors"] = boundary
        with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        for info in components.values():
            with open(os.path.join(directory, info["file"]), "w", encoding="utf-8") as handle:
                handle.write("onnx")

    chain = ["embedding", "layer_0", "norm", "lm_head", "other"]

    try:
        unknown = {"components": {"other": {"file": "other.onnx"}}}
        assert merger.analyze_component_dependencies("other", unknown) is None
        assert merger.find_correct_insert_position(None, ["embedding", "lm_head"]) == 2

        external = {
            "components": {
                "embedding": component("embedding.onnx", output_names=["hidden"]),
                "other": component("other.onnx", input_names=["input_ids"]),
            }
        }
        placement = merger.analyze_component_dependencies("other", external)
        assert placement["depends_on"] == []
        assert merger.find_correct_insert_position(placement, ["embedding", "layer_0", "norm", "lm_head"]) == 0

        with tempfile.TemporaryDirectory() as directory:
            write_info(directory, {
                "embedding": component("embedding.onnx", output_names=["hidden"]),
                "layer_0": component("layer_0.onnx", input_names=["hidden"], output_names=["h0"]),
                "norm": component("norm.onnx", input_names=["h0"]),
                "lm_head": component("lm_head.onnx"),
                "other": component("other.onnx", input_names=["input_ids", "position_ids"]),
            })
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["other", "embedding", "layer_0", "norm", "lm_head"]
            assert captured["outputs"] == ["lm_head_out"]

            write_info(directory, {
                "embedding": component("embedding.onnx", output_names=["hidden"]),
                "layer_0": component("layer_0.onnx"),
                "norm": component("norm.onnx"),
                "lm_head": component("lm_head.onnx"),
                "other": component("other.onnx", input_names=["hidden"]),
            })
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["embedding", "other", "layer_0", "norm", "lm_head"]

            write_info(directory, {
                "embedding": component("embedding.onnx"),
                "layer_0": component("layer_0.onnx", output_names=["h0"]),
                "norm": component("norm.onnx"),
                "lm_head": component("lm_head.onnx"),
                "other": component("other.onnx", input_names=["h0"]),
            })
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["embedding", "layer_0", "other", "norm", "lm_head"]

            write_info(directory, {
                "embedding": component("embedding.onnx", output_names=["token"]),
                "layer_0": component("layer_0.onnx"),
                "norm": component("norm.onnx"),
                "lm_head": component("lm_head.onnx"),
                "other": component("other.onnx", depends_on=["embedding"]),
            })
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["embedding", "other", "layer_0", "norm", "lm_head"]

            write_info(
                directory,
                {
                    "embedding": component("embedding.onnx"),
                    "layer_0": component("layer_0.onnx"),
                    "norm": component("norm.onnx"),
                    "lm_head": component("lm_head.onnx"),
                    "other": component("other.onnx"),
                },
                boundary=[{
                    "producer": "embedding",
                    "producer_component": "embedding",
                    "consumers": ["other"],
                    "consumer_components": ["other"],
                }],
            )
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["embedding", "other", "layer_0", "norm", "lm_head"]

            write_info(directory, {
                "embedding": component("embedding.onnx"),
                "layer_0": component("layer_0.onnx", input_names=["ngram"]),
                "norm": component("norm.onnx"),
                "lm_head": component("lm_head.onnx"),
                "other": component("other.onnx", output_names=["ngram"], input_names=["input_ids"]),
            })
            loaded.clear()
            merger.merge_onnx(directory)
            assert loaded == ["other", "embedding", "layer_0", "norm", "lm_head"]

            cycle = {
                "num_layers": 1,
                "components": {
                    "embedding": component("embedding.onnx"),
                    "layer_0": component("layer_0.onnx", output_names=["h0"], input_names=["side"]),
                    "norm": component("norm.onnx"),
                    "lm_head": component("lm_head.onnx"),
                    "other": component("other.onnx", input_names=["h0"], output_names=["side"]),
                },
            }
            try:
                merger.analyze_component_dependencies("other", cycle)
                merger.find_correct_insert_position(
                    merger.analyze_component_dependencies("other", cycle),
                    ["embedding", "layer_0", "norm", "lm_head"],
                )
                raise AssertionError("a dependency cycle was accepted")
            except ValueError as exc:
                assert "reads" in str(exc)

            missing = {
                "num_layers": 1,
                "components": {
                    "other": component("other.onnx", depends_on=["moe_router"]),
                },
            }
            try:
                merger.find_correct_insert_position(
                    merger.analyze_component_dependencies("other", missing),
                    ["embedding", "layer_0", "norm", "lm_head"],
                )
                raise AssertionError("a dependency outside the chain was accepted")
            except ValueError as exc:
                assert "not in the merge order" in str(exc)

            try:
                merger.analyze_component_dependencies("other", {
                    "components": {"other": {"depends_on": "embedding"}},
                })
                raise AssertionError("a string depends_on was accepted")
            except ValueError as exc:
                assert "list" in str(exc)

            write_info(directory, {
                "embedding": component("embedding.onnx"),
                "other": component("other.onnx", input_names=["input_ids"]),
            })
            loaded.clear()
            merger.merge_onnx(directory, execution_plan={
                "steps": [
                    {"component_name": "embedding"},
                    {"component_name": "other", "depends_on": ["embedding"]},
                ],
            })
            assert loaded == ["embedding", "other"]
    finally:
        merger.onnx.load = original_load
        merger.helper = original_helper

    assert chain[0] == "embedding"
    print("other dependencies: OK")
    print("  external inputs run before the layers")
    print("  an embedding input runs before layer_0")
    return True


def test_boundary_reconnection():
    """Boundary outputs reconnect to downstream inputs with the same name, dtype, and shape."""
    print("=" * 50)
    print("Testing boundary reconnection...")
    print("=" * 50)

    import json
    import os
    import sys
    import tempfile
    import types

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    class _Dim:
        def __init__(self, value=None, param=""):
            self.dim_value = 0 if value is None else value
            self.dim_param = "" if value is not None else (param or "sym")

    class _Value:
        def __init__(self, name, elem=None, shape=None):
            self.name = name
            if elem is None and shape is None:
                return
            dims = None
            if shape is not None:
                dims = [_Dim(dim) for dim in shape]
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(
                    elem_type=0 if elem is None else elem,
                    shape=None if dims is None else types.SimpleNamespace(dim=dims),
                )
            )

    def model(graph_name, inputs, outputs):
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                name=graph_name,
                input=list(inputs),
                output=list(outputs),
                node=[],
                initializer=[],
            ),
            opset_import=[],
            ir_version=8,
        )

    def component(filename, **fields):
        record = {"file": filename}
        record.update(fields)
        return record

    hidden = {
        "name": "hidden",
        "dtype": "float16",
        "shape": [1, None, 16],
        "producer": "embedding",
        "producer_component": "embedding",
        "consumers": ["layer_0"],
        "consumer_components": ["layer_0"],
    }
    embedding = model(
        "embedding_graph",
        [_Value("input_ids", 7, [1, None])],
        [_Value("hidden", 10, [1, None, 16])],
    )
    other = model(
        "other_graph",
        [_Value("input_ids", 7, [1, None])],
        [_Value("ngram", 7, [1, None])],
    )
    layer = model(
        "layer_0_graph",
        [_Value("hidden", 10, [1, 4, 16]), _Value("ngram", 7, [1, None])],
        [_Value("h0", 10, [1, 4, 16])],
    )
    chain_info = {
        "num_layers": 1,
        "components": {
            "embedding": component("embedding.onnx", output_names=["hidden"]),
            "other": component("other.onnx", output_names=["ngram"]),
            "layer_0": component("layer_0.onnx", input_names=["hidden", "ngram"], output_names=["h0"]),
        },
        "boundary_tensors": [
            hidden,
            {
                "name": "ngram",
                "dtype": "int64",
                "shape": [1, None],
                "producer": "other",
                "consumers": ["layer_0"],
            },
        ],
    }
    report = merger.validate_boundary_reconnection([embedding, other, layer], chain_info)
    assert report["valid"] is True
    assert report["errors"] == []

    early = model("embedding_graph", [_Value("input_ids", 7, [1, None])], [_Value("token", 7, [1, None])])
    middle = model("layer_0_graph", [_Value("hidden", 10, [1, None, 16])], [_Value("h0", 10, [1, None, 16])])
    late = model("lm_head_graph", [_Value("h0", 10, [1, None, 16])], [_Value("hidden", 10, [1, None, 16])])
    unresolved = merger.validate_boundary_reconnection([early, middle, late], {"components": {}})
    assert unresolved["valid"] is False
    assert "Unresolved input: hidden" in unresolved["errors"]

    renamed_layer = model(
        "layer_0_graph",
        [_Value("hidden_states", 10, [1, None, 16])],
        [_Value("h0", 10, [1, None, 16])],
    )
    renamed = merger.validate_boundary_reconnection(
        [embedding, renamed_layer],
        {"components": chain_info["components"], "boundary_tensors": [hidden]},
    )
    assert renamed["valid"] is False
    assert any("not an input of layer_0" in error for error in renamed["errors"])

    wide = model(
        "layer_0_graph",
        [_Value("hidden", 1, [1, 8, 32])],
        [_Value("h0", 1, [1, 8, 32])],
    )
    mismatched = merger.validate_boundary_reconnection(
        [embedding, wide],
        {"components": chain_info["components"], "boundary_tensors": [hidden]},
    )
    assert mismatched["valid"] is False
    assert any("dtype" in error for error in mismatched["errors"])
    assert any("shape" in error for error in mismatched["errors"])

    missing_output = model(
        "embedding_graph",
        [_Value("input_ids", 7, [1, None])],
        [_Value("token", 7, [1, None])],
    )
    absent = merger.validate_boundary_reconnection(
        [missing_output, layer],
        {"components": chain_info["components"], "boundary_tensors": [hidden]},
    )
    assert absent["valid"] is False
    assert any("not an output of embedding" in error for error in absent["errors"])

    alias = dict(hidden)
    alias["dtype"] = "fp16"
    aliased = merger.validate_boundary_reconnection(
        [embedding, layer],
        {"components": chain_info["components"], "boundary_tensors": [alias]},
    )
    assert aliased["valid"] is True

    try:
        merger.validate_boundary_reconnection("components", {})
        raise AssertionError("a string component list was accepted")
    except ValueError as exc:
        assert "list" in str(exc)

    try:
        merger.validate_boundary_reconnection([], [])
        raise AssertionError("a list model_info was accepted")
    except ValueError as exc:
        assert "dict" in str(exc)

    try:
        merger.validate_boundary_reconnection([], {"boundary_tensors": "hidden"})
        raise AssertionError("a string boundary list was accepted")
    except ValueError as exc:
        assert "list" in str(exc)

    try:
        merger.validate_boundary_reconnection(
            [model("embedding_graph", [_Value(" ")], [])],
            {},
        )
        raise AssertionError("a blank tensor name was accepted")
    except ValueError as exc:
        assert "name" in str(exc)

    try:
        merger.validate_boundary_reconnection([], {"graph_inputs": "input_ids"})
        raise AssertionError("a string graph_inputs was accepted")
    except ValueError as exc:
        assert "list" in str(exc)

    captured = {}

    def fake_load(filepath, load_external_data=True):
        name = os.path.splitext(os.path.basename(filepath))[0]
        if name == "embedding":
            return embedding
        if name == "layer_0":
            return wide
        return model(name + "_graph", [], [_Value(name + "_out")])

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["built"] = True
        return types.SimpleNamespace(node=list(nodes), input=list(inputs), output=list(outputs), initializer=list(initializer))

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(graph=graph, producer_name=producer_name, ir_version=None)

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    if not hasattr(merger.onnx, "save"):
        merger.onnx.save = None
    original_load = merger.onnx.load
    original_helper = merger.helper
    merger.onnx.load = fake_load
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)
    try:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "num_layers": 1,
                "components": {
                    "embedding": component("embedding.onnx", output_names=["hidden"]),
                    "layer_0": component("layer_0.onnx", input_names=["hidden"], output_names=["h0"]),
                    "norm": component("norm.onnx"),
                    "lm_head": component("lm_head.onnx"),
                    "other": component("other.onnx"),
                },
                "boundary_tensors": [hidden],
            }
            with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            for info in payload["components"].values():
                with open(os.path.join(directory, info["file"]), "w", encoding="utf-8") as handle:
                    handle.write("onnx")
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a dtype mismatch was merged")
            except ValueError as exc:
                assert "boundary reconnection failed" in str(exc)
                assert "dtype" in str(exc)
            assert captured.get("built") is not True

            result = merger.merge_onnx_with_result(directory)
            assert result.success is False
            assert "boundary" in result.error

            payload["boundary_tensors"] = []
            payload["components"] = {
                "embedding": component("embedding.onnx"),
                "layer_0": component("layer_0.onnx"),
            }
            with open(os.path.join(directory, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            captured.clear()
            merger.onnx.load = lambda filepath, load_external_data=True: model(
                os.path.splitext(os.path.basename(filepath))[0] + "_graph",
                [_Value(os.path.splitext(os.path.basename(filepath))[0] + "_in")],
                [_Value(os.path.splitext(os.path.basename(filepath))[0] + "_out")],
            )
            merged = merger.merge_onnx(directory, execution_plan={
                "steps": [
                    {"component_name": "embedding"},
                    {"component_name": "layer_0", "depends_on": ["embedding"]},
                ],
            })
            assert captured.get("built") is True
            assert merged.producer_name == "chunking_utils_merged"
    finally:
        merger.onnx.load = original_load
        merger.helper = original_helper

    print("boundary reconnection: OK")
    print("  a renamed input, a dtype mismatch, and a shape mismatch raise")
    return True


def test_quantization_coherence():
    """Scale, zero-point, and packing metadata stay intact across split and merge."""
    print("=" * 50)
    print("Testing quantization coherence...")
    print("=" * 50)

    import inspect
    import json
    import os
    import sys
    import tempfile
    import types

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    assert list(inspect.signature(merger.merge_onnx).parameters) == [
        "output_dir",
        "save_path",
        "execution_plan",
    ]

    def attr(name, value):
        return types.SimpleNamespace(name=name, type=2, i=value)

    def init(name, dtype):
        return types.SimpleNamespace(name=name, data_type=dtype)

    def quant_node(op_type, inputs, attributes=None):
        return types.SimpleNamespace(
            op_type=op_type,
            name=op_type if isinstance(op_type, str) else "quant",
            input=list(inputs),
            attribute=[] if attributes is None else list(attributes),
        )

    def quant_model(op_type="MatMulNBits", inputs=None, attributes=None, initializers=None):
        if inputs is None:
            inputs = ["a", "w", "s", "z"]
        if attributes is None:
            attributes = [attr("bits", 4), attr("block_size", 32)]
        if initializers is None:
            initializers = [init("w", 2), init("s", 10), init("z", 2)]
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                name="quant_graph",
                node=[quant_node(op_type, inputs, attributes)],
                initializer=list(initializers),
                input=[],
                output=[],
                value_info=[],
            ),
            opset_import=[],
            ir_version=8,
        )

    original = quant_model()
    report = merger.verify_quantization_coherence(original, quant_model())
    assert report["coherent"] is True
    assert report["original"] == report["merged"]
    assert report["original"]["ops"]["MatMulNBits"] == 1
    assert report["original"]["ops"]["DequantizeLinear"] == 0
    assert report["original"]["scale_tensors"] == ["s"]
    assert report["original"]["zero_point_tensors"] == ["z"]
    assert report["original"]["scale_initializers"] == ["s"]
    assert report["original"]["zero_point_initializers"] == ["z"]
    assert report["original"]["packing_formats"] == ["packed_uint8"]
    assert report["original"]["bits"] == [4]
    assert report["original"]["group_sizes"] == [32]
    assert report["original"]["unset_bits"] == 0

    dropped_scale = quant_model(inputs=["a", "w", "", "z"])
    dropped = merger.verify_quantization_coherence(original, dropped_scale)
    assert dropped["coherent"] is False
    assert dropped["merged"]["scale_tensors"] == []

    missing_scale = quant_model(initializers=[init("w", 2), init("z", 2)])
    missing = merger.verify_quantization_coherence(original, missing_scale)
    assert missing["coherent"] is False
    assert missing["merged"]["scale_tensors"] == ["s"]
    assert missing["merged"]["scale_initializers"] == []

    dropped_zero = quant_model(inputs=["a", "w", "s"])
    no_zero = merger.verify_quantization_coherence(original, dropped_zero)
    assert no_zero["coherent"] is False
    assert no_zero["merged"]["zero_point_tensors"] == []

    packed_int = quant_model(initializers=[init("w", 6), init("s", 10), init("z", 2)])
    repacked = merger.verify_quantization_coherence(original, packed_int)
    assert repacked["coherent"] is False
    assert repacked["merged"]["packing_formats"] == ["packed_int32"]

    wider = quant_model(attributes=[attr("bits", 8), attr("block_size", 32)])
    wider_report = merger.verify_quantization_coherence(original, wider)
    assert wider_report["coherent"] is False
    assert wider_report["merged"]["bits"] == [8]
    assert wider_report["merged"]["packing_formats"] == ["dense"]

    dequantized = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            name="dequant_graph",
            node=[
                quant_node("DequantizeLinear", ["w", "s", "z"]),
                quant_node("MatMul", ["a", "w_f"]),
            ],
            initializer=[init("w", 2), init("s", 10), init("z", 2)],
            input=[],
            output=[],
            value_info=[],
        )
    )
    expanded = merger.verify_quantization_coherence(original, dequantized)
    assert expanded["coherent"] is False
    assert expanded["merged"]["ops"]["MatMulNBits"] == 0
    assert expanded["merged"]["ops"]["DequantizeLinear"] == 1
    assert expanded["original"]["ops"]["DequantizeLinear"] == 0

    dense = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            name="dense_graph",
            node=[quant_node("MatMul", ["a", "b"])],
            initializer=[],
            input=[],
            output=[],
        )
    )
    dense_report = merger.verify_quantization_coherence(dense, dense)
    assert dense_report["coherent"] is True
    assert dense_report["original"]["ops"]["MatMulNBits"] == 0
    assert dense_report["original"]["scale_tensors"] == []
    assert dense_report["original"]["packing_formats"] == []

    qlinear = quant_model(
        op_type="QLinearMatMul",
        inputs=["a", "a_s", "a_z", "w", "s", "z"],
        attributes=[attr("bits", 8)],
    )
    qdq = merger.verify_quantization_coherence(qlinear, quant_model(
        op_type="QLinearMatMul",
        inputs=["a", "a_s", "a_z", "w", "s", "z"],
        attributes=[attr("bits", 8)],
    ))
    assert qdq["coherent"] is True
    assert qdq["original"]["ops"]["QLinearMatMul"] == 1
    assert qdq["original"]["packing_formats"] == ["qdq"]

    try:
        merger.verify_quantization_coherence("original", original)
        raise AssertionError("a string model was counted")
    except ValueError as exc:
        assert "graph" in str(exc)

    duplicated = quant_model(initializers=[init("w", 2), init("s", 10), init("s", 10)])
    try:
        merger.count_quant_ops(duplicated)
        raise AssertionError("a duplicated initializer was accepted")
    except ValueError as exc:
        assert "duplicated" in str(exc)

    blank_name = quant_model(initializers=[init("  ", 2)])
    try:
        merger.count_quant_ops(blank_name)
        raise AssertionError("a blank initializer name was accepted")
    except ValueError as exc:
        assert "initializer name" in str(exc)

    bad_op = quant_model()
    bad_op.graph.node[0].op_type = 4
    try:
        merger.count_quant_ops(bad_op)
        raise AssertionError("a non-string op_type was accepted")
    except ValueError as exc:
        assert "op_type" in str(exc)

    captured = {}
    state = {"model": quant_model()}

    def fake_load(filepath, load_external_data=True):
        return state["model"]

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["built"] = True
        return types.SimpleNamespace(
            node=list(nodes),
            name=name,
            input=list(inputs),
            output=list(outputs),
            initializer=list(initializer),
            value_info=[],
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(graph=graph, producer_name=producer_name, ir_version=None)

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    if not hasattr(merger.onnx, "save"):
        merger.onnx.save = None
    original_load = merger.onnx.load
    original_helper = merger.helper
    merger.onnx.load = fake_load
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)
    try:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "num_layers": 0,
                "components": {"embedding": {"file": "embedding.onnx"}},
                "quantization_info": {
                    "method": "MatMulNBits",
                    "has_zero_point": True,
                    "packing_format": "packed_uint8",
                    "bits": 4,
                    "group_size": 32,
                },
            }
            info_path = os.path.join(directory, "model_info.json")
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with open(os.path.join(directory, "embedding.onnx"), "w", encoding="utf-8") as handle:
                handle.write("onnx")

            state["model"] = types.SimpleNamespace(
                graph=types.SimpleNamespace(
                    name="empty_graph",
                    node=[],
                    initializer=[],
                    input=[],
                    output=[],
                ),
                opset_import=[],
                ir_version=8,
            )
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a merge without MatMulNBits was accepted")
            except ValueError as exc:
                assert "quantization coherence failed" in str(exc)
                assert "has none" in str(exc)
            assert captured.get("built") is True

            result = merger.merge_onnx_with_result(directory)
            assert result.success is False
            assert "quantization" in result.error

            payload["quantization_info"] = "MatMulNBits"
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a string quantization_info was accepted")
            except ValueError as exc:
                assert "quantization_info must be a dict" in str(exc)

            state["model"] = quant_model()
            payload["quantization_info"] = {
                "method": "MatMulNBits",
                "has_zero_point": True,
                "packing_format": "packed_uint8",
                "bits": 4,
                "group_size": 32,
            }
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            captured.clear()
            merged = merger.merge_onnx(directory)
            assert captured.get("built") is True
            assert merged.producer_name == "chunking_utils_merged"
            checked = merger.verify_quantization_coherence(state["model"], merged)
            assert checked["coherent"] is True

            payload["quantization_info"] = {"method": None}
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            try:
                merger.merge_onnx(directory)
                raise AssertionError("MatMulNBits survived a dense contract")
            except ValueError as exc:
                assert "quantization coherence failed" in str(exc)
                assert "MatMulNBits" in str(exc)

            payload["quantization_info"] = {
                "method": "MatMulNBits",
                "packing_format": "packed_int32",
                "bits": 4,
                "group_size": 32,
                "has_zero_point": True,
            }
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a packing change was accepted")
            except ValueError as exc:
                assert "quantization coherence failed" in str(exc)
                assert "packing" in str(exc)

            both = quant_model()
            both.graph.node.append(quant_node("DequantizeLinear", ["w", "s", "z"]))
            state["model"] = both
            payload["quantization_info"] = {
                "method": "MatMulNBits",
                "has_zero_point": True,
                "packing_format": "packed_uint8",
                "bits": 4,
                "group_size": 32,
            }
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            try:
                merger.merge_onnx(directory)
                raise AssertionError("a dequantized MatMul was accepted")
            except ValueError as exc:
                assert "quantization coherence failed" in str(exc)
                assert "DequantizeLinear" in str(exc)

            del payload["quantization_info"]
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            state["model"] = quant_model()
            skipped = merger.merge_onnx(directory)
            assert skipped.producer_name == "chunking_utils_merged"
    finally:
        merger.onnx.load = original_load
        merger.helper = original_helper

    print("quantization coherence: OK")
    print("  a dropped scale, a dropped zero point, a packing change, and a dequantized MatMul are not coherent")
    return True


def test_merge_psnr():
    """Merged outputs match the original model within the documented PSNR."""
    print("=" * 50)
    print("Testing merge PSNR...")
    print("=" * 50)

    import inspect
    import json
    import math
    import os
    import sys
    import tempfile
    import types
    import warnings

    import numpy as np

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    import chunking_utils.onnx as onnx_pkg
    from chunking_utils.onnx.types import OnnxMergeResult

    assert list(inspect.signature(merger.merge_onnx).parameters) == [
        "output_dir",
        "save_path",
        "execution_plan",
    ]
    assert list(inspect.signature(merger.merge_and_verify).parameters) == [
        "output_dir",
        "sample_inputs",
        "save_path",
        "min_psnr",
    ]
    assert "merge_and_verify" not in onnx_pkg.__all__
    assert OnnxMergeResult(success=True).psnr is None

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    expect_value_error(
        lambda: merger.merge_and_verify("unused", "x"),
        "a string sample_inputs was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {}),
        "an empty sample_inputs was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {"  ": [1.0]}),
        "a blank sample input name was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {"x": "text"}),
        "a non-numeric sample input was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {"x": np.array([True, False])}),
        "a bool sample input was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {"x": [1.0], " x": [1.0]}),
        "a duplicated sample input name was accepted",
    )
    expect_value_error(
        lambda: merger.merge_and_verify("unused", {"x": [1.0]}, min_psnr=True),
        "a bool min_psnr was accepted",
    )

    import onnxruntime as ort

    class RecordingSession:
        created = []
        next_outputs = [types.SimpleNamespace(name="y")]
        next_values = [np.array([1.0, 2.0], dtype=np.float32)]

        def __init__(self, model, providers=None):
            self.model = model
            self.providers = providers
            self.outputs = RecordingSession.next_outputs
            self.values = RecordingSession.next_values
            RecordingSession.created.append(self)

        def get_outputs(self):
            return self.outputs

        def run(self, names, feeds):
            self.names = names
            self.feeds = feeds
            return self.values

    original_session = ort.InferenceSession
    ort.InferenceSession = RecordingSession
    try:
        feeds = {"x": np.array([1.0, 2.0], dtype=np.float32)}

        class MemoryModel:
            def SerializeToString(self):
                return b"onnx-bytes"

        ran = merger._run_onnx_outputs(MemoryModel(), feeds)
        assert np.array_equal(ran["y"], feeds["x"])
        assert RecordingSession.created[-1].names == ["y"]
        assert RecordingSession.created[-1].feeds is feeds
        assert RecordingSession.created[-1].providers == ["CPUExecutionProvider"]
        assert RecordingSession.created[-1].model == b"onnx-bytes"

        merger._run_onnx_outputs("original.onnx", feeds)
        assert RecordingSession.created[-1].model == "original.onnx"

        class TextModel:
            def SerializeToString(self):
                return "not-bytes"

        expect_value_error(
            lambda: merger._run_onnx_outputs(TextModel(), feeds),
            "a text serialization was accepted",
        )
        expect_value_error(
            lambda: merger._run_onnx_outputs(object(), feeds),
            "a model without serialization was accepted",
        )

        RecordingSession.next_outputs = [types.SimpleNamespace(name="  ")]
        expect_value_error(
            lambda: merger._run_onnx_outputs(MemoryModel(), feeds),
            "a blank output name was accepted",
        )
        RecordingSession.next_outputs = [
            types.SimpleNamespace(name="y"),
            types.SimpleNamespace(name=" y"),
        ]
        expect_value_error(
            lambda: merger._run_onnx_outputs(MemoryModel(), feeds),
            "a duplicated output name was accepted",
        )
        RecordingSession.next_outputs = []
        expect_value_error(
            lambda: merger._run_onnx_outputs(MemoryModel(), feeds),
            "an empty output list was accepted",
        )
        RecordingSession.next_outputs = [types.SimpleNamespace(name="y")]
        RecordingSession.next_values = []
        expect_value_error(
            lambda: merger._run_onnx_outputs(MemoryModel(), feeds),
            "a short runtime result was accepted",
        )
    finally:
        ort.InferenceSession = original_session

    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    removed_runtime = sys.modules.pop("onnxruntime", None)
    builtins.__import__ = blocked_import
    try:
        try:
            merger._run_onnx_outputs("original.onnx", {"x": [1.0]})
            raise AssertionError("a missing onnxruntime was accepted")
        except ValueError as exc:
            assert "onnxruntime is required" in str(exc)
    finally:
        builtins.__import__ = real_import
        if removed_runtime is not None:
            sys.modules["onnxruntime"] = removed_runtime

    def component():
        return types.SimpleNamespace(
            graph=types.SimpleNamespace(
                name="embed_graph",
                node=[types.SimpleNamespace(op_type="Identity", input=["x"], output=["y"])],
                initializer=[],
                input=[],
                output=[],
                value_info=[],
            ),
            opset_import=[],
            ir_version=8,
        )

    calls = []
    script = []
    saved = {}

    def fake_run(model, feeds):
        calls.append((model, feeds))
        if not script:
            raise AssertionError("unexpected runtime call")
        return script.pop(0)

    def fake_load(filepath, load_external_data=True):
        return component()

    def make_graph(nodes, name, inputs, outputs, initializer):
        return types.SimpleNamespace(
            node=list(nodes),
            name=name,
            input=list(inputs),
            output=list(outputs),
            initializer=list(initializer),
            value_info=[],
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(graph=graph, producer_name=producer_name, ir_version=None)

    def fake_save(model, path):
        saved["path"] = path
        saved["model"] = model

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    if not hasattr(merger.onnx, "save"):
        merger.onnx.save = None
    original_load = merger.onnx.load
    original_save = merger.onnx.save
    original_helper = merger.helper
    original_run = merger._run_onnx_outputs
    original_psnr = merger.compute_psnr
    merger.onnx.load = fake_load
    merger.onnx.save = fake_save
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)
    merger._run_onnx_outputs = fake_run
    try:
        with tempfile.TemporaryDirectory() as root:
            chunks = os.path.join(root, "chunks")
            os.makedirs(chunks)
            source = os.path.join(root, "source.onnx")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("original")
            with open(os.path.join(chunks, "embedding.onnx"), "w", encoding="utf-8") as handle:
                handle.write("onnx")
            info_path = os.path.join(chunks, "model_info.json")

            def write_info(original_model="source.onnx"):
                payload = {
                    "num_layers": 0,
                    "components": {"embedding": {"file": "embedding.onnx"}},
                }
                if original_model is not None:
                    payload["original_model"] = original_model
                with open(info_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            reference = [1.0, 1.0]
            ids = np.array([1, 2], dtype=np.int64)
            feeds = {"ids": ids, "x": reference}

            def verify(responses, **kwargs):
                calls.clear()
                script.clear()
                script.extend(responses)
                return merger.merge_and_verify(chunks, feeds, **kwargs)

            write_info()
            exact = verify([{"y": reference}, {"y": list(reference)}])
            assert exact.success is True
            assert math.isinf(exact.psnr)
            assert exact.error is None
            assert exact.output_path is None
            assert exact.num_nodes == 1
            assert exact.num_initializers == 0
            assert calls[0][0] == source
            assert calls[1][0].producer_name == "chunking_utils_merged"
            assert calls[0][1]["ids"] is ids

            close = verify([
                {"y": reference},
                {"y": [1.01, 1.01]},
            ])
            assert close.success is True
            assert math.isclose(close.psnr, 40.0, abs_tol=1e-6)
            assert close.error is None

            production = verify(
                [{"y": reference}, {"y": [1.005, 1.005]}],
                min_psnr=40.0,
            )
            assert production.success is True
            assert production.psnr > 40.0

            delta = 10 ** -1.75
            threshold = verify([
                {"y": reference},
                {"y": [1.0 + delta, 1.0 + delta]},
            ])
            assert threshold.success is True
            assert threshold.psnr >= 35.0

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                failed = verify([
                    {"y": reference},
                    {"y": [1.1, 1.1]},
                ])
            assert failed.success is False
            assert math.isclose(failed.psnr, 20.0, abs_tol=1e-6)
            assert failed.num_nodes == 1
            assert failed.error is not None
            assert "PSNR" in failed.error
            assert "below" in failed.error
            assert "35" in failed.error
            assert any("PSNR verification failed" in str(item.message) for item in caught)

            under = verify(
                [{"y": reference}, {"y": [1.01, 1.01]}],
                min_psnr=41.0,
            )
            assert under.success is False
            assert "41" in under.error

            merger.compute_psnr = lambda reference, candidate, data_range=None: 35.0
            try:
                boundary = verify([{"y": reference}, {"y": reference}])
            finally:
                merger.compute_psnr = original_psnr
            assert boundary.success is True
            assert boundary.psnr == 35.0

            worst = verify([
                {"y": reference, "z": reference},
                {"y": list(reference), "z": [1.1, 1.1]},
            ])
            assert worst.success is False
            assert math.isclose(worst.psnr, 20.0, abs_tol=1e-6)

            try:
                verify([{"y": reference}, {"z": reference}])
                raise AssertionError("disagreeing output names were accepted")
            except ValueError as exc:
                assert "output names disagree" in str(exc)

            try:
                verify([{}, {}])
                raise AssertionError("empty outputs were accepted")
            except ValueError as exc:
                assert "at least one" in str(exc)

            saved.clear()
            destination = os.path.join(root, "merged.onnx")
            stored = verify(
                [{"y": reference}, {"y": list(reference)}],
                save_path=destination,
            )
            assert stored.success is True
            assert stored.output_path == destination
            assert saved["path"] == destination

            unsaved = verify(
                [{"y": reference}, {"y": list(reference)}],
                save_path="",
            )
            assert unsaved.output_path is None

            write_info(original_model=None)
            try:
                verify([{"y": reference}, {"y": reference}])
                raise AssertionError("a missing original model was accepted")
            except ValueError as exc:
                assert "original_model" in str(exc)

            model_dir = os.path.join(chunks, "original.onnx")
            os.makedirs(model_dir)
            write_info("original.onnx")
            try:
                verify([{"y": reference}, {"y": reference}])
                raise AssertionError("a directory original model was accepted")
            except ValueError as exc:
                assert "directory" in str(exc)
            os.rmdir(model_dir)

            write_info("missing.onnx")
            try:
                verify([{"y": reference}, {"y": reference}])
                raise AssertionError("a missing original file was accepted")
            except FileNotFoundError as exc:
                assert "original model not found" in str(exc)

            local = os.path.join(chunks, "local.onnx")
            with open(local, "w", encoding="utf-8") as handle:
                handle.write("local")
            write_info("local.onnx")
            local_result = verify([{"y": reference}, {"y": list(reference)}])
            assert local_result.success is True
            assert calls[0][0] == local

            write_info(os.path.abspath(source))
            absolute = verify([{"y": reference}, {"y": list(reference)}])
            assert absolute.success is True
            assert calls[0][0] == os.path.abspath(source)

            empty = tempfile.TemporaryDirectory()
            try:
                try:
                    merger.merge_and_verify(empty.name, {"x": [1.0, 1.0]})
                    raise AssertionError("a missing model_info was accepted")
                except FileNotFoundError as exc:
                    assert "model_info.json" in str(exc)
            finally:
                empty.cleanup()
    finally:
        merger.onnx.load = original_load
        merger.onnx.save = original_save
        merger.helper = original_helper
        merger._run_onnx_outputs = original_run
        merger.compute_psnr = original_psnr

    print("merge PSNR: OK")
    print("  identical outputs succeed, a small error stays above 35 dB, a large error fails, and a missing original model raises")
    return True


def test_streaming_merge():
    """One component is loaded, appended, and released before the next file."""
    print("=" * 50)
    print("Testing streaming merge...")
    print("=" * 50)

    import inspect
    import json
    import os
    import sys
    import tempfile
    import types
    import weakref

    try:
        import chunking_utils.onnx.merger as merger
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.merger as merger

    import chunking_utils.onnx as onnx_pkg

    assert list(inspect.signature(merger.merge_onnx).parameters) == [
        "output_dir",
        "save_path",
        "execution_plan",
    ]
    assert list(inspect.signature(merger.merge_onnx_streaming).parameters) == [
        "output_dir",
        "save_path",
        "memory_budget_mb",
    ]
    assert (
        inspect.signature(merger.merge_onnx_streaming).parameters["memory_budget_mb"].default
        == 4096
    )
    assert "merge_onnx_streaming" not in onnx_pkg.__all__

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    loads = []
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", "merged.onnx", memory_budget_mb=True),
        "a bool memory budget was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", "merged.onnx", memory_budget_mb=0),
        "a zero memory budget was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", "merged.onnx", memory_budget_mb=-1),
        "a negative memory budget was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", "merged.onnx", memory_budget_mb=4.5),
        "a float memory budget was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", None),
        "a missing save path was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", ""),
        "a blank save path was accepted",
    )
    expect_value_error(
        lambda: merger.merge_onnx_streaming("unused", "   "),
        "a whitespace save path was accepted",
    )
    assert loads == []

    mebibyte = 1024 * 1024
    refs = []
    load_external = []
    saved = {}
    captured = {}

    class Component:
        def __init__(self, weight_name, weight_bytes, output_name):
            self.graph = types.SimpleNamespace(
                name="component_graph",
                node=[types.SimpleNamespace(op_type="Identity", input=["x"], output=[output_name])],
                initializer=[
                    types.SimpleNamespace(name=weight_name, memory_bytes=weight_bytes)
                ],
                input=[],
                output=[types.SimpleNamespace(name=output_name)],
                value_info=[],
            )
            self.opset_import = []
            self.ir_version = 8

    def component(weight_name, weight_bytes, output_name):
        return Component(weight_name, weight_bytes, output_name)

    def fake_load(filepath, load_external_data=True):
        load_external.append(load_external_data)
        if any(ref() is not None for ref in refs):
            raise AssertionError(f"a previous component was still loaded at {filepath}")
        name = os.path.basename(filepath)
        model = factories[name]()
        refs.append(weakref.ref(model))
        return model

    def make_graph(nodes, name, inputs, outputs, initializer):
        captured["nodes"] = list(nodes)
        captured["outputs"] = list(outputs)
        captured["initializers"] = list(initializer)
        return types.SimpleNamespace(
            node=list(nodes),
            name=name,
            input=list(inputs),
            output=list(outputs),
            initializer=list(initializer),
            value_info=[],
        )

    def make_model(graph, producer_name, opset_imports):
        return types.SimpleNamespace(
            graph=graph,
            producer_name=producer_name,
            ir_version=None,
        )

    def fake_save(model, path):
        saved["path"] = path
        saved["model"] = model

    factories = {
        "embedding.onnx": lambda: component("embed_w", 2 * mebibyte, "hidden"),
        "norm.onnx": lambda: component("norm_w", 2 * mebibyte, "norm_out"),
        "lm_head.onnx": lambda: component("head_w", 1, "logits"),
    }

    if not hasattr(merger.onnx, "load"):
        merger.onnx.load = None
    if not hasattr(merger.onnx, "save"):
        merger.onnx.save = None
    original_load = merger.onnx.load
    original_save = merger.onnx.save
    original_helper = merger.helper
    merger.onnx.load = fake_load
    merger.onnx.save = fake_save
    merger.helper = types.SimpleNamespace(make_graph=make_graph, make_model=make_model)
    try:
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "chunks")
            os.makedirs(directory)
            destination = os.path.join(root, "merged.onnx")
            info_path = os.path.join(directory, "model_info.json")

            def write_case(payload, filenames):
                refs.clear()
                load_external.clear()
                saved.clear()
                captured.clear()
                for filename in ("embedding.onnx", "norm.onnx", "lm_head.onnx"):
                    path = os.path.join(directory, filename)
                    if filename in filenames:
                        with open(path, "w", encoding="utf-8") as handle:
                            handle.write("onnx")
                    elif os.path.isfile(path):
                        os.remove(path)
                with open(info_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            payload = {
                "num_layers": 0,
                "components": {
                    "embedding": {"file": "embedding.onnx"},
                    "norm": {"file": "norm.onnx"},
                    "lm_head": {"file": "lm_head.onnx"},
                },
            }
            write_case(payload, ["embedding.onnx", "norm.onnx", "lm_head.onnx"])
            merged = merger.merge_onnx_streaming(directory, destination)
            assert merged.producer_name == "chunking_utils_merged"
            assert saved["path"] == destination
            assert saved["model"] is merged
            assert load_external == [True, True, True]
            assert len(captured["nodes"]) == 3
            assert [item.name for item in captured["initializers"]] == [
                "embed_w",
                "norm_w",
                "head_w",
            ]
            assert [item.name for item in captured["outputs"]] == ["logits"]
            assert all(ref() is None for ref in refs)

            write_case(payload, ["embedding.onnx", "norm.onnx", "lm_head.onnx"])
            factories["embedding.onnx"] = lambda: component("embed_w", 3 * mebibyte, "hidden")
            factories["norm.onnx"] = lambda: component("norm_w", 3 * mebibyte, "norm_out")
            try:
                merger.merge_onnx_streaming(directory, destination, memory_budget_mb=4)
                raise AssertionError("a merge past the memory budget was accepted")
            except ValueError as exc:
                assert "memory budget" in str(exc)
                assert "4" in str(exc)
            assert saved == {}
            assert load_external == [True, True]
            assert all(ref() is None for ref in refs)

            write_case(payload, ["embedding.onnx", "norm.onnx"])
            factories["embedding.onnx"] = lambda: component("shared_w", 3 * mebibyte, "hidden")
            factories["norm.onnx"] = lambda: component("shared_w", 8 * mebibyte, "norm_out")
            duplicate = merger.merge_onnx_streaming(
                directory,
                destination,
                memory_budget_mb=4,
            )
            assert duplicate.producer_name == "chunking_utils_merged"
            assert [item.name for item in captured["initializers"]] == ["shared_w"]
            assert saved["path"] == destination

            write_case(payload, ["embedding.onnx"])
            factories["embedding.onnx"] = lambda: component("embed_w", 4 * mebibyte, "hidden")
            exact = merger.merge_onnx_streaming(directory, destination, memory_budget_mb=4)
            assert exact.producer_name == "chunking_utils_merged"
            assert len(captured["initializers"]) == 1

            try:
                merger.merge_onnx_streaming(root, destination)
                raise AssertionError("a missing model_info was accepted")
            except FileNotFoundError as exc:
                assert "model_info.json" in str(exc)

            with tempfile.TemporaryDirectory() as save_dir:
                expect_value_error(
                    lambda: merger.merge_onnx_streaming(directory, save_dir),
                    "a directory save path was accepted",
                )
    finally:
        merger.onnx.load = original_load
        merger.onnx.save = original_save
        merger.helper = original_helper

    print("streaming merge: OK")
    print("  one component is live at a time, external data is loaded, a small budget raises, and the merged file is saved")
    return True


def test_boundary_tensor_info():
    """Boundary tensors record dtype, shape, cache role, and device on a component."""
    print("=" * 50)
    print("Testing boundary tensor info...")
    print("=" * 50)

    import sys
    import types

    try:
        from chunking_utils.onnx.types import BoundaryTensorInfo, ComponentInfo
        import chunking_utils.onnx as onnx_pkg
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.types import BoundaryTensorInfo, ComponentInfo
        import chunking_utils.onnx as onnx_pkg

    assert "BoundaryTensorInfo" not in onnx_pkg.__all__

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    embedding = ComponentInfo(component_type="embedding", node_indices=[0])
    other = ComponentInfo(component_type="embedding", node_indices=[0])
    assert embedding.name == "embedding"
    assert embedding.num_nodes == 1
    assert embedding.boundary_inputs == []
    assert embedding.boundary_outputs == []
    assert embedding.boundary_inputs is not other.boundary_inputs
    assert embedding.boundary_outputs is not other.boundary_outputs

    caller_shape = [1, None, 8, 64]
    past = BoundaryTensorInfo(
        name=" past_key_values.0.key ",
        dtype=" float16 ",
        shape=caller_shape,
        is_stateful=False,
    )
    assert past.name == "past_key_values.0.key"
    assert past.dtype == "float16"
    assert past.shape == [1, None, 8, 64]
    assert past.shape is not caller_shape
    assert past.is_stateful is True
    assert past.cache_role == "past_key"
    assert past.device_preference == "npu"
    caller_shape.append(1)
    assert past.shape == [1, None, 8, 64]

    hidden = BoundaryTensorInfo(
        name="hidden_states",
        dtype="float16",
        shape=[1, None, 4096],
        device_preference="rdna",
    )
    assert hidden.is_stateful is False
    assert hidden.cache_role is None
    assert hidden.device_preference == "igpu"

    layer = ComponentInfo(
        component_type="layer",
        layer_id=0,
        boundary_inputs=[{
            "name": "past_key_values.0.key",
            "dtype": "float16",
            "shape": [1, None, 8, 64],
        }],
        boundary_outputs=[hidden],
    )
    assert layer.boundary_inputs[0].is_stateful is True
    assert layer.boundary_inputs[0].cache_role == "past_key"
    assert layer.boundary_inputs[0].device_preference == "npu"
    assert layer.boundary_outputs[0] is not hidden
    assert layer.boundary_outputs[0].device_preference == "igpu"
    hidden.shape.append(3)
    assert layer.boundary_outputs[0].shape == [1, None, 4096]

    recorded = layer.to_dict()
    assert recorded["type"] == "layer"
    assert recorded["input_names"] == []
    assert recorded["boundary_inputs"] == [{
        "name": "past_key_values.0.key",
        "dtype": "float16",
        "shape": [1, None, 8, 64],
        "is_stateful": True,
        "cache_role": "past_key",
        "device_preference": "npu",
    }]
    assert recorded["boundary_outputs"][0]["name"] == "hidden_states"
    assert recorded["boundary_outputs"][0]["device_preference"] == "igpu"
    recorded["boundary_inputs"][0]["shape"].append(9)
    assert layer.boundary_inputs[0].shape == [1, None, 8, 64]

    expect_value_error(
        lambda: BoundaryTensorInfo(name=" ", dtype="float16", shape=[1]),
        "a blank boundary name was accepted",
    )
    expect_value_error(
        lambda: BoundaryTensorInfo(name="hidden", dtype="float16", shape=[True]),
        "a bool shape axis was accepted",
    )
    expect_value_error(
        lambda: BoundaryTensorInfo(
            name="hidden",
            dtype="float16",
            shape=[1],
            cache_role="hidden",
        ),
        "an unknown cache role was accepted",
    )
    expect_value_error(
        lambda: BoundaryTensorInfo(
            name="hidden",
            dtype="float16",
            shape=[1],
            device_preference="tpu",
        ),
        "an unknown device was accepted",
    )
    expect_value_error(
        lambda: ComponentInfo(component_type="embedding", boundary_inputs="hidden"),
        "a string boundary list was accepted",
    )
    expect_value_error(
        lambda: ComponentInfo(
            component_type="layer",
            layer_id=0,
            boundary_outputs=[hidden, hidden],
        ),
        "a duplicate boundary name was accepted",
    )

    print("boundary tensor info: OK")
    print("  a past key is stateful on the NPU, a hidden activation keeps its device, two components do not share the boundary lists, and a blank name raises")
    return True


def test_quantization_info():
    """An ONNX split records MatMulNBits, AWQ, or GPTQ quantization."""
    print("=" * 50)
    print("Testing quantization info...")
    print("=" * 50)

    import sys
    import types

    try:
        from chunking_utils.onnx.types import OnnxSplitResult, QuantizationInfo
        import chunking_utils.onnx as onnx_pkg
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.types import OnnxSplitResult, QuantizationInfo
        import chunking_utils.onnx as onnx_pkg

    assert "QuantizationInfo" not in onnx_pkg.__all__

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    bare = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
    )
    assert bare.format.value == "onnx"
    assert bare.quantization is None
    assert bare.to_dict()["quantization"] is None

    shared = QuantizationInfo(
        method="MatMulNBits",
        weight_bits=4,
        weight_dtype="int4",
        activation_dtype="fp16",
        group_size=128,
        has_zero_point=True,
    )
    assert shared.method == "matmul_nbits"
    assert shared.weight_bits == 4
    assert shared.weight_dtype == "int4"
    assert shared.activation_dtype == "float16"
    assert shared.group_size == 128
    assert shared.has_zero_point is True

    first = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
        quantization=shared,
    )
    second = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
        quantization={
            "method": "AWQ",
            "weight_bits": 4,
            "weight_dtype": "int4",
            "activation_dtype": "float16",
            "group_size": 128,
            "has_zero_point": False,
        },
    )
    assert first.quantization is not shared
    assert first.quantization is not second.quantization
    assert second.quantization.method == "awq"
    assert second.quantization.has_zero_point is False
    first.quantization.group_size = 32
    assert shared.group_size == 128
    assert second.quantization.group_size == 128

    recorded = first.to_dict()["quantization"]
    assert recorded["method"] == "matmul_nbits"
    assert recorded["weight_bits"] == 4
    assert recorded["has_zero_point"] is True
    recorded["group_size"] = 0
    assert first.quantization.group_size == 32

    expect_value_error(
        lambda: QuantizationInfo(method="smoothquant"),
        "an unknown quantization method was accepted",
    )
    expect_value_error(
        lambda: QuantizationInfo(weight_bits=True),
        "a bool weight bit count was accepted",
    )
    expect_value_error(
        lambda: QuantizationInfo(weight_dtype="int4", weight_bits=16),
        "a bit count that disagrees with the weight dtype was accepted",
    )
    expect_value_error(
        lambda: QuantizationInfo(group_size=0),
        "a zero group size was accepted",
    )
    expect_value_error(
        lambda: QuantizationInfo(group_size=True),
        "a bool group size was accepted",
    )
    expect_value_error(
        lambda: QuantizationInfo(has_zero_point=1),
        "an integer zero point flag was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            quantization="MatMulNBits",
        ),
        "a string quantization record was accepted",
    )

    print("quantization info: OK")
    print("  MatMulNBits stores int4 weights and a zero point, two split results do not share the record, and a bool bit count raises")
    return True


def test_kv_cache_contract():
    """An ONNX split records past and present key and value patterns."""
    print("=" * 50)
    print("Testing kv cache contract...")
    print("=" * 50)

    import sys
    import types

    try:
        from chunking_utils.onnx.types import KVCacheContract, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.types import KVCacheContract, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg

    assert "KVCacheContract" not in onnx_pkg.__all__

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    bare = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
    )
    assert bare.kv_cache is None
    assert bare.to_dict()["kv_cache"] is None

    shared = KVCacheContract(
        num_layers=2,
        num_kv_heads=8,
        head_dim=64,
        dtype="fp16",
        layout="bnsd",
        max_seq_len=4096,
        past_key_pattern=" past_key_values.{layer}.key ",
        past_value_pattern="past_key_values.{layer}.value",
        present_key_pattern="present.{layer}.key",
        present_value_pattern="present.{layer}.value",
    )
    assert shared.dtype == "float16"
    assert shared.layout == "BNSD"
    assert shared.past_key_pattern == "past_key_values.{layer}.key"
    assert "{layer}" in shared.past_key_pattern
    assert shared.present_value_pattern == "present.{layer}.value"

    first = OnnxSplitResult(
        output_dir="out",
        num_layers=2,
        component_files={},
        model_info={},
        kv_cache=shared,
    )
    second = OnnxSplitResult(
        output_dir="out",
        num_layers=2,
        component_files={},
        model_info={},
        kv_cache={
            "num_layers": 2,
            "num_kv_heads": 8,
            "head_dim": 64,
            "dtype": "float16",
            "layout": "BNSD",
            "max_seq_len": 2048,
            "past_key_pattern": "past_key_values.{layer}.key",
            "past_value_pattern": "past_key_values.{layer}.value",
            "present_key_pattern": "present.{layer}.key",
            "present_value_pattern": "present.{layer}.value",
        },
    )
    assert first.kv_cache is not shared
    assert first.kv_cache is not second.kv_cache
    assert second.kv_cache.max_seq_len == 2048
    first.kv_cache.max_seq_len = 1024
    assert shared.max_seq_len == 4096
    assert second.kv_cache.max_seq_len == 2048

    recorded = first.to_dict()["kv_cache"]
    assert recorded["num_kv_heads"] == 8
    assert recorded["layout"] == "BNSD"
    assert recorded["past_key_pattern"] == "past_key_values.{layer}.key"
    recorded["head_dim"] = 0
    assert first.kv_cache.head_dim == 64

    expect_value_error(
        lambda: KVCacheContract(
            num_layers=2,
            num_kv_heads=True,
            head_dim=64,
            dtype="float16",
            layout="BNSD",
            max_seq_len=4096,
            past_key_pattern="past_key_values.{layer}.key",
            past_value_pattern="past_key_values.{layer}.value",
            present_key_pattern="present.{layer}.key",
            present_value_pattern="present.{layer}.value",
        ),
        "a bool head count was accepted",
    )
    expect_value_error(
        lambda: KVCacheContract(
            num_layers=2,
            num_kv_heads=8,
            head_dim=0,
            dtype="float16",
            layout="BNSD",
            max_seq_len=4096,
            past_key_pattern="past_key_values.{layer}.key",
            past_value_pattern="past_key_values.{layer}.value",
            present_key_pattern="present.{layer}.key",
            present_value_pattern="present.{layer}.value",
        ),
        "a zero head dim was accepted",
    )
    expect_value_error(
        lambda: KVCacheContract(
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
            layout="BNSD",
            max_seq_len=4096,
            past_key_pattern=" ",
            past_value_pattern="past_key_values.{layer}.value",
            present_key_pattern="present.{layer}.key",
            present_value_pattern="present.{layer}.value",
        ),
        "a blank pattern was accepted",
    )
    expect_value_error(
        lambda: KVCacheContract(
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
            layout="BNSD",
            max_seq_len=4096,
            past_key_pattern="past_key_values.0.key",
            past_value_pattern="past_key_values.{layer}.value",
            present_key_pattern="present.{layer}.key",
            present_value_pattern="present.{layer}.value",
        ),
        "a pattern without a layer placeholder was accepted",
    )
    expect_value_error(
        lambda: KVCacheContract(
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
            layout="BNSD",
            max_seq_len=4096,
            past_key_pattern="past_key_values.{layer}.value",
            past_value_pattern="past_key_values.{layer}.value",
            present_key_pattern="present.{layer}.key",
            present_value_pattern="present.{layer}.value",
        ),
        "a value pattern in the key slot was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=2,
            component_files={},
            model_info={},
            kv_cache="past_key_values.{layer}.key",
        ),
        "a string kv cache record was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            kv_cache=shared,
        ),
        "a layer count mismatch was accepted",
    )

    print("kv cache contract: OK")
    print("  a past key pattern keeps the layer placeholder, BNSD stays the layout, two split results do not share the record, and a bool head count raises")
    return True


def test_execution_plan_type():
    """An ONNX placement plan names the device, phase, and residency of each component."""
    print("=" * 50)
    print("Testing execution plan type...")
    print("=" * 50)

    import sys
    import types

    try:
        from chunking_utils.onnx.types import ExecutionPlan, ExecutionStep, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils
        from chunking_utils.common.types import ExecutionPlan as CommonPlan
        from chunking_utils.common.types import ExecutionPolicy
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.types import ExecutionPlan, ExecutionStep, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils
        from chunking_utils.common.types import ExecutionPlan as CommonPlan
        from chunking_utils.common.types import ExecutionPolicy

    assert "ExecutionPlan" not in onnx_pkg.__all__
    assert "ExecutionStep" not in onnx_pkg.__all__
    assert chunking_utils.ExecutionPlan is CommonPlan
    assert ExecutionPlan is not CommonPlan

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    bare = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
    )
    assert type(bare.execution_plan) is CommonPlan
    assert bare.execution_plan.steps == []
    assert bare.execution_plan.policy is ExecutionPolicy.AUTO

    shared_step = ExecutionStep(
        component_name=" layer_0 ",
        provider=" VitisAIExecutionProvider ",
        device="xdna",
        phase=" Decode ",
    )
    assert shared_step.component_name == "layer_0"
    assert shared_step.provider == "VitisAIExecutionProvider"
    assert shared_step.device == "npu"
    assert shared_step.phase == "decode"
    assert shared_step.resident is True

    first = ExecutionPlan(
        steps=[shared_step],
        policy=ExecutionPolicy.HYBRID_NPU_IGPU,
        npu_safe=True,
        prefill_device="xdna",
        decode_device="rdna",
    )
    second = ExecutionPlan(
        steps=[{
            "component_name": "expert_0",
            "device": "igpu",
            "phase": "moe_router",
            "resident": False,
        }],
        policy="auto",
        npu_safe=False,
        prefill_device="npu",
        decode_device="igpu",
    )
    assert first.policy == "hybrid_npu_igpu"
    assert first.npu_safe is True
    assert first.prefill_device == "npu"
    assert first.decode_device == "igpu"
    assert first.steps[0] is not shared_step
    assert first.steps[0].phase == "decode"
    assert second.steps[0].phase == "router"
    assert second.steps[0].resident is False
    assert second.policy == "auto"
    first.steps[0].resident = False
    assert shared_step.resident is True
    assert second.steps[0].resident is False

    recorded = first.to_dict()
    assert recorded["policy"] == "hybrid_npu_igpu"
    assert recorded["npu_safe"] is True
    assert recorded["steps"][0]["device"] == "npu"
    assert recorded["steps"][0]["phase"] == "decode"
    recorded["steps"][0]["resident"] = True
    recorded["steps"].append({"component_name": "norm"})
    assert first.steps[0].resident is False
    assert len(first.steps) == 1

    expect_value_error(
        lambda: ExecutionStep(component_name="layer_0", resident=1),
        "an integer resident flag was accepted",
    )
    expect_value_error(
        lambda: ExecutionStep(component_name=" "),
        "a blank component name was accepted",
    )
    expect_value_error(
        lambda: ExecutionStep(component_name="layer_0", phase="norm"),
        "an unknown phase was accepted",
    )
    expect_value_error(
        lambda: ExecutionPlan(steps="layer_0"),
        "a string step list was accepted",
    )
    expect_value_error(
        lambda: ExecutionPlan(steps=[shared_step, shared_step]),
        "a repeated component was accepted",
    )
    expect_value_error(
        lambda: ExecutionPlan(steps=[{
            "component_name": "layer_0",
            "depends_on": ["embedding"],
        }]),
        "a dependency field was accepted",
    )
    expect_value_error(
        lambda: ExecutionPlan(npu_safe=1),
        "a bool npu flag was accepted",
    )
    expect_value_error(
        lambda: ExecutionPlan(policy="hybrid_phase_split"),
        "an unknown policy was accepted",
    )

    print("execution plan type: OK")
    print("  decode stays the phase, xdna stays the npu, two plans do not share a step, and an integer resident flag raises")
    return True


def test_schema_version():
    """An ONNX split records the metadata generation for model_info.json."""
    print("=" * 50)
    print("Testing schema version...")
    print("=" * 50)

    import sys
    import types

    try:
        from chunking_utils.onnx.types import SCHEMA_VERSION, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        from chunking_utils.onnx.types import SCHEMA_VERSION, OnnxSplitResult
        import chunking_utils.onnx as onnx_pkg

    assert "SCHEMA_VERSION" not in onnx_pkg.__all__
    assert SCHEMA_VERSION == "2.0.0"

    def expect_value_error(callable_check, text):
        try:
            callable_check()
            raise AssertionError(text)
        except ValueError:
            return

    bare = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
    )
    assert bare.schema_version == "2.0.0"
    assert bare.to_dict()["schema_version"] == "2.0.0"

    pinned = OnnxSplitResult(
        output_dir="out",
        num_layers=1,
        component_files={},
        model_info={},
        schema_version=" 2.0.0 ",
    )
    assert pinned.schema_version == "2.0.0"
    recorded = pinned.to_dict()
    recorded["schema_version"] = "1.0.0"
    assert pinned.schema_version == "2.0.0"

    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            schema_version=" ",
        ),
        "a blank version was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            schema_version=True,
        ),
        "a bool version was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            schema_version="1.0.0",
        ),
        "a different version was accepted",
    )
    expect_value_error(
        lambda: OnnxSplitResult(
            output_dir="out",
            num_layers=1,
            component_files={},
            model_info={},
            schema_version="2.0",
        ),
        "a short version was accepted",
    )

    print("schema version: OK")
    print("  2.0.0 stays the schema version, a copied record does not change the field, and a blank version raises")
    return True


def test_external_data_fallback():
    """A component graph can read a shared master data file."""
    print("=" * 50)
    print("Testing external data fallback...")
    print("=" * 50)

    import inspect
    import os
    import sys
    import tempfile
    import types

    try:
        import chunking_utils.onnx.loader as loader
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx.loader as loader

    signature = inspect.signature(loader.load_onnx_model)
    assert list(signature.parameters)[:2] == ["model_path", "external_data_dir"]
    assert signature.parameters["external_data_dir"].default is None

    class _Entry:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _Tensor:
        def __init__(self, name, location, offset, length):
            self.name = name
            self.external_data = [
                _Entry("location", location),
                _Entry("offset", offset),
                _Entry("length", length),
            ]
            self.raw_data = b""

    class _Model:
        def __init__(self, tensors):
            self.graph = types.SimpleNamespace(
                initializer=list(tensors),
                sparse_initializer=[],
                node=[],
            )
            self.ir_version = 8
            self.producer_name = "test"

    calls = []

    def fake_load(path, load_external_data=True):
        calls.append((path, load_external_data))
        if load_external_data and not os.path.isfile(path + ".data"):
            raise FileNotFoundError(path + ".data")
        model = _Model([
            _Tensor("kept", "note.onnx.data", "0", "4"),
            _Tensor("layer", "layer_0.onnx.data", "4", "7"),
        ])
        if load_external_data:
            with open(path + ".data", "rb") as handle:
                model.graph.initializer[1].raw_data = handle.read()
        return model

    def expect(callable_check, error_type, text):
        try:
            callable_check()
            raise AssertionError(text)
        except error_type:
            return

    had_load = hasattr(loader.onnx, "load")
    original_load = getattr(loader.onnx, "load", None)
    loader.onnx.load = fake_load
    try:
        with tempfile.TemporaryDirectory() as root:
            split_dir = os.path.join(root, "split")
            os.makedirs(split_dir)
            component = os.path.join(split_dir, "layer_0.onnx")
            with open(component, "wb") as handle:
                handle.write(b"graph")
            with open(os.path.join(split_dir, "note.onnx.data"), "wb") as handle:
                handle.write(b"NOTE")
            with open(os.path.join(split_dir, "model.onnx.data"), "wb") as handle:
                handle.write(b"HEADWEIGHTSTAIL")

            calls.clear()
            loaded = loader.load_onnx_model(component, external_data_dir=split_dir)
            kept, layer = loaded.graph.initializer
            assert calls == [(component, False)]
            assert kept.raw_data == b"NOTE"
            assert kept.external_data[0].value == "note.onnx.data"
            assert layer.raw_data == b"WEIGHTS"
            assert layer.external_data[0].value == "model.onnx.data"

            calls.clear()
            again = loader.load_onnx_model(component, external_data_dir="  " + split_dir + "  ")
            assert again.graph.initializer[0] is not loaded.graph.initializer[0]
            assert again.graph.initializer[1].raw_data == b"WEIGHTS"
            again.graph.initializer[1].raw_data = b"CHANGED"
            assert loaded.graph.initializer[1].raw_data == b"WEIGHTS"

            master_dir = os.path.join(root, "master")
            os.makedirs(master_dir)
            master_file = os.path.join(master_dir, "model.onnx.data")
            with open(master_file, "wb") as handle:
                handle.write(b"HEADWEIGHTSTAIL")
            os.remove(os.path.join(split_dir, "model.onnx.data"))
            direct = loader.load_onnx_model(component, external_data_dir=master_file)
            assert direct.graph.initializer[1].raw_data == b"WEIGHTS"
            assert direct.graph.initializer[1].external_data[0].value == os.path.abspath(master_file)

            single_dir = os.path.join(root, "single")
            os.makedirs(single_dir)
            with open(os.path.join(single_dir, "weights.onnx.data"), "wb") as handle:
                handle.write(b"HEADWEIGHTSTAIL")
            single = loader.load_onnx_model(component, external_data_dir=single_dir)
            assert single.graph.initializer[1].raw_data == b"WEIGHTS"
            assert single.graph.initializer[1].external_data[0].value == os.path.abspath(
                os.path.join(single_dir, "weights.onnx.data")
            )

            sibling = os.path.join(split_dir, "layer_0.onnx.data")
            with open(sibling, "wb") as handle:
                handle.write(b"SIBLING-DATA")
            calls.clear()
            owned = loader.load_onnx_model(component, external_data_dir=master_file)
            assert calls == [(component, True)]
            assert owned.graph.initializer[1].external_data[0].value == "layer_0.onnx.data"
            assert owned.graph.initializer[1].raw_data == b"SIBLING-DATA"

            os.remove(sibling)
            calls.clear()
            graph_only = loader.load_onnx_model(
                component,
                external_data_dir=master_file,
                load_external_data=False,
            )
            assert calls == [(component, False)]
            assert graph_only.graph.initializer[1].raw_data == b""
            assert graph_only.graph.initializer[1].external_data[0].value == "layer_0.onnx.data"

            calls.clear()
            lazy = loader.load_onnx_model(
                component,
                external_data_dir=master_file,
                lazy_load_threshold_mb=0,
            )
            assert calls == [(component, False)]
            assert lazy.graph.initializer[1].raw_data == b""
            assert lazy.graph.initializer[1].external_data[0].value == "layer_0.onnx.data"

            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=" "),
                ValueError,
                "a blank path was accepted",
            )
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=True),
                ValueError,
                "a bool path was accepted",
            )
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=os.path.join(root, "missing")),
                FileNotFoundError,
                "a missing directory was accepted",
            )
            empty_dir = os.path.join(root, "empty")
            os.makedirs(empty_dir)
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=empty_dir),
                FileNotFoundError,
                "an empty directory was accepted",
            )
            plain = os.path.join(root, "notes.txt")
            with open(plain, "wb") as handle:
                handle.write(b"text")
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=plain),
                ValueError,
                "a non-data file was accepted",
            )
            mixed = os.path.join(root, "mixed")
            os.makedirs(mixed)
            for name in ("a.data", "b.data"):
                with open(os.path.join(mixed, name), "wb") as handle:
                    handle.write(b"HEADWEIGHTSTAIL")
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=mixed),
                ValueError,
                "several data files were accepted",
            )

            def broken_offset(path, load_external_data=True):
                return _Model([_Tensor("layer", "layer_0.onnx.data", True, "4")])

            loader.onnx.load = broken_offset
            expect(
                lambda: loader.load_onnx_model(component, external_data_dir=master_file),
                ValueError,
                "a bool offset was accepted",
            )
    finally:
        if had_load:
            loader.onnx.load = original_load
        elif hasattr(loader.onnx, "load"):
            del loader.onnx.load

    print("external data fallback: OK")
    print("  a component reads the master data file, the component data file wins when it exists, and a blank path raises")
    return True


def test_genai_external_data():
    """A GenAI model keeps every numbered external data file."""
    print("=" * 50)
    print("Testing GenAI external data files...")
    print("=" * 50)

    import os
    import sys
    import tempfile
    import types

    try:
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader

    assert "find_all_external_data_files" not in onnx_pkg.__all__

    class _Entry:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _Tensor:
        def __init__(self, name, location, offset, length):
            self.name = name
            self.external_data = [
                _Entry("location", location),
                _Entry("offset", offset),
                _Entry("length", length),
            ]
            self.raw_data = b""

    class _Model:
        def __init__(self, tensors):
            self.graph = types.SimpleNamespace(
                initializer=list(tensors),
                sparse_initializer=[],
                node=[],
            )
            self.ir_version = 8
            self.producer_name = "genai"

    calls = []

    def fake_load(path, load_external_data=True):
        calls.append((path, load_external_data))
        if load_external_data and not os.path.isfile(path + ".data"):
            raise FileNotFoundError(path + ".data")
        return _Model([
            _Tensor("a", "model.onnx.data", "0", "4"),
            _Tensor("b", "model.onnx.data_1", "0", "4"),
            _Tensor("c", "model.onnx.data_2", "1", "3"),
            _Tensor("d", "model.onnx_1.data", "0", "4"),
        ])

    def expect(callable_check, error_type, text):
        try:
            callable_check()
            raise AssertionError(text)
        except error_type:
            return

    had_load = hasattr(loader.onnx, "load")
    original_load = getattr(loader.onnx, "load", None)
    loader.onnx.load = fake_load
    try:
        with tempfile.TemporaryDirectory() as root:
            model = os.path.join(root, "model.onnx")
            with open(model, "wb") as handle:
                handle.write(b"graph")
            payloads = {
                "model.onnx.data": b"AAAA",
                "model.onnx.data_2": b"XCCCC",
                "model.onnx.data_1": b"BBBB",
                "model.onnx.data_10": b"TEN!",
                "model.onnx_2.data": b"EEEE",
                "model.onnx_1.data": b"DDDD",
                "model.onnx.data_0": b"ZERO",
                "model.onnx.data_01": b"LEAD",
                "other.onnx.data": b"OTHER",
                "model.onnx.data.bak": b"BAK",
            }
            for name, payload in payloads.items():
                with open(os.path.join(root, name), "wb") as handle:
                    handle.write(payload)

            found = loader.find_all_external_data_files("  " + model + "  ")
            assert found == [
                os.path.normpath(os.path.join(root, "model.onnx.data")),
                os.path.normpath(os.path.join(root, "model.onnx.data_1")),
                os.path.normpath(os.path.join(root, "model.onnx.data_2")),
                os.path.normpath(os.path.join(root, "model.onnx.data_10")),
                os.path.normpath(os.path.join(root, "model.onnx_1.data")),
                os.path.normpath(os.path.join(root, "model.onnx_2.data")),
            ]
            found.append("extra")
            again = loader.find_all_external_data_files(model)
            assert again == found[:-1]
            assert again is not found

            calls.clear()
            loaded = loader.load_onnx_model(model)
            assert calls == [(model, False)]
            raw = [tensor.raw_data for tensor in loaded.graph.initializer]
            assert raw == [b"AAAA", b"BBBB", b"CCC", b"DDDD"]
            assert loaded.graph.initializer[1].external_data[0].value == "model.onnx.data_1"
            loaded.graph.initializer[1].raw_data = b"CHANGED"
            second = loader.load_onnx_model(model)
            assert second.graph.initializer[1] is not loaded.graph.initializer[1]
            assert second.graph.initializer[1].raw_data == b"BBBB"

            os.remove(os.path.join(root, "model.onnx.data"))
            with open(os.path.join(root, "model.onnx.data"), "wb") as handle:
                handle.write(b"")
            calls.clear()
            lazy = loader.load_onnx_model(model, lazy_load_threshold_mb=0)
            assert calls == [(model, False)]
            assert lazy.graph.initializer[1].raw_data == b""
            assert lazy.graph.initializer[1].external_data[0].value == "model.onnx.data_1"

            for name in (
                "model.onnx.data",
                "model.onnx.data_2",
                "model.onnx.data_10",
                "model.onnx_1.data",
                "model.onnx_2.data",
            ):
                os.remove(os.path.join(root, name))

            def data_1_only(path, load_external_data=True):
                calls.append((path, load_external_data))
                return _Model([_Tensor("b", "model.onnx.data_1", "0", "4")])

            loader.onnx.load = data_1_only
            calls.clear()
            lone = loader.load_onnx_model(model)
            assert calls == [(model, False)]
            assert lone.graph.initializer[0].raw_data == b"BBBB"
            assert lone.graph.initializer[0].external_data[0].value == "model.onnx.data_1"

            def missing_shard(path, load_external_data=True):
                calls.append((path, load_external_data))
                return _Model([_Tensor("gone", "model.onnx.data_3", "0", "4")])

            loader.onnx.load = missing_shard
            expect(
                lambda: loader.load_onnx_model(model),
                FileNotFoundError,
                "a missing data_3 shard was accepted",
            )

            shards = os.path.join(root, "shards")
            os.makedirs(shards)
            with open(os.path.join(shards, "model.onnx.data"), "wb") as handle:
                handle.write(b"AAAA")
            with open(os.path.join(shards, "model.onnx.data_1"), "wb") as handle:
                handle.write(b"BBBB")
            elsewhere = os.path.join(root, "elsewhere")
            os.makedirs(elsewhere)
            remote = os.path.join(elsewhere, "model.onnx")
            with open(remote, "wb") as handle:
                handle.write(b"graph")

            def remote_model(path, load_external_data=True):
                calls.append((path, load_external_data))
                return _Model([
                    _Tensor("a", "model.onnx.data", "0", "4"),
                    _Tensor("b", "model.onnx.data_1", "0", "4"),
                ])

            loader.onnx.load = remote_model
            calls.clear()
            moved = loader.load_onnx_model(remote, external_data_dir=shards)
            assert calls == [(remote, False)]
            assert [tensor.raw_data for tensor in moved.graph.initializer] == [b"AAAA", b"BBBB"]
            assert moved.graph.initializer[1].external_data[0].value == "model.onnx.data_1"

            expect(
                lambda: loader.find_all_external_data_files(" "),
                ValueError,
                "a blank path was accepted",
            )
            expect(
                lambda: loader.find_all_external_data_files(True),
                ValueError,
                "a bool path was accepted",
            )
            expect(
                lambda: loader.find_all_external_data_files(root),
                ValueError,
                "a directory was accepted",
            )
            expect(
                lambda: loader.find_all_external_data_files(os.path.join(root, "missing", "model.onnx")),
                FileNotFoundError,
                "a missing directory was accepted",
            )
    finally:
        if had_load:
            loader.onnx.load = original_load
        elif hasattr(loader.onnx, "load"):
            del loader.onnx.load

    print("genai external data: OK")
    print("  a genai model reads data_1, a numbered file stays in order, and a blank path raises")
    return True


def test_lazy_shape_inference():
    """A lazy-loaded model warns when shape inference cannot see weight values."""
    print("=" * 50)
    print("Testing lazy shape inference...")
    print("=" * 50)

    import io
    import os
    import sys
    import tempfile
    import types
    from contextlib import redirect_stdout

    try:
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader
        import chunking_utils.onnx.splitter as splitter
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader
        import chunking_utils.onnx.splitter as splitter

    assert "infer_shapes_with_lazy_fallback" not in onnx_pkg.__all__
    assert splitter.infer_shapes_with_lazy_fallback is loader.infer_shapes_with_lazy_fallback

    class _Entry:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _Dim:
        def __init__(self, value=0, param=""):
            self.dim_value = value
            self.dim_param = param

    class _Value:
        def __init__(self, name, dims):
            self.name = name
            self.type = types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(
                    shape=types.SimpleNamespace(dim=list(dims)),
                )
            )

    class _Init:
        def __init__(self, name, location, raw):
            self.name = name
            self.external_data = [_Entry("location", location)] if location else []
            self.raw_data = raw

    class _Node:
        def __init__(self, op_type, inputs, outputs):
            self.op_type = op_type
            self.input = inputs
            self.output = outputs

    class _Model:
        def __init__(self, nodes, initializers, value_info):
            self.graph = types.SimpleNamespace(
                node=list(nodes),
                initializer=list(initializers),
                value_info=list(value_info),
                output=[],
                sparse_initializer=[],
            )
            self.ir_version = 8
            self.producer_name = "lazy"

    def reshape_model(raw, dims, lazy_flag):
        model = _Model(
            [_Node("Reshape", ["x", "shape"], ["y"])],
            [_Init("shape", "model.onnx.data", raw)],
            [_Value("y", dims)],
        )
        model._chunking_lazy_loaded = lazy_flag
        return model

    def identity(model):
        return model

    def boom(model):
        raise RuntimeError("missing weight")

    def capture(model, infer):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = loader.infer_shapes_with_lazy_fallback(model, infer_shapes=infer)
        return result, buffer.getvalue()

    def expect(callable_check, error_type, text):
        try:
            callable_check()
            raise AssertionError(text)
        except error_type:
            return

    calls = []

    def fake_load(path, load_external_data=True):
        calls.append((path, load_external_data))
        return _Model([], [], [])

    had_load = hasattr(loader.onnx, "load")
    original_load = getattr(loader.onnx, "load", None)
    loader.onnx.load = fake_load
    try:
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "model.onnx")
            with open(model_path, "wb") as handle:
                handle.write(b"graph")
            with open(model_path + ".data", "wb") as handle:
                handle.write(b"AAAA")

            calls.clear()
            lazy = loader.load_onnx_model(model_path, lazy_load_threshold_mb=0)
            assert calls == [(model_path, False)]
            assert lazy._chunking_lazy_loaded is True

            calls.clear()
            eager = loader.load_onnx_model(model_path, lazy_load_threshold_mb=10000)
            assert calls == [(model_path, True)]
            assert eager._chunking_lazy_loaded is False

        lazy_dynamic = reshape_model(b"", [_Dim(0, "unk")], True)
        kept, text = capture(lazy_dynamic, boom)
        assert kept is lazy_dynamic
        assert "lazy-loaded" in text
        assert "Falling back" in text
        assert "Static NPU buffers" in text

        inferred, text = capture(lazy_dynamic, identity)
        assert inferred is lazy_dynamic
        assert "dynamic shapes" in text
        assert "Shape inference successful." not in text

        concrete, text = capture(reshape_model(b"", [_Dim(8, "")], True), identity)
        assert concrete._chunking_lazy_loaded is True
        assert "Shape inference successful." in text
        assert "dynamic shapes" not in text

        resident, text = capture(
            reshape_model(b"\x08\x00\x00\x00", [_Dim(0, "batch")], True),
            identity,
        )
        assert resident is not None
        assert "dynamic shapes" not in text
        assert "Shape inference successful." in text

        constant = _Model(
            [_Node("ConstantOfShape", ["shape"], ["y"])],
            [_Init("shape", "model.onnx.data", b"")],
            [_Value("y", [_Dim(0, "unk")])],
        )
        constant._chunking_lazy_loaded = True
        _, text = capture(constant, identity)
        assert "dynamic shapes" in text

        scalar = _Model(
            [_Node("ConstantOfShape", ["shape"], ["y"])],
            [_Init("shape", "model.onnx.data", b"")],
            [_Value("y", [])],
        )
        scalar._chunking_lazy_loaded = True
        _, text = capture(scalar, identity)
        assert "dynamic shapes" not in text
        assert "Shape inference successful." in text

        bool_dim = reshape_model(b"", [_Dim(True, "")], True)
        _, text = capture(bool_dim, identity)
        assert "dynamic shapes" in text

        plain = reshape_model(b"", [_Dim(0, "unk")], False)
        kept_plain, text = capture(plain, boom)
        assert kept_plain is plain
        assert "Subgraphs may have unknown shapes." in text
        assert "lazy-loaded" not in text

        def cloned(model):
            return _Model(model.graph.node, model.graph.initializer, model.graph.value_info)

        source = reshape_model(b"\x01", [_Dim(4, "")], True)
        copied, text = capture(source, cloned)
        assert copied is not source
        assert copied._chunking_lazy_loaded is True
        assert "Shape inference successful." in text

        def none_infer(model):
            return None

        none_kept, text = capture(source, none_infer)
        assert none_kept is source
        assert "no model returned" in text
        assert "Falling back" in text

        expect(
            lambda: loader.infer_shapes_with_lazy_fallback(True),
            ValueError,
            "a bool model was accepted",
        )
        expect(
            lambda: loader.infer_shapes_with_lazy_fallback(" "),
            ValueError,
            "a blank model was accepted",
        )
        expect(
            lambda: loader.infer_shapes_with_lazy_fallback(None),
            ValueError,
            "a missing model was accepted",
        )
        expect(
            lambda: loader.infer_shapes_with_lazy_fallback(source, infer_shapes=True),
            ValueError,
            "a bool infer_shapes was accepted",
        )
        expect(
            lambda: loader.infer_shapes_with_lazy_fallback(root),
            ValueError,
            "a directory path was accepted",
        )
    finally:
        if had_load:
            loader.onnx.load = original_load
        elif hasattr(loader.onnx, "load"):
            del loader.onnx.load

    print("lazy shape inference: OK")
    print("  a failed lazy inference keeps the graph, a dynamic Reshape warns, and a bool model raises")
    return True


def test_onnx_model_validation():
    """A lazy model skips tensor checks, and provider kernels must exist."""
    print("=" * 50)
    print("Testing ONNX model validation...")
    print("=" * 50)

    import io
    import os
    import sys
    import tempfile
    import types
    from contextlib import redirect_stdout

    try:
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx as onnx_pkg
        import chunking_utils.onnx.loader as loader

    assert "validate_onnx_model" not in onnx_pkg.__all__

    class _Node:
        def __init__(self, op_type, domain=""):
            self.op_type = op_type
            self.domain = domain

    class _Named:
        def __init__(self, name):
            self.name = name

    class _Tensor:
        def __init__(self, name, location):
            self.name = name
            self.raw_data = b""
            self.external_data = [types.SimpleNamespace(key="location", value=location)]

    class _Model:
        def __init__(self, nodes, initializers=None, inputs=None, outputs=None, lazy=False):
            self.graph = types.SimpleNamespace(
                node=list(nodes),
                initializer=list(initializers or []),
                sparse_initializer=[],
                input=list(inputs if inputs is not None else [_Named("x")]),
                output=list(outputs if outputs is not None else [_Named("y")]),
            )
            self.ir_version = 8
            self.producer_name = "validate"
            self._chunking_lazy_loaded = lazy

    checker_calls = []

    def check_model(model):
        checker_calls.append(model)

    def boom(model):
        raise RuntimeError("tensor data is empty")

    def expect(callable_check, error_type, text):
        try:
            callable_check()
            raise AssertionError(text)
        except error_type:
            return

    def capture(path, **kwargs):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = loader.validate_onnx_model(path, **kwargs)
        return result, buffer.getvalue()

    models = {}

    def fake_load(path, **kwargs):
        key = os.path.normpath(path)
        if os.path.isdir(key) or key not in models:
            raise FileNotFoundError(key)
        return models[key]

    def injected_runtime():
        return (
            ["CPUExecutionProvider", "VitisAIExecutionProvider"],
            [
                ("", "MatMul", "CPUExecutionProvider"),
                ("com.microsoft", "MatMulNBits", "CPUExecutionProvider"),
            ],
        )

    original_load = loader.load_onnx_model
    original_runtime = loader._installed_runtime
    had_checker = hasattr(loader.onnx, "checker")
    original_checker = getattr(loader.onnx, "checker", None)
    loader.load_onnx_model = fake_load
    loader._installed_runtime = injected_runtime
    loader.onnx.checker = types.SimpleNamespace(check_model=check_model)
    try:
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "model.onnx")
            with open(model_path, "wb") as handle:
                handle.write(b"graph")
            empty = os.path.join(root, "empty")
            os.makedirs(empty)
            key = os.path.normpath(model_path)

            models[key] = _Model([_Node("MatMulNBits", "com.microsoft")], lazy=True)
            checker_calls.clear()
            ok, text = capture(model_path)
            assert ok is True
            assert checker_calls == []
            assert "skipped tensor-data checks" in text
            assert "Model validation successful." in text

            models[key] = _Model([_Node("  ")], lazy=True)
            checker_calls.clear()
            failed, text = capture(model_path)
            assert failed is False
            assert checker_calls == []
            assert "blank op_type" in text

            models[key] = _Model(
                [_Node("MatMul")],
                initializers=[_Tensor("w", "  ")],
                lazy=True,
            )
            failed, text = capture(model_path)
            assert failed is False
            assert "no location" in text

            models[key] = _Model([_Node(True)], lazy=True)
            failed, text = capture(model_path)
            assert failed is False
            assert "blank op_type" in text

            eager = _Model([_Node("MatMulNBits", "com.microsoft")], lazy=False)
            models[key] = eager
            checker_calls.clear()
            ok, text = capture(model_path, providers=["cpu"])
            assert ok is True
            assert checker_calls == [eager]
            assert "Model validation successful." in text

            loader.onnx.checker = types.SimpleNamespace(check_model=boom)
            failed, text = capture(model_path)
            assert failed is False
            assert "tensor data is empty" in text
            loader.onnx.checker = types.SimpleNamespace(check_model=check_model)

            failed, text = capture(model_path, providers=["vitisai"])
            assert failed is False
            assert "MatMulNBits (com.microsoft)" in text
            assert "VitisAIExecutionProvider" in text
            assert "is not supported by" in text

            failed, text = capture(model_path, providers=["ryzenai"])
            assert failed is False
            assert "VitisAIExecutionProvider" in text

            models[key] = _Model([_Node("MatMulNBits", "")], lazy=False)
            ok, text = capture(model_path, providers=["CPUExecutionProvider"])
            assert ok is True

            models[key] = _Model([_Node("MatMulNBits", "com.example")], lazy=False)
            failed, text = capture(model_path, providers=["cpu"])
            assert failed is False
            assert "MatMulNBits (com.example)" in text

            failed, text = capture(model_path, providers=["CUDAExecutionProvider"])
            assert failed is False
            assert "CUDAExecutionProvider is not an installed" in text

            failed, text = capture(os.path.join(root, "missing.onnx"))
            assert failed is False
            failed, text = capture(empty)
            assert failed is False

            expect(
                lambda: loader.validate_onnx_model(" "),
                ValueError,
                "a blank path was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(True),
                ValueError,
                "a bool path was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers=True),
                ValueError,
                "a bool provider list was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers="cpu"),
                ValueError,
                "a provider string was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers=[]),
                ValueError,
                "an empty provider list was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers=[""]),
                ValueError,
                "a blank provider was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers=[True]),
                ValueError,
                "a bool provider was accepted",
            )
            expect(
                lambda: loader.validate_onnx_model(model_path, providers=["gpu"]),
                ValueError,
                "an unknown provider name was accepted",
            )

            loader._installed_runtime = original_runtime
            runtime = loader._installed_runtime()
            assert runtime is not None
            available, real_kernels = runtime
            assert "CPUExecutionProvider" in available
            assert any(
                op_name == "MatMulNBits" and provider == "CPUExecutionProvider"
                for _domain, op_name, provider in real_kernels
            )
            models[key] = _Model([_Node("MatMulNBits", "com.microsoft")], lazy=False)
            checker_calls.clear()
            ok, text = capture(model_path, providers=["CPUExecutionProvider"])
            assert ok is True
            assert checker_calls
            assert "Model validation successful." in text
            if "VitisAIExecutionProvider" not in available:
                failed, text = capture(model_path, providers=["vitisai"])
                assert failed is False
                assert "VitisAIExecutionProvider is not an installed" in text
            else:
                has_kernel = any(
                    op_name == "MatMulNBits" and provider == "VitisAIExecutionProvider"
                    for _domain, op_name, provider in real_kernels
                )
                result, text = capture(model_path, providers=["VitisAIExecutionProvider"])
                assert result is has_kernel
    finally:
        loader.load_onnx_model = original_load
        loader._installed_runtime = original_runtime
        if had_checker:
            loader.onnx.checker = original_checker
        elif hasattr(loader.onnx, "checker"):
            del loader.onnx.checker

    print("onnx model validation: OK")
    print("  a lazy model skips the checker, MatMulNBits on VitisAI fails, and a blank path raises")
    return True


def test_split_inference():
    """Sequential, persistent, and IO-bound execution of a split ONNX graph."""
    print("=" * 50)
    print("Testing split inference...")
    print("=" * 50)

    import inspect
    import io
    import json
    import os
    import sys
    import tempfile
    import types
    import warnings
    from contextlib import redirect_stdout

    import numpy as np

    try:
        import chunking_utils.onnx as onnx_pkg
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "chunking_utils.onnx" or name.startswith("chunking_utils.onnx."):
                del sys.modules[name]
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.ModelProto = type("ModelProto", (), {})
        onnx_stub.numpy_helper = types.SimpleNamespace()
        onnx_stub.helper = types.SimpleNamespace()
        onnx_stub.TensorProto = types.SimpleNamespace()
        onnx_stub.shape_inference = types.SimpleNamespace()
        sys.modules["onnx"] = onnx_stub
        import chunking_utils.onnx as onnx_pkg

    import chunking_utils
    from chunking_utils.onnx import inference
    from chunking_utils.onnx.inference import (
        InferenceConfig,
        InferenceMode,
        SplitModelInferenceEngine,
        create_inference_engine,
        run_inference,
    )
    from pipeline_utils.session_manager import get_provider_for_device

    assert "run_inference" in onnx_pkg.__all__
    assert "SplitModelInferenceEngine" not in onnx_pkg.__all__
    assert "InferenceConfig" not in onnx_pkg.__all__
    assert "create_inference_engine" not in onnx_pkg.__all__
    assert chunking_utils.infer_onnx is run_inference
    assert list(inspect.signature(run_inference).parameters) == [
        "output_dir",
        "prompt",
        "config",
        "max_new_tokens",
    ]

    cpu = InferenceConfig(providers=["cpu"], mode="sequential", optimization_level="basic")
    assert cpu.providers == ["CPUExecutionProvider"]
    assert cpu.mode is InferenceMode.SEQUENTIAL
    assert cpu.optimization_level == "BASIC"
    assert InferenceConfig(providers=["ryzenai"]).providers == ["VitisAIExecutionProvider"]
    assert InferenceConfig(providers=["RyzenAIExecutionProvider"]).providers == [
        "RyzenAIExecutionProvider"
    ]
    npu = InferenceConfig(device="xdna", providers=["cpu"])
    assert npu.device == "npu"
    assert npu.providers == get_provider_for_device("npu")
    igpu = InferenceConfig(device="rdna")
    assert igpu.device == "igpu"
    assert igpu.providers == get_provider_for_device("gpu")
    bound = InferenceConfig(mode="io_bound", use_io_binding=False)
    assert bound.mode is InferenceMode.IO_BOUND
    assert bound.use_io_binding is True
    copied = InferenceConfig(stop_token_ids=[4])
    copied.stop_token_ids.append(9)
    again = InferenceConfig()
    assert again.stop_token_ids == []

    rejected = (
        ("providers", {"providers": True}),
        ("providers", {"providers": "cpu"}),
        ("providers", {"providers": []}),
        ("provider", {"providers": [""]}),
        ("provider", {"providers": [True]}),
        ("provider", {"providers": ["gpu"]}),
        ("mode", {"mode": True}),
        ("mode", {"mode": "draft"}),
        ("device", {"device": " "}),
        ("device", {"device": "tpu"}),
        ("max_new_tokens", {"max_new_tokens": True}),
        ("max_new_tokens", {"max_new_tokens": 0}),
        ("temperature", {"temperature": True}),
        ("temperature", {"temperature": -1}),
        ("warmup_iterations", {"warmup_iterations": -1}),
        ("max_seq_len", {"max_seq_len": True}),
        ("static_kv_cache", {"static_kv_cache": 1}),
        ("optimization_level", {"optimization_level": "fast"}),
        ("stop_token_ids", {"stop_token_ids": "4"}),
        ("stop token", {"stop_token_ids": [True]}),
        ("io_binding_device", {"io_binding_device": " "}),
        ("kv_cache_dtype", {"kv_cache_dtype": " "}),
        ("tokenizer_path", {"tokenizer_path": " "}),
    )
    for label, kwargs in rejected:
        try:
            InferenceConfig(**kwargs)
            raise AssertionError(label + " was accepted")
        except ValueError as exc:
            assert label.split()[0] in str(exc)

    class Port:
        def __init__(self, name, type_name):
            self.name = name
            self.type = type_name

    class Session:
        def __init__(self, runner, inputs, outputs):
            self.direct = 0
            self._runner = runner
            self._inputs = inputs
            self._outputs = outputs

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, feeds):
            self.direct += 1
            return self._runner(output_names, feeds)

        def produce(self, output_names, feeds):
            return self._runner(output_names, feeds)

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, prompt, return_tensors="np"):
            assert prompt == "Hello"
            return {"input_ids": np.array([[4, 6]], dtype=np.int64)}

        def decode(self, ids):
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            return ",".join(str(int(item)) for item in ids)

    ran = []
    past_shapes = []
    head_calls = []

    def embed_run(output_names, feeds):
        ran.append("embedding")
        seq = int(np.asarray(feeds["input_ids"]).shape[-1])
        return [np.ones((1, seq, 4), dtype=np.float32)]

    def layer_run(output_names, feeds):
        ran.append("layer_0")
        past = np.asarray(feeds["past_key_values.0.key"])
        past_shapes.append(tuple(past.shape))
        seq = int(np.asarray(feeds["attention_mask"]).shape[-1])
        present = np.zeros((1, 1, seq, 8), dtype=np.float16)
        hidden = np.zeros((1, seq, 4), dtype=np.float32)
        values = {
            "hidden": hidden,
            "present.0.key": present,
            "present.0.value": present.copy(),
        }
        return [values[name] for name in output_names]

    def head_run(output_names, feeds):
        ran.append("lm_head")
        head_calls.append(1)
        seq = int(np.asarray(feeds["hidden"]).shape[1])
        logits = np.zeros((1, seq, 8), dtype=np.float32)
        token = 5 if len(head_calls) == 1 else 2
        logits[0, -1, token] = 10.0
        return [logits]

    def make_sessions(direct):
        embedding = Session(
            embed_run,
            [Port("input_ids", "tensor(int64)")],
            [Port("hidden", "tensor(float32)")],
        )
        layer = Session(
            layer_run,
            [
                Port("hidden", "tensor(float32)"),
                Port("past_key_values.0.key", "tensor(float16)"),
                Port("past_key_values.0.value", "tensor(float16)"),
                Port("attention_mask", "tensor(int64)"),
                Port("position_ids", "tensor(int64)"),
            ],
            [
                Port("hidden", "tensor(float32)"),
                Port("present.0.key", "tensor(float16)"),
                Port("present.0.value", "tensor(float16)"),
            ],
        )
        head = Session(
            head_run,
            [Port("hidden", "tensor(float32)")],
            [Port("logits", "tensor(float32)")],
        )
        if not direct:
            def refuse(output_names, feeds):
                raise RuntimeError("direct session.run")
            embedding.run = refuse
            layer.run = refuse
            head.run = refuse
        return {"embedding": embedding, "layer_0": layer, "lm_head": head}

    original_session = inference.ort.InferenceSession
    original_manager = inference.SessionManager
    original_binding = inference.IOBindingManager
    original_choice = np.random.choice
    try:
        with tempfile.TemporaryDirectory() as root:
            split_dir = os.path.join(root, "split")
            os.makedirs(split_dir)
            info = {
                "num_layers": 1,
                "num_kv_heads": 1,
                "head_dim": 8,
                "components": {
                    "embedding": {
                        "file": "embedding.onnx",
                        "input_names": ["input_ids"],
                        "output_names": ["hidden"],
                    },
                    "layer_0": {
                        "file": "layer_0.onnx",
                        "input_names": [
                            "hidden",
                            "past_key_values.0.key",
                            "past_key_values.0.value",
                            "attention_mask",
                            "position_ids",
                        ],
                        "output_names": ["hidden", "present.0.key", "present.0.value"],
                    },
                    "lm_head": {
                        "file": "lm_head.onnx",
                        "input_names": ["hidden"],
                        "output_names": ["logits"],
                    },
                },
            }
            for name in ("embedding.onnx", "layer_0.onnx", "lm_head.onnx"):
                with open(os.path.join(split_dir, name), "wb") as handle:
                    handle.write(b"onnx")
            with open(os.path.join(split_dir, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(info, handle)

            def reject_manager(*args, **kwargs):
                raise AssertionError("SessionManager was constructed")

            sequential_sessions = make_sessions(True)

            def sequential_ort(path, sess_options=None, providers=None):
                assert providers == ["CPUExecutionProvider"]
                return sequential_sessions[os.path.basename(path).replace(".onnx", "")]

            inference.SessionManager = reject_manager
            inference.ort.InferenceSession = sequential_ort

            try:
                SplitModelInferenceEngine(" ", verbose=False)
                raise AssertionError("a blank path was accepted")
            except ValueError as exc:
                assert "model_dir" in str(exc)
            try:
                SplitModelInferenceEngine(True, verbose=False)
                raise AssertionError("a bool path was accepted")
            except ValueError as exc:
                assert "model_dir" in str(exc)
            try:
                SplitModelInferenceEngine(os.path.join(root, "missing"), verbose=False)
                raise AssertionError("a missing path was accepted")
            except FileNotFoundError as exc:
                assert "not found" in str(exc)
            empty = os.path.join(root, "empty")
            os.makedirs(empty)
            try:
                SplitModelInferenceEngine(empty, verbose=False)
                raise AssertionError("a directory without model_info was accepted")
            except FileNotFoundError as exc:
                assert "model_info.json" in str(exc)

            def write_info(folder, payload):
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, "model_info.json"), "w", encoding="utf-8") as handle:
                    handle.write(payload)

            cases = (
                ("not valid JSON", "{"),
                ("JSON object", "[]"),
                ("num_layers", '{"num_layers": true}'),
                ("missing num_layers", "{}"),
                ("components", '{"num_layers": 1, "components": []}'),
            )
            for label, payload in cases:
                folder = os.path.join(root, label.split()[0])
                write_info(folder, payload)
                try:
                    SplitModelInferenceEngine(folder, verbose=False)
                    raise AssertionError(label + " was accepted")
                except ValueError as exc:
                    assert label.split()[-1] in str(exc) or label in str(exc)

            mono_dir = os.path.join(root, "mono")
            os.makedirs(mono_dir)
            mono_path = os.path.join(mono_dir, "model.onnx")
            with open(mono_path, "wb") as handle:
                handle.write(b"onnx")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                monolith = SplitModelInferenceEngine(mono_path, verbose=False)
            assert monolith.is_monolith is True
            assert monolith.num_layers == 32
            assert any("No config found" in str(item.message) for item in caught)
            monolith.cleanup()

            try:
                SplitModelInferenceEngine(
                    split_dir,
                    config=InferenceConfig(kv_cache_dtype="not-a-dtype"),
                    verbose=False,
                )
                raise AssertionError("an unknown cache dtype was accepted")
            except ValueError as exc:
                assert "dtype" in str(exc).lower() or "not-a-dtype" in str(exc)

            static_engine = SplitModelInferenceEngine(
                split_dir,
                config=InferenceConfig(static_kv_cache=True, max_seq_len=32),
                verbose=False,
            )
            assert static_engine.kv_cache.static is True
            assert static_engine.kv_cache.max_seq_len == 32
            static_engine.cleanup()

            engine = SplitModelInferenceEngine(split_dir, verbose=True)
            assert engine.num_layers == 1
            assert "sequential" in repr(engine)
            assert engine.get_result().text == ""
            streamed = io.StringIO()
            with redirect_stdout(streamed):
                engine.warmup(1)
            assert "failed" in streamed.getvalue()
            try:
                engine.warmup(True)
                raise AssertionError("a bool warmup count was accepted")
            except ValueError as exc:
                assert "warmup_iterations" in str(exc)
            engine.tokenizer = Tokenizer()
            present_key, present_value, past_key, past_value = engine._detect_kv_format()
            assert present_key == "present.{layer}.key"
            assert present_value == "present.{layer}.value"
            assert past_key == "past_key_values.{layer}.key"
            assert past_value == "past_key_values.{layer}.value"
            heard = []

            def callback(index, text, elapsed):
                heard.append((index, text))

            text = engine.generate("Hello", max_new_tokens=4, stream_callback=callback)
            assert text == "4,6,5,2"
            assert ran == ["embedding", "layer_0", "lm_head", "embedding", "layer_0", "lm_head"]
            assert past_shapes == [(1, 1, 0, 8), (1, 1, 2, 8)]
            assert heard == [(0, "5"), (1, "2")]
            result = engine.get_result()
            assert result.text == text
            assert result.token_ids == [5, 2]
            assert result.num_tokens_generated == 2
            result.token_ids.append(9)
            assert engine.get_result().token_ids == [5, 2]
            assert engine.kv_cache.get_sequence_length() == 3
            stats = engine.get_stats()
            assert stats["num_tokens"] == 2
            engine.cleanup()

            ran.clear()
            past_shapes.clear()
            head_calls.clear()
            engine.tokenizer = Tokenizer()
            engine.config.temperature = 0.5
            engine.config.stop_token_ids = [3]

            def choice(size, p=None):
                return 3

            np.random.choice = choice
            sampled = engine.generate("Hello", max_new_tokens=4)
            assert sampled == "4,6,3"
            assert engine.get_result().token_ids == [3]
            np.random.choice = original_choice

            def broken_session(path, sess_options=None, providers=None):
                raise RuntimeError("unavailable")

            inference.ort.InferenceSession = broken_session
            try:
                engine.generate("Hello", max_new_tokens=1)
                raise AssertionError("a broken session was accepted")
            except RuntimeError as exc:
                assert "Failed to load embedding" in str(exc)

            try:
                engine.generate(True, max_new_tokens=1)
                raise AssertionError("a bool prompt was accepted")
            except ValueError as exc:
                assert "prompt" in str(exc)

            created = []

            class Manager:
                def __init__(self, model_dir, config=None, model_info=None):
                    self.model_dir = model_dir
                    self.config = config
                    self.model_info = model_info
                    self.loaded = 0
                    self.unloaded = 0
                    self.sessions = make_sessions(False)
                    created.append(self)

                def load_all(self):
                    self.loaded += 1
                    return {}

                def get_session(self, name):
                    return self.sessions.get(name)

                def get_memory_usage(self):
                    return {"total_mb": 2.5}

                def unload_all(self):
                    self.unloaded += 1
                    return 0

            binders = []

            class Binder:
                def __init__(self, device_type="cpu", device_id=0, auto_reallocate=True, max_seq_len=4096):
                    self.device_type = device_type
                    self.max_seq_len = max_seq_len
                    self.calls = 0
                    binders.append(self)

                def run_with_binding(self, session, inputs):
                    self.calls += 1
                    names = [item.name for item in session.get_outputs()]
                    values = session.produce(names, inputs)
                    return {name: value for name, value in zip(names, values)}

                def deallocate_all(self):
                    return 0

            ran.clear()
            past_shapes.clear()
            head_calls.clear()
            inference.SessionManager = Manager
            inference.IOBindingManager = Binder
            inference.ort.InferenceSession = broken_session
            io_engine = SplitModelInferenceEngine(
                split_dir,
                config=InferenceConfig(mode=InferenceMode.IO_BOUND, max_new_tokens=1),
                verbose=False,
            )
            io_engine.tokenizer = Tokenizer()
            io_text = io_engine.generate("Hello")
            assert io_text == "4,6,5"
            assert created[0].loaded == 1
            assert created[0].config.providers == ["CPUExecutionProvider"]
            assert binders[0].device_type == "cpu"
            assert binders[0].max_seq_len == 4096
            assert binders[0].calls == 3
            assert past_shapes == [(1, 1, 0, 8)]
            assert io_engine.get_result().peak_memory_mb == 2.5
            assert io_engine.get_result().token_ids == [5]
            io_engine.cleanup()
            assert created[0].unloaded == 1

            built = create_inference_engine(split_dir, providers=["cpu"], persist=False)
            assert built.config.mode is InferenceMode.SEQUENTIAL
            assert built.config.providers == ["CPUExecutionProvider"]
            built.cleanup()
            try:
                create_inference_engine(split_dir, persist=1)
                raise AssertionError("a non-bool persist flag was accepted")
            except ValueError as exc:
                assert "persist" in str(exc)
            try:
                run_inference(" ", prompt="Hello")
                raise AssertionError("a blank inference path was accepted")
            except ValueError as exc:
                assert "model_dir" in str(exc)
            try:
                run_inference(split_dir, prompt=True)
                raise AssertionError("a bool prompt was accepted")
            except ValueError as exc:
                assert "prompt" in str(exc)
            try:
                run_inference(split_dir, max_new_tokens=True)
                raise AssertionError("a bool token count was accepted")
            except ValueError as exc:
                assert "max_new_tokens" in str(exc)
            try:
                run_inference(split_dir, config=object())
                raise AssertionError("a plain config object was accepted")
            except ValueError as exc:
                assert "InferenceConfig" in str(exc)
            try:
                run_inference(split_dir, prompt="Hello", max_new_tokens=1)
                raise AssertionError("a missing tokenizer was accepted")
            except RuntimeError as exc:
                assert "tokenizer" in str(exc)
    finally:
        inference.ort.InferenceSession = original_session
        inference.SessionManager = original_manager
        inference.IOBindingManager = original_binding
        np.random.choice = original_choice

    print("split inference: OK")
    print("  a layer receives its past keys, IO binding skips session.run, and a blank path raises")
    return True


def test_path_manager():
    """ONNX discovery, external data, split directories, and NPU cache paths."""
    print("=" * 50)
    print("Testing path manager...")
    print("=" * 50)

    import json
    import os
    import tempfile
    import chunking_utils
    import chunking_utils.common as common
    from chunking_utils.common.path_manager import (
        find_external_data,
        find_external_data_files,
        find_onnx_model,
        get_cache_dir,
        get_npu_cache_dir,
        resolve_npu_artifact_cache,
        sanitize_model_name,
        validate_split_model,
        validate_split_model_dir,
    )
    from chunking_utils.common.path_utils import is_local_path

    assert common.find_onnx_model is find_onnx_model
    assert common.find_external_data is find_external_data
    assert common.find_external_data_files is find_external_data_files
    assert chunking_utils.find_external_data_files is find_external_data_files
    assert common.validate_split_model is validate_split_model
    assert chunking_utils.validate_split_model is validate_split_model
    assert common.validate_split_model_dir is validate_split_model_dir
    assert common.get_cache_dir is get_cache_dir
    assert common.get_npu_cache_dir is get_npu_cache_dir
    assert chunking_utils.get_npu_cache_dir is get_npu_cache_dir
    assert common.resolve_npu_artifact_cache is resolve_npu_artifact_cache
    assert chunking_utils.find_onnx_model is find_onnx_model
    assert chunking_utils.resolve_npu_artifact_cache is resolve_npu_artifact_cache
    assert sanitize_model_name("meta-llama/Llama-2-7b") == "meta-llama--Llama-2-7b"
    assert is_local_path("/home/user/model") is True
    assert is_local_path("C:\\Users\\model") is True
    assert is_local_path("./weights") is True
    assert is_local_path("../weights") is True
    assert is_local_path("meta-llama/Llama-2-7b") is False
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as local_root:
        try:
            os.chdir(local_root)
            with open("llama", "w", encoding="utf-8") as handle:
                handle.write("file")
            os.makedirs("models")
            assert is_local_path("llama") is False
            assert is_local_path("meta-llama/llama") is False
            assert is_local_path("models") is True
            assert sanitize_model_name("meta-llama/Llama-2-7b") == "meta-llama--Llama-2-7b"
        finally:
            os.chdir(previous)

    def _location(text):
        encoded = text.encode("utf-8")
        return b"\x0a\x08location\x12" + bytes([len(encoded)]) + encoded

    with tempfile.TemporaryDirectory() as root:
        empty = os.path.join(root, "empty")
        os.makedirs(empty)
        assert find_onnx_model(empty) is None
        nested = os.path.join(empty, ".ort_cache")
        os.makedirs(nested)
        with open(os.path.join(nested, "model.onnx"), "wb") as handle:
            handle.write(b"nested")
        assert find_onnx_model(empty) is None

        mono = os.path.join(root, "mono")
        os.makedirs(mono)
        model = os.path.join(mono, "model.onnx")
        other = os.path.join(mono, "layer_0.onnx")
        with open(model, "wb") as handle:
            handle.write(_location("weights.onnx.data"))
        with open(other, "wb") as handle:
            handle.write(b"layer")
        assert find_onnx_model(mono) == model
        assert find_onnx_model(model) == model
        assert find_onnx_model(os.path.join(mono, "readme.txt")) is None

        weights = os.path.join(mono, "weights.onnx.data")
        sibling = model + ".data"
        with open(weights, "wb") as handle:
            handle.write(b"weights")
        with open(sibling, "wb") as handle:
            handle.write(b"sibling")
        assert find_external_data(mono) == weights
        assert find_external_data(weights) == weights
        os.remove(weights)
        assert find_external_data(model) == sibling
        os.remove(sibling)
        assert find_external_data(model) is None

        named = os.path.join(root, "named")
        os.makedirs(named)
        graph = os.path.join(named, "model.onnx")
        with open(graph, "wb") as handle:
            handle.write(b"graph")
        onnx_data = graph + ".data"
        base_data = os.path.splitext(graph)[0] + ".data"
        underscore = os.path.splitext(graph)[0] + "_data"
        for path, payload in (
            (onnx_data, b"onnx-data"),
            (base_data, b"base-data"),
            (underscore, b"underscore"),
        ):
            with open(path, "wb") as handle:
                handle.write(payload)
        expected = [onnx_data, base_data, underscore]
        assert find_external_data_files(graph) == expected
        assert find_external_data_files(named) == expected
        os.remove(base_data)
        os.remove(underscore)
        assert find_external_data_files(graph) == [onnx_data]
        os.remove(onnx_data)
        assert find_external_data_files(graph) == []
        assert find_external_data_files(os.path.join(root, "missing.onnx")) == []

        layers = os.path.join(root, "layers")
        os.makedirs(layers)
        embed = os.path.join(layers, "embeddings.onnx")
        layer = os.path.join(layers, "layer_1.onnx")
        with open(embed, "wb") as handle:
            handle.write(b"embed")
        with open(layer, "wb") as handle:
            handle.write(b"layer")
        assert find_onnx_model(layers) == embed
        assert validate_split_model_dir(layers) is False
        assert validate_split_model_dir(mono) is False

        info = os.path.join(layers, "model_info.json")
        with open(info, "w", encoding="utf-8") as handle:
            json.dump(
                {"components": {"embedding": {"file": "embeddings.onnx"}, "layer_1": {"file": "layer_1.onnx"}}},
                handle,
            )
        assert validate_split_model_dir(layers) is True
        with open(info, "w", encoding="utf-8") as handle:
            handle.write("{")
        assert validate_split_model_dir(layers) is False
        with open(info, "w", encoding="utf-8") as handle:
            json.dump({"num_layers": 1}, handle)
        assert validate_split_model_dir(layers) is True
        with open(info, "w", encoding="utf-8") as handle:
            json.dump({"components": {"layer_9": {"file": "missing.onnx"}}}, handle)
        assert validate_split_model_dir(layers) is False
        assert validate_split_model_dir(os.path.join(root, "missing")) is False

        checked = os.path.join(root, "checked")
        os.makedirs(checked)
        names = ("embedding.onnx", "layer_0.onnx", "norm.onnx", "lm_head.onnx")
        for name in names:
            with open(os.path.join(checked, name), "wb") as handle:
                handle.write(name.encode("utf-8"))
        missing_info = validate_split_model(checked)
        assert missing_info["valid"] is False
        assert missing_info["errors"] == ["model_info.json not found"]
        assert missing_info["components"] == []
        manifest = {
            "components": {
                "embedding": {"file": "embedding.onnx"},
                "layer_0": {"file": "layer_0.onnx"},
                "norm": {"file": "norm.onnx"},
                "lm_head": {"file": "lm_head.onnx"},
            }
        }
        info_path = os.path.join(checked, "model_info.json")
        with open(info_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        complete = validate_split_model(checked)
        assert complete["valid"] is True
        assert complete["errors"] == []
        assert complete["components"] == ["embedding", "layer_0", "norm", "lm_head"]
        manifest["components"]["layer_1"] = {"file": "layer_1.onnx"}
        with open(info_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        partial = validate_split_model(checked)
        assert partial["valid"] is False
        assert partial["components"] == ["embedding", "layer_0", "norm", "lm_head"]
        assert partial["errors"] == ["Missing: " + os.path.join(checked, "layer_1.onnx")]
        with open(info_path, "w", encoding="utf-8") as handle:
            handle.write("{")
        broken = validate_split_model(checked)
        assert broken["valid"] is False
        assert broken["errors"] == ["model_info.json is not valid JSON"]
        assert broken["components"] == []

        cache = get_cache_dir(os.path.join(root, "caches"), "meta-llama/Llama-2-7b")
        assert cache == os.path.join(root, "caches", "meta-llama--Llama-2-7b")
        assert os.path.isdir(cache)
        assert get_cache_dir(os.path.join(root, "caches"), "meta-llama/Llama-2-7b") == cache

        artifact = resolve_npu_artifact_cache(mono)
        assert artifact == os.path.join(mono, ".ort_cache")
        assert os.path.isdir(os.path.join(artifact, "xclbin"))
        assert os.path.isdir(os.path.join(artifact, "vitisai"))
        from_file = resolve_npu_artifact_cache(model)
        assert from_file == artifact

        npu = get_npu_cache_dir(mono)
        assert npu == os.path.join(mono, ".npu_cache", "vitisai")
        assert os.path.isdir(npu)
        assert get_npu_cache_dir(mono) == npu
        ryzen = get_npu_cache_dir(mono, "RyzenAI")
        assert ryzen == os.path.join(mono, ".npu_cache", "ryzenai")
        assert ryzen != npu
        assert os.path.isdir(ryzen)
        assert get_npu_cache_dir(model) == npu
        assert get_npu_cache_dir(model, "VitisAI") == npu

    for label, func, bad in (
        ("directory", find_onnx_model, "  "),
        ("model_path", find_external_data, None),
        ("model_path", find_external_data_files, "  "),
        ("directory", validate_split_model, " "),
        ("directory", validate_split_model_dir, ""),
        ("base_dir", get_cache_dir, ""),
        ("model_dir", resolve_npu_artifact_cache, None),
        ("model_dir", get_npu_cache_dir, "  "),
    ):
        try:
            if func is get_cache_dir:
                func(bad, "model")
            else:
                func(bad)
            raise AssertionError(f"a blank {label} was accepted")
        except ValueError as exc:
            assert label in str(exc)
    try:
        get_cache_dir(root, "<>")
        raise AssertionError("an empty model folder was accepted")
    except ValueError as exc:
        assert "model_id" in str(exc)
    try:
        get_npu_cache_dir(root, "  ")
        raise AssertionError("a blank provider was accepted")
    except ValueError as exc:
        assert "provider" in str(exc)
    try:
        get_npu_cache_dir(root, "../escape")
        raise AssertionError("a provider path was accepted")
    except ValueError as exc:
        assert "provider" in str(exc)

    print("  onnx discovery: model.onnx, embeddings, no nested cache")
    print("  external data: embedded location, then onnx.data sibling")
    print("  split dir: model_info.json and component files")
    print("  cache: sanitized model folder, .ort_cache xclbin and vitisai")
    print("  npu cache: .npu_cache provider folder, same path on second call")
    print("path manager: OK\n")
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

    try:
        results.append(("hf_integration", test_hf_integration()))
    except Exception as e:
        print(f"hf_integration: FAILED - {e}")
        results.append(("hf_integration", False))

    try:
        results.append(("architecture", test_architecture()))
    except Exception as e:
        print(f"architecture: FAILED - {e}")
        results.append(("architecture", False))

    try:
        results.append(("graph_topology", test_graph_topology()))
    except Exception as e:
        print(f"graph_topology: FAILED - {e}")
        results.append(("graph_topology", False))

    try:
        results.append(("boundary_contract", test_boundary_contract()))
    except Exception as e:
        print(f"boundary_contract: FAILED - {e}")
        results.append(("boundary_contract", False))

    try:
        results.append(("output_detection", test_output_detection()))
    except Exception as e:
        print(f"output_detection: FAILED - {e}")
        results.append(("output_detection", False))

    try:
        results.append(("dtype_resolution", test_dtype_resolution()))
    except Exception as e:
        print(f"dtype_resolution: FAILED - {e}")
        results.append(("dtype_resolution", False))

    try:
        results.append(("quantization_metadata", test_quantization_metadata()))
    except Exception as e:
        print(f"quantization_metadata: FAILED - {e}")
        results.append(("quantization_metadata", False))

    try:
        results.append(("kv_cache_metadata", test_kv_cache_metadata()))
    except Exception as e:
        print(f"kv_cache_metadata: FAILED - {e}")
        results.append(("kv_cache_metadata", False))

    try:
        results.append(("execution_policy", test_execution_policy()))
    except Exception as e:
        print(f"execution_policy: FAILED - {e}")
        results.append(("execution_policy", False))

    try:
        results.append(("split_validation", test_split_validation()))
    except Exception as e:
        print(f"split_validation: FAILED - {e}")
        results.append(("split_validation", False))

    try:
        results.append(("weight_storage", test_weight_storage()))
    except Exception as e:
        print(f"weight_storage: FAILED - {e}")
        results.append(("weight_storage", False))

    try:
        results.append(("package_structure", test_package_structure()))
    except Exception as e:
        print(f"package_structure: FAILED - {e}")
        results.append(("package_structure", False))

    try:
        results.append(("merge_order", test_merge_order()))
    except Exception as e:
        print(f"merge_order: FAILED - {e}")
        results.append(("merge_order", False))

    try:
        results.append(("other_dependencies", test_other_dependencies()))
    except Exception as e:
        print(f"other_dependencies: FAILED - {e}")
        results.append(("other_dependencies", False))

    try:
        results.append(("boundary_reconnection", test_boundary_reconnection()))
    except Exception as e:
        print(f"boundary_reconnection: FAILED - {e}")
        results.append(("boundary_reconnection", False))

    try:
        results.append(("quantization_coherence", test_quantization_coherence()))
    except Exception as e:
        print(f"quantization_coherence: FAILED - {e}")
        results.append(("quantization_coherence", False))

    try:
        results.append(("merge_psnr", test_merge_psnr()))
    except Exception as e:
        print(f"merge_psnr: FAILED - {e}")
        results.append(("merge_psnr", False))

    try:
        results.append(("streaming_merge", test_streaming_merge()))
    except Exception as e:
        print(f"streaming_merge: FAILED - {e}")
        results.append(("streaming_merge", False))

    try:
        results.append(("boundary_tensor_info", test_boundary_tensor_info()))
    except Exception as e:
        print(f"boundary_tensor_info: FAILED - {e}")
        results.append(("boundary_tensor_info", False))

    try:
        results.append(("quantization_info", test_quantization_info()))
    except Exception as e:
        print(f"quantization_info: FAILED - {e}")
        results.append(("quantization_info", False))

    try:
        results.append(("kv_cache_contract", test_kv_cache_contract()))
    except Exception as e:
        print(f"kv_cache_contract: FAILED - {e}")
        results.append(("kv_cache_contract", False))

    try:
        results.append(("execution_plan_type", test_execution_plan_type()))
    except Exception as e:
        print(f"execution_plan_type: FAILED - {e}")
        results.append(("execution_plan_type", False))

    try:
        results.append(("schema_version", test_schema_version()))
    except Exception as e:
        print(f"schema_version: FAILED - {e}")
        results.append(("schema_version", False))

    try:
        results.append(("external_data_fallback", test_external_data_fallback()))
    except Exception as e:
        print(f"external_data_fallback: FAILED - {e}")
        results.append(("external_data_fallback", False))

    try:
        results.append(("genai_external_data", test_genai_external_data()))
    except Exception as e:
        print(f"genai_external_data: FAILED - {e}")
        results.append(("genai_external_data", False))

    try:
        results.append(("lazy_shape_inference", test_lazy_shape_inference()))
    except Exception as e:
        print(f"lazy_shape_inference: FAILED - {e}")
        results.append(("lazy_shape_inference", False))

    try:
        results.append(("onnx_model_validation", test_onnx_model_validation()))
    except Exception as e:
        print(f"onnx_model_validation: FAILED - {e}")
        results.append(("onnx_model_validation", False))

    try:
        results.append(("split_inference", test_split_inference()))
    except Exception as e:
        print(f"split_inference: FAILED - {e}")
        results.append(("split_inference", False))

    try:
        results.append(("path_manager", test_path_manager()))
    except Exception as e:
        print(f"path_manager: FAILED - {e}")
        results.append(("path_manager", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

