"""Quark pre-split stage. No device, no foundation model, no XDNA compile."""

import inspect
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking_utils.onnx.quark_optimize import (
    NPU_PIPELINE,
    PRESET_FOR_BITS,
    QuarkOptimizeConfig,
    QuarkUnavailable,
    documented_overrides,
    fusion_extra_options,
    optimize_for_npu,
    pipeline_order,
    pre_split_plan,
    prepare_model_for_split,
    quark_import_info,
    quark_imported,
)
from chunking_utils.onnx.splitter import split_onnx


def test_config_validation():
    try:
        QuarkOptimizeConfig(bits=3)
    except ValueError as exc:
        assert "4 or 8" in str(exc)
    else:
        raise AssertionError("bits=3 was accepted")

    try:
        QuarkOptimizeConfig(bits=True)
    except ValueError as exc:
        assert "4 or 8" in str(exc)
    else:
        raise AssertionError("bits=True was accepted")

    try:
        QuarkOptimizeConfig(calibration="banana")
    except ValueError as exc:
        assert "calibration" in str(exc)
    else:
        raise AssertionError("an unknown calibration method was accepted")

    try:
        QuarkOptimizeConfig(bits=4, group_size=0)
    except ValueError as exc:
        assert "group_size" in str(exc)
    else:
        raise AssertionError("group_size 0 was accepted")

    try:
        QuarkOptimizeConfig(bits=8, group_size=32)
    except ValueError as exc:
        assert "int4" in str(exc)
    else:
        raise AssertionError("group_size was accepted for int8")

    try:
        optimize_for_npu("", os.path.join(tempfile.gettempdir(), "quark-out.onnx"), bits=8)
    except ValueError as exc:
        assert "model_path" in str(exc)
    else:
        raise AssertionError("an empty model path was accepted")

    try:
        optimize_for_npu("model.onnx", "  ", bits=8)
    except ValueError as exc:
        assert "output_path" in str(exc)
    else:
        raise AssertionError("an empty output path was accepted")

    int8 = QuarkOptimizeConfig()
    assert int8.bits == 8
    assert int8.preset == PRESET_FOR_BITS[8] == "INT8_TRANSFORMER_DEFAULT"
    assert int8.fusion is True
    int4 = QuarkOptimizeConfig(bits=4, calibration="MinMSE")
    assert int4.preset == "MATMUL_NBITS"
    assert int4.group_size == 128
    assert int4.calibration == "minmse"
    assert int4.to_dict()["pipeline"] == ["rewrite", "quark", "split"]
    print("  config validation: bad bits, empty path, int4 and int8 presets")


def test_stage_order():
    names = [stage.name for stage in NPU_PIPELINE]
    assert names == ["rewrite", "quark", "split"]
    assert pipeline_order() == ("rewrite", "quark", "split")
    assert pre_split_plan(True) == ("rewrite", "quark", "split")
    assert pre_split_plan(False) == ("rewrite", "split")
    assert names.index("rewrite") < names.index("quark") < names.index("split")
    assert [stage.owner for stage in NPU_PIPELINE] == [
        "caller",
        "optimize_for_npu",
        "split_onnx",
    ]
    turned_on = fusion_extra_options(True)
    assert turned_on == {
        "OptimizeModel": True,
        "FuseGelu": True,
        "FuseLayerNorm": True,
    }
    turned_off = fusion_extra_options(False)
    assert turned_off["OptimizeModel"] is False
    assert turned_off["SkipPreprocess"] is True
    int4 = documented_overrides(
        QuarkOptimizeConfig(bits=4, fusion=True),
        use_random_data=True,
    )
    assert int4["preset"] == "MATMUL_NBITS"
    assert int4["extra_options"]["UseMatMulNBits"] is True
    assert int4["extra_options"]["MatMulNBitsParams"]["Bits"] == 4
    assert int4["extra_options"]["UseRandomData"] is True
    assert "prepare_model_for_split" in split_onnx.__code__.co_names
    assert "pre_split_plan" in prepare_model_for_split.__code__.co_names
    assert "optimize_for_npu" in prepare_model_for_split.__code__.co_names
    print("  stage order: rewrite, then quark, then split")


def test_missing_quark_is_not_success():
    if quark_imported():
        info = quark_import_info()
        assert info["imported"] is True
        assert info["model_quantizer"].endswith("ModelQuantizer")
        assert info["get_default_config"].endswith("get_default_config")
        assert info["xdna_compiled"] is False
        print("  quark import: present, import path only, no XDNA compile")
        return

    output_dir = tempfile.mkdtemp(prefix="quark-missing-")
    target = os.path.join(output_dir, "quantized.onnx")
    try:
        optimize_for_npu("model.onnx", target, bits=8)
    except QuarkUnavailable as exc:
        text = str(exc)
        assert "no model was quantized" in text.lower()
    else:
        raise AssertionError("missing Quark reported success")
    assert not os.path.exists(target)

    try:
        split_onnx(model="model.onnx", output=output_dir, quark=True, split_only=True)
    except QuarkUnavailable as exc:
        assert "no model was quantized" in str(exc).lower()
    else:
        raise AssertionError("split_onnx with quark=True reported success without Quark")
    written = [name for name in os.listdir(output_dir) if name.endswith(".onnx")]
    assert written == []
    print("  missing quark: raises QuarkUnavailable and writes no ONNX file")


def test_split_flag_defaults_off():
    signature = inspect.signature(split_onnx)
    assert signature.parameters["quark"].default is False
    calls = []

    def _spy(model_path, output_dir, *, quark=False, config=None):
        calls.append(quark)
        assert quark is False
        assert pre_split_plan(quark) == ("rewrite", "split")
        return model_path

    import chunking_utils.onnx.splitter as splitter_module

    original = splitter_module.prepare_model_for_split
    splitter_module.prepare_model_for_split = _spy
    output_dir = tempfile.mkdtemp(prefix="quark-default-off-")
    try:
        try:
            split_onnx(model="missing-model.onnx", output=output_dir, split_only=True)
        except QuarkUnavailable:
            raise AssertionError("default split tried to run Quark")
        except Exception:
            pass
    finally:
        splitter_module.prepare_model_for_split = original
    assert calls == [False]
    unchanged = prepare_model_for_split("missing-model.onnx", output_dir, quark=False)
    assert unchanged == "missing-model.onnx"
    written = [name for name in os.listdir(output_dir) if name.endswith(".onnx")]
    assert written == []
    print("  split flag: quark defaults off and does not call Quark")


def main():
    print("=" * 50)
    print("Testing Quark pre-split stage...")
    print("=" * 50)
    test_config_validation()
    test_stage_order()
    test_missing_quark_is_not_success()
    test_split_flag_defaults_off()
    print("test_quark: OK")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
