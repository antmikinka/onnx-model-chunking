"""
AMD Quark pre-split stage.

The NPU pipeline order is data, and both ``optimize_for_npu`` and
``split_onnx`` read it:

    rewrite (caller-supplied) -> quark -> split

Rewrite of KV cache, attention masks, phase graphs, vocabulary chunks,
and static shapes stays on the branches that own those passes. This
module only runs Quark, and only when the caller asks.

Quark 0.13 quantizes an ONNX graph. ``ModelQuantizer.quantize_model``
writes another ONNX file. It does not compile an XDNA binary. INT8 uses
the ``INT8_TRANSFORMER_DEFAULT`` preset (``enable_npu_transformer``).
INT4 uses ``MATMUL_NBITS`` (weight-only, 4-bit ``MatMulNBitsParams``).
Fusion maps to the documented flags ``OptimizeModel``, ``FuseGelu``, and
``FuseLayerNorm``. A missing ``quark`` import raises ``QuarkUnavailable``
and writes nothing.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class QuarkUnavailable(RuntimeError):
    """Quark did not import, or its preset could not be built. Nothing was quantized."""


@dataclass(frozen=True)
class PipelineStage:
    """One named step. ``owner`` is the code that is allowed to run it."""

    name: str
    owner: str


NPU_PIPELINE: Tuple[PipelineStage, ...] = (
    PipelineStage("rewrite", "caller"),
    PipelineStage("quark", "optimize_for_npu"),
    PipelineStage("split", "split_onnx"),
)

PRESET_FOR_BITS = {
    4: "MATMUL_NBITS",
    8: "INT8_TRANSFORMER_DEFAULT",
}

PRESET_BITS = {
    "MATMUL_NBITS": 4,
    "INT8_TRANSFORMER_DEFAULT": 8,
    "INT8_TRANSFORMER_ACCURATE": 8,
    "XINT8": 8,
    "A8W8": 8,
    "A16W8": 8,
}

CALIBRATION_METHODS = ("minmax", "minmse", "percentile", "entropy")

_CALIBRATION_ATTR = {
    "minmax": ("calibration_method", "MinMax"),
    "percentile": ("calibration_method", "Percentile"),
    "entropy": ("calibration_method", "Entropy"),
    "minmse": ("power_of_two", "MinMSE"),
}


def pipeline_order() -> Tuple[str, ...]:
    """Return the stage names in the order the pipeline executes them."""
    return tuple(stage.name for stage in NPU_PIPELINE)


def pre_split_plan(quark: bool) -> Tuple[str, ...]:
    """
    Stages that lead up to and include split.

    ``rewrite`` is always first and is already done by the caller.
    ``quark`` is present only when the caller set the flag.
    ``split`` is always last.
    """
    if not isinstance(quark, bool):
        raise ValueError(f"quark must be a bool, got {quark!r}")
    order = pipeline_order()
    expected = ("rewrite", "quark", "split")
    if order != expected:
        raise RuntimeError(f"NPU pipeline order is {order}, expected {expected}")
    rewrite_at = order.index("rewrite")
    quark_at = order.index("quark")
    split_at = order.index("split")
    if not (rewrite_at < quark_at < split_at):
        raise RuntimeError(f"NPU pipeline order is {order}")
    if quark:
        return order
    return ("rewrite", "split")


def fusion_extra_options(enabled: bool) -> Dict[str, Any]:
    """Documented Quark graph-optimization flags. No MatMul+Add fusion key exists."""
    if not isinstance(enabled, bool):
        raise ValueError(f"fusion must be a bool, got {enabled!r}")
    if enabled:
        return {
            "OptimizeModel": True,
            "FuseGelu": True,
            "FuseLayerNorm": True,
        }
    return {
        "OptimizeModel": False,
        "FuseGelu": False,
        "FuseLayerNorm": False,
        "SkipPreprocess": True,
    }


@dataclass(frozen=True)
class QuarkOptimizeConfig:
    """
    Quantization request recorded before ``split_onnx``.

    ``bits`` is 4 or 8. ``calibration`` None keeps the preset's method.
    ``fusion`` turns the documented optimize and fuse flags on or off.
    ``preset`` None selects ``PRESET_FOR_BITS``.
    """

    bits: int = 8
    calibration: Optional[str] = None
    fusion: bool = True
    group_size: Optional[int] = None
    preset: Optional[str] = None
    use_external_data: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or self.bits not in PRESET_FOR_BITS:
            raise ValueError(f"quark bits must be 4 or 8, got {self.bits!r}")
        if not isinstance(self.fusion, bool):
            raise ValueError(f"fusion must be a bool, got {self.fusion!r}")
        if not isinstance(self.use_external_data, bool):
            raise ValueError(
                f"use_external_data must be a bool, got {self.use_external_data!r}"
            )
        calibration = self.calibration
        if calibration is not None:
            if not isinstance(calibration, str) or not calibration.strip():
                raise ValueError(
                    f"calibration must be a method name, got {calibration!r}"
                )
            calibration = calibration.strip().lower()
            if calibration not in CALIBRATION_METHODS:
                known = ", ".join(CALIBRATION_METHODS)
                raise ValueError(
                    f"calibration {calibration!r} is not one of: {known}"
                )
            object.__setattr__(self, "calibration", calibration)
        if self.bits == 4:
            group_size = 128 if self.group_size is None else self.group_size
            if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(
                    f"group_size must be a positive integer, got {self.group_size!r}"
                )
            object.__setattr__(self, "group_size", group_size)
        elif self.group_size is not None:
            raise ValueError("group_size applies to int4 MatMulNBits only")
        preset = PRESET_FOR_BITS[self.bits] if self.preset is None else self.preset
        if not isinstance(preset, str) or preset not in PRESET_BITS:
            known = ", ".join(sorted(PRESET_BITS))
            raise ValueError(f"quark preset {preset!r} is not one of: {known}")
        if PRESET_BITS[preset] != self.bits:
            raise ValueError(
                f"preset {preset} is a {PRESET_BITS[preset]}-bit config, "
                f"got bits {self.bits}"
            )
        object.__setattr__(self, "preset", preset)

    def to_dict(self) -> Dict[str, Any]:
        """Copy the request. The splitter can store this before it splits."""
        return {
            "bits": self.bits,
            "calibration": self.calibration,
            "fusion": self.fusion,
            "group_size": self.group_size,
            "preset": self.preset,
            "use_external_data": self.use_external_data,
            "pipeline": list(pipeline_order()),
            "fusion_options": fusion_extra_options(self.fusion),
        }


@dataclass(frozen=True)
class QuarkOptimizeResult:
    """A quantized ONNX file. ``xdna_compiled`` stays false: Quark does not emit an xclbin."""

    output_path: str
    preset: str
    bits: int
    calibration: Optional[str]
    fusion: bool
    pipeline: Tuple[str, ...]
    quantized: bool
    xdna_compiled: bool = False


@dataclass(frozen=True)
class _QuarkAPI:
    model_quantizer: Any
    config_type: Any
    get_default_config: Any
    calibration_method: Any
    power_of_two: Any

    @property
    def model_quantizer_name(self) -> str:
        cls = self.model_quantizer
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def get_default_config_name(self) -> str:
        fn = self.get_default_config
        return f"{fn.__module__}.{fn.__qualname__}"


def _non_empty_path(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip()


def load_quark_api() -> _QuarkAPI:
    """
    Import the Quark 0.13 ONNX entry points.

    ``get_default_config`` returns a copied ``QuantizationConfig``.
    ``Config(global_quant_config=...)`` is what ``ModelQuantizer`` accepts.
    """
    try:
        from quark.onnx import CalibrationMethod, ModelQuantizer, PowerOfTwoMethod
        from quark.onnx.quantization.config import Config, get_default_config
    except Exception as exc:
        raise QuarkUnavailable(
            "AMD Quark did not import, so no model was quantized. "
            "Install amd-quark (import name quark) and its runtime dependencies, "
            "then rerun optimize_for_npu. "
            f"Original error: {exc}"
        ) from exc
    if not callable(getattr(ModelQuantizer, "quantize_model", None)):
        raise QuarkUnavailable(
            "quark.onnx.ModelQuantizer.quantize_model is missing. "
            "No model was quantized."
        )
    if not callable(get_default_config):
        raise QuarkUnavailable(
            "quark.onnx.quantization.config.get_default_config is missing. "
            "No model was quantized."
        )
    return _QuarkAPI(
        model_quantizer=ModelQuantizer,
        config_type=Config,
        get_default_config=get_default_config,
        calibration_method=CalibrationMethod,
        power_of_two=PowerOfTwoMethod,
    )


def quark_imported() -> bool:
    """True only when the Quark ONNX quantizer imports."""
    try:
        load_quark_api()
    except QuarkUnavailable:
        return False
    return True


def quark_import_info() -> Dict[str, Any]:
    """Names of the live Quark callables. Raises when Quark did not import."""
    api = load_quark_api()
    return {
        "imported": True,
        "model_quantizer": api.model_quantizer_name,
        "get_default_config": api.get_default_config_name,
        "xdna_compiled": False,
    }


def documented_overrides(config: QuarkOptimizeConfig, *, use_random_data: bool) -> Dict[str, Any]:
    """The extra_options this stage writes onto a Quark preset."""
    if not isinstance(config, QuarkOptimizeConfig):
        raise ValueError(
            f"quark config must be QuarkOptimizeConfig, got {type(config).__name__}"
        )
    if not isinstance(use_random_data, bool):
        raise ValueError(f"use_random_data must be a bool, got {use_random_data!r}")
    extra = fusion_extra_options(config.fusion)
    if config.bits == 4:
        extra["UseMatMulNBits"] = True
        extra["MatMulNBitsParams"] = {
            "Bits": 4,
            "GroupSize": config.group_size,
            "Symmetric": True,
        }
    if use_random_data:
        extra["UseRandomData"] = True
    return {
        "preset": config.preset,
        "optimize_model": config.fusion,
        "use_external_data_format": config.use_external_data,
        "calibration": config.calibration,
        "extra_options": extra,
    }


def _coerce_config(config: Any, options: Dict[str, Any]) -> QuarkOptimizeConfig:
    if config is not None and options:
        raise ValueError("pass a QuarkOptimizeConfig or keyword options, not both")
    if config is None:
        return QuarkOptimizeConfig(**options)
    if isinstance(config, QuarkOptimizeConfig):
        return config
    if isinstance(config, dict):
        return QuarkOptimizeConfig(**config)
    raise ValueError(
        f"quark config must be QuarkOptimizeConfig, got {type(config).__name__}"
    )


def _calibration_value(api: _QuarkAPI, name: str) -> Any:
    owner_name, attr = _CALIBRATION_ATTR[name]
    owner = getattr(api, owner_name)
    value = getattr(owner, attr, None)
    if value is None:
        raise QuarkUnavailable(
            f"Quark has no {owner_name}.{attr} for calibration {name}. "
            "No model was quantized."
        )
    return value


def _inner_config(api: _QuarkAPI, built: Any) -> Any:
    """Accept a QuantizationConfig or a Config that wraps one."""
    inner = built
    if hasattr(built, "global_quant_config"):
        inner = built.global_quant_config
    extra = getattr(inner, "extra_options", None)
    if not isinstance(extra, dict):
        raise QuarkUnavailable(
            "Quark preset has no extra_options dict. No model was quantized."
        )
    return inner


def _apply_overrides(api: _QuarkAPI, inner: Any, config: QuarkOptimizeConfig, reader: Any) -> Any:
    overrides = documented_overrides(config, use_random_data=reader is None)
    extra = inner.extra_options
    if config.calibration is not None:
        if not hasattr(inner, "calibrate_method"):
            raise QuarkUnavailable(
                "Quark config has no calibrate_method. No model was quantized."
            )
        inner.calibrate_method = _calibration_value(api, config.calibration)
    if hasattr(inner, "optimize_model"):
        inner.optimize_model = overrides["optimize_model"]
    if hasattr(inner, "use_external_data_format"):
        inner.use_external_data_format = overrides["use_external_data_format"]
    incoming = overrides["extra_options"]
    nbits = incoming.get("MatMulNBitsParams")
    if isinstance(nbits, dict):
        current = extra.get("MatMulNBitsParams")
        if not isinstance(current, dict):
            raise QuarkUnavailable(
                "Quark MATMUL_NBITS preset has no MatMulNBitsParams. "
                "Refusing to invent an INT4 quantizer."
            )
        current["Bits"] = nbits["Bits"]
        current["GroupSize"] = nbits["GroupSize"]
        if "Symmetric" in nbits and "Symmetric" not in current:
            current["Symmetric"] = nbits["Symmetric"]
        incoming = {key: value for key, value in incoming.items() if key != "MatMulNBitsParams"}
    extra.update(incoming)
    if hasattr(inner, "global_quant_config"):
        return inner
    return api.config_type(global_quant_config=inner)


def _build_quant_config(api: _QuarkAPI, config: QuarkOptimizeConfig, reader: Any) -> Any:
    try:
        built = api.get_default_config(config.preset)
    except ValueError as exc:
        raise QuarkUnavailable(
            f"Quark has no preset {config.preset}. No model was quantized."
        ) from exc
    inner = _inner_config(api, built)
    wrapped = _apply_overrides(api, inner, config, reader)
    return wrapped


def optimize_for_npu(
    model_path: str,
    output_path: str,
    config: Any = None,
    calibration_reader: Any = None,
    **options: Any,
) -> QuarkOptimizeResult:
    """
    Quantize ``model_path`` with AMD Quark and write ``output_path``.

    The caller is responsible for the rewrite stage. This function is the
    quark stage and it runs before split. A missing Quark install raises
    ``QuarkUnavailable`` before any output file is created.
    """
    cfg = _coerce_config(config, options)
    model_path = _non_empty_path(model_path, "model_path")
    output_path = _non_empty_path(output_path, "output_path")
    plan = pre_split_plan(quark=True)
    if plan != ("rewrite", "quark", "split"):
        raise RuntimeError(f"NPU pipeline plan is {plan}")
    api = load_quark_api()
    if not os.path.isfile(model_path):
        raise ValueError(f"missing file {model_path}")
    quant_config = _build_quant_config(api, cfg, calibration_reader)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    quantizer = api.model_quantizer(quant_config)
    quantizer.quantize_model(model_path, output_path, calibration_reader)
    if not os.path.isfile(output_path):
        raise QuarkUnavailable(
            "Quark quantize_model returned without writing a quantized model. "
            f"Expected {output_path}."
        )
    return QuarkOptimizeResult(
        output_path=output_path,
        preset=str(cfg.preset),
        bits=cfg.bits,
        calibration=cfg.calibration,
        fusion=cfg.fusion,
        pipeline=plan,
        quantized=True,
        xdna_compiled=False,
    )


def prepare_model_for_split(
    model_path: str,
    output_dir: str,
    *,
    quark: bool = False,
    config: Any = None,
) -> str:
    """
    Return the ONNX path ``split_onnx`` should load.

    With ``quark`` false, the path is unchanged and Quark is not imported.
    With ``quark`` true, Quark runs and the quantized file is returned.
    """
    plan = pre_split_plan(quark)
    if plan[0] != "rewrite" or plan[-1] != "split":
        raise RuntimeError(f"NPU pipeline plan is {plan}")
    if "quark" not in plan:
        return model_path
    if plan != ("rewrite", "quark", "split"):
        raise RuntimeError(f"NPU pipeline plan is {plan}")
    model_path = _non_empty_path(model_path, "model_path")
    output_dir = _non_empty_path(output_dir, "output_path")
    quantized_path = os.path.join(output_dir, "quark_pre_split.onnx")
    result = optimize_for_npu(model_path, quantized_path, config=config)
    if not result.quantized or result.xdna_compiled:
        raise QuarkUnavailable(
            "optimize_for_npu did not return a quantized ONNX file. "
            "No XDNA compile is performed."
        )
    return result.output_path


__all__ = [
    "CALIBRATION_METHODS",
    "NPU_PIPELINE",
    "PRESET_BITS",
    "PRESET_FOR_BITS",
    "PipelineStage",
    "QuarkOptimizeConfig",
    "QuarkOptimizeResult",
    "QuarkUnavailable",
    "documented_overrides",
    "fusion_extra_options",
    "load_quark_api",
    "optimize_for_npu",
    "pipeline_order",
    "pre_split_plan",
    "prepare_model_for_split",
    "quark_import_info",
    "quark_imported",
]
