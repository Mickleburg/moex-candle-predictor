"""Inference service.

Package-level re-exports are LAZY (PEP 562 ``__getattr__``): importing a submodule such as
``ml.src.service.dividend_sleeve`` or ``.risk_analytics`` (the only pieces the slim VDS runtime uses)
must NOT drag in the optional V1 serving stack — ``.api`` needs fastapi, ``.schemas`` needs pydantic,
``.research_artifact``/``.predictor`` need torch — none of which are in the runtime ``requirements.txt``.
Those names still resolve on first access for callers (and the dev env) that have the extra deps.
"""

import importlib

# exported name -> submodule that defines it (imported on first attribute access, not at package init)
_LAZY = {
    "app": ".api",
    "CandlePredictor": ".predictor",
    "CandleBatch": ".contracts",
    "load_candle_batch_json": ".contracts",
    "candle_batch_to_dataframe": ".contracts",
    "build_ml_prediction_response": ".contracts",
    "load_research_artifact": ".research_artifact",
    "predict_with_artifact": ".research_artifact",
    "TickerModelRouter": ".model_registry",
    "predict_candle_batch": ".model_registry",
    "resolve_artifact_dir": ".model_registry",
    "Candle": ".schemas",
    "PredictRequest": ".schemas",
    "PredictResponse": ".schemas",
    "HealthResponse": ".schemas",
    "ErrorResponse": ".schemas",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
