"""Model artifact facts measured from a real model file (no claims about data).

Used by the evaluation CLI to record reproducible, verifiable information about
the *artifact itself*: parameter count, footprint on disk, input shape and
single-image inference latency. These are true of the file on disk regardless
of any dataset, and are recorded separately from metric claims.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _shape_list(shape) -> list | None:
    """Normalize a Keras shape (.as_list()) or a plain tuple/list to a list."""
    if shape is None:
        return None
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    if not isinstance(shape, (tuple, list)):
        return None
    return [int(d) if d is not None else None for d in shape]


def model_spec(model_path: Path, input_size: int = 64) -> dict:
    """Return architecture-level facts for a Keras model artifact."""
    from tensorflow import keras

    model = keras.models.load_model(str(model_path))
    try:
        parameters = int(model.count_params())
        summary = _arch_summary_lines(model)
        inputs = model.inputs if model.inputs else ()
        input_shape = None
        if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
            input_shape = _shape_list(getattr(inputs[0], "shape", inputs[0]))
        output = _output_spec(model)
        layers = len(model.layers)
    finally:
        keras.backend.clear_session()

    return {
        "architecture_hint": summary.get("architecture", ""),
        "parameter_count": parameters,
        "weight_layers": layers,
        "input_shape": input_shape,
        "output": output,
        "file_size_bytes": Path(model_path).stat().st_size,
        "keras_version": keras.__version__,
    }


def _arch_summary_lines(model) -> dict:
    import io

    buf = io.StringIO()
    model.summary(line_length=100, print_fn=lambda line: buf.write(line + "\n"))
    return {"architecture": model.name or type(model).__name__, "layers": buf.getvalue().splitlines()}


def _output_spec(model) -> dict:
    if not model.outputs:
        return {"shape": None, "classes": 0}
    shape = _shape_list(getattr(model.outputs[0], "shape", model.outputs[0]))
    classes = int(shape[-1]) if shape and shape[-1] else 0
    return {"shape": shape, "classes": classes}


def measure_latency(model_path: Path, input_size: int = 64, repeats: int = 5) -> dict:
    """Median single-image inference latency on this machine, plus histogram."""
    from tensorflow import keras

    model = keras.models.load_model(str(model_path))
    sample = np.random.default_rng(42).uniform(0, 1, (1, input_size, input_size, 3))
    timings: list[float] = []
    try:
        model.predict(sample, verbose=0)  # warm-up
        for _ in range(repeats):
            started = time.perf_counter()
            model.predict(sample, verbose=0)
            timings.append((time.perf_counter() - started) * 1000)
    finally:
        keras.backend.clear_session()

    timings = sorted(timings)
    return {
        "repeats": repeats,
        "median_ms": round(float(timings[len(timings) // 2]), 2),
        "min_ms": round(float(timings[0]), 2),
        "max_ms": round(float(timings[-1]), 2),
        "platform_note": (
            "Measured locally; latency depends on hardware and backend and is "
            "reported only for reproducibility context."
        ),
    }


def measure_serving_latency(backend, input_size: int = 64, repeats: int = 3) -> dict:
    """Median end-to-end latency of the app's own backend (includes Grad-CAM)."""
    sample = np.random.default_rng(7).uniform(0, 1, (1, input_size, input_size, 3))
    timings: list[float] = []
    backend.grad_cam(sample, 0)  # warm-up
    for _ in range(repeats):
        started = time.perf_counter()
        backend.predict(sample)
        backend.grad_cam(sample, int(backend.predict(sample).argmax()))
        timings.append((time.perf_counter() - started) * 1000)
    timings = sorted(timings)
    return {
        "repeats": repeats,
        "median_ms": round(float(timings[len(timings) // 2]), 2),
        "min_ms": round(float(timings[0]), 2),
        "max_ms": round(float(timings[-1]), 2),
        "includes": "predict + Grad-CAM on the application backend",
    }