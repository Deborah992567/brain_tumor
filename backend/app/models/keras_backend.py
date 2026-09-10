"""Keras/TensorFlow model backend.

Implements prediction and Grad-CAM for the existing CNN (``brain_tumor.h5``)
as well as any future Keras models. TensorFlow is imported lazily so that the
rest of the application (and its tests) can run without TensorFlow present.
"""

from __future__ import annotations

import threading

import numpy as np

from app.services.errors import ModelUnavailableError
from app.services.model_backend import ModelBackend

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]


class KerasModelBackend(ModelBackend):
    def __init__(
        self,
        file_path: str,
        name: str,
        version: str,
        architecture: str,
        input_size: int = 64,
        classes: list[str] | None = None,
    ) -> None:
        self.file_path = file_path
        self.name = name
        self.version = version
        self.architecture = architecture
        self.input_size = input_size
        self.classes = classes or CLASSES
        self._model: object | None = None
        self._submodel: object | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<KerasModelBackend {self.name} v{self.version} ({self.file_path})>"

    def _import_tf(self):
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from exc
        return tf

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            tf = self._import_tf()
            try:
                model = tf.keras.models.load_model(self.file_path, compile=False)
                if model is None:
                    raise ValueError("model could not be loaded")
            except Exception as exc:
                raise ModelUnavailableError(
                    f"Model '{self.name} v{self.version}' could not be loaded from "
                    f"{self.file_path}."
                ) from exc

            last_conv = self._find_last_conv(model)
            inputs = model.inputs[0]

            if last_conv is not None:
                try:
                    submodel = tf.keras.Model(
                        inputs=inputs, outputs=[last_conv.output, model.output]
                    )
                except Exception:  # pragma: no cover - defensive
                    submodel = None
            else:
                submodel = None

            self._model = model
            self._submodel = submodel

    def _find_last_conv(self, model):
        for layer in reversed(model.layers):
            if "conv" in layer.__class__.__name__.lower() and hasattr(layer, "output"):
                return layer
        return None

    def must_reload(self) -> bool:
        return self._model is None

    def predict(self, preprocessed: np.ndarray) -> np.ndarray:
        self.load()
        if self._model is None:  # pragma: no cover - defensive
            raise ModelUnavailableError("Model is not available.")
        probs = self._model.predict(preprocessed, verbose=0)
        return np.asarray(probs, dtype=np.float32).reshape(-1)

    def grad_cam(self, preprocessed: np.ndarray, class_index: int) -> np.ndarray:
        self.load()
        if self._submodel is None:
            raise ModelUnavailableError(
                "Attention visualization is not available for this model."
            )
        tf = self._import_tf()
        conv_output, predictions = self._submodel(preprocessed, training=False)
        class_channel = predictions[:, class_index]
        with tf.GradientTape() as tape:
            tape.watch(conv_output)
            grads = tape.gradient(class_channel, conv_output)
            if grads is None:
                raise ModelUnavailableError(
                    "Attention visualization could not be computed for this model."
                )
        pooled = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(tf.multiply(pooled[:, tf.newaxis, tf.newaxis, :], conv_output), axis=-1)
        heatmap = tf.squeeze(heatmap, axis=0)
        heatmap = tf.maximum(heatmap, 0)

        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        else:
            heatmap = tf.zeros_like(heatmap)

        target = (self.input_size, self.input_size)
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], target, method="bilinear")
        return np.asarray(heatmap[..., 0], dtype=np.float32)

    def is_loaded(self) -> bool:
        return self._model is not None

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._submodel = None