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
            inputs = None
            try:
                inputs = model.inputs[0]  # type: ignore[index]
            except Exception:
                inputs = None
            if inputs is None:
                inputs = model.layers[0].input  # legacy Sequential models

            if last_conv is not None:
                try:
                    # Legacy Sequential models have no `model.output` until called;
                    # use the final layer's output symbol directly instead.
                    classification_output = getattr(model, "output", None)
                    if classification_output is None:
                        classification_output = model.layers[-1].output
                    submodel = tf.keras.Model(
                        inputs=inputs, outputs=[last_conv.output, classification_output]
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
        if self._submodel is not None:
            try:
                heatmap = self._grad_cam_via_submodel(preprocessed, class_index)
                return heatmap
            except ModelUnavailableError:
                pass  # fall back to the raw tensor forward below
        return self._grad_cam_via_tensor_forward(preprocessed, class_index)

    def _grad_cam_via_submodel(
        self, preprocessed: np.ndarray, class_index: int
    ) -> np.ndarray:
        tf = self._import_tf()
        with tf.GradientTape() as tape:
            conv_output, predictions = self._submodel(preprocessed, training=False)
            class_channel = predictions[:, class_index]
            grads = tape.gradient(class_channel, conv_output)
        if grads is None:
            raise ModelUnavailableError("Grad-CAM via submodel returned no gradients.")
        return self._finalize_heatmap(tf, conv_output, grads)

    def _grad_cam_via_tensor_forward(
        self, preprocessed: np.ndarray, class_index: int
    ) -> np.ndarray:
        """Grad-CAM via a raw tensor forward pass.

        Keras 3 layer calls are wrapped in compiled functions whose variables
        are not ``tf.Variable`` instances, so ``GradientTape`` cannot record
        gradients through normal layer calls. This implementation recomputes
        the CNN forward using ``tf.nn`` ops and the layer weight buffers so the
        tape sees a standard eager graph.
        """
        tf = self._import_tf()
        x = tf.convert_to_tensor(preprocessed)
        with tf.GradientTape() as tape:
            conv_out: object | None = None
            for layer in self._model.layers:
                x, conv_out = self._apply_layer_tensor(layer, x, conv_out)
            logits = x
            if conv_out is None:
                raise ModelUnavailableError(
                    "Attention visualization requires a convolutional layer."
                )
            class_channel = logits[:, class_index]
            grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise ModelUnavailableError(
                "Attention visualization could not be computed for this model."
            )
        return self._finalize_heatmap(tf, conv_out, grads)

    def _apply_layer_tensor(self, layer, x, conv_out):
        """Apply one layer with tf.nn ops, mirroring the Keras forward pass."""
        tf = self._import_tf()
        name = layer.__class__.__name__

        def weight(name_: str):
            for w in layer.weights:
                if w.name.split("/")[-1].split(":")[0] == name_:
                    return w
            raise ModelUnavailableError(
                f"Layer {layer.name} has no '{name_}' weight."
            )

        if name == "Conv2D":
            kernel = weight("kernel")
            stride = tuple(layer.strides)
            padding = "VALID" if layer.padding == "valid" else "SAME"
            # input is NHWC; conv2d expects 4D
            x = tf.nn.conv2d(
                x, kernel, strides=(1, stride[0], stride[1], 1), padding=padding
            )
            if layer.use_bias:
                x = tf.nn.bias_add(x, weight("bias"))
            if layer.activation is not None:
                x = self._apply_activation(layer.activation, x)
            conv_out = x
        elif name == "MaxPooling2D":
            ks = tuple(layer.pool_size)
            st = tuple(layer.strides)
            padding = "VALID" if layer.padding == "valid" else "SAME"
            x = tf.nn.max_pool2d(
                x,
                ksize=(1, ks[0], ks[1], 1),
                strides=(1, st[0], st[1], 1),
                padding=padding,
            )
        elif name == "Flatten":
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        elif name == "Dense":
            kernel = weight("kernel")
            x = tf.matmul(x, kernel)
            if layer.use_bias:
                x = tf.nn.bias_add(x, weight("bias"))
            if layer.activation is not None:
                x = self._apply_activation(layer.activation, x)
        elif name == "Activation":
            x = self._apply_activation(layer.activation, x)
        elif name == "Dropout":
            x = x  # inference: drop nothing
        elif name == "GlobalAveragePooling2D":
            x = tf.reduce_mean(x, axis=(1, 2))
        elif name == "BatchNormalization":
            mean = weight("moving_mean")
            variance = weight("moving_variance")
            gamma = weight("gamma") if any(
                "gamma" in w.name for w in layer.weights
            ) else tf.ones_like(mean)
            beta = weight("beta") if any(
                "beta" in w.name for w in layer.weights
            ) else tf.zeros_like(mean)
            x = (x - mean) / tf.sqrt(variance + float(layer.epsilon))
            x = x * gamma + beta
        else:
            raise ModelUnavailableError(
                f"Layer type '{name}' is not supported for attention visualization."
            )
        return x, conv_out

    @staticmethod
    def _apply_activation(activation, x):
        tf = __import__("tensorflow")
        tf_name = str(getattr(activation, "__name__", "")).lower()
        if tf_name == "softmax":
            return tf.nn.softmax(x)
        if tf_name == "relu":
            return tf.nn.relu(x)
        if tf_name == "sigmoid":
            return tf.nn.sigmoid(x)
        return x

    def _finalize_heatmap(self, tf, conv_output, grads) -> np.ndarray:
        pooled = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(
            tf.multiply(pooled[:, tf.newaxis, tf.newaxis, :], conv_output), axis=-1
        )
        heatmap = tf.maximum(tf.squeeze(heatmap, axis=0), 0)
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