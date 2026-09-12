"""Candidate architectures for the improved brain-tumor classifier.

Four families are evaluated on the held-out test set before one is selected:

* an improved custom CNN (batch norm + dropout, addressing the classic
  Conv2Dx2 overfitting seen in the legacy model)
* EfficientNetB0 / B1
* ResNet50
* MobileNetV2
* DenseNet121

Transfer-learning models load ImageNet weights with a fresh classification
head (global average pooling -> dropout -> softmax). The base weights are
frozen for the head-only training phase and optionally fine-tuned afterwards.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

N_CLASSES = 4

ARCHITECTURES = {
    "efficientnet_b0",
    "resnet50",
    "mobilenet_v2",
    "densenet121",
    "custom_cnn",
}


def _classification_head(features, dropout_rate: float) -> keras.Model:
    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(N_CLASSES, activation="softmax")(x)
    return x


def build_custom_cnn(input_size: int = 64, dropout_rate: float = 0.35) -> keras.Model:
    """Improved custom CNN: conv blocks with batch-norm, pooling and dropout."""
    inputs = keras.Input(shape=(input_size, input_size, 3))

    def conv_block(x, filters, pool=True):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)
        return x

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128, pool=False)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_learning(
    architecture: str,
    input_size: int = 128,
    dropout_rate: float = 0.4,
    frozen: bool = True,
) -> keras.Model:
    """Build a transfer-learning model with a fresh classification head."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    build_map = {
        "efficientnet_b0": lambda: keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3)
        ),
        "resnet50": lambda: keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3)
        ),
        "mobilenet_v2": lambda: keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3)
        ),
        "densenet121": lambda: keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3)
        ),
    }

    base = build_map[architecture]()
    base.trainable = not frozen
    x = _classification_head(base.output, dropout_rate)
    model = keras.Model(base.input, x, name=architecture)

    if frozen:
        base.trainable = False
    return model


def build(architecture: str, input_size: int = 128, dropout_rate: float = 0.4) -> keras.Model:
    if architecture == "custom_cnn":
        return build_custom_cnn(input_size=input_size, dropout_rate=dropout_rate)
    return build_transfer_learning(
        architecture, input_size=input_size, dropout_rate=dropout_rate, frozen=True
    )