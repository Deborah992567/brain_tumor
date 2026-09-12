"""tf.data feeding pipeline.

The legacy model was trained with ``ImageDataGenerator(rescale=1/255)`` with
shear/zoom/flip augmentation. The canonical preprocessing (RGB -> bilinear
resize -> ``1/255``) lives in ``app.core.preprocessing`` and is applied here
identically to the inference service, guaranteeing the model sees the same
input distribution in both paths. Small, clinically-defensible augmentations
are applied on top of it after normalisation:

* random left/right flip (axial-flip safe for MRIs)
* random brightness/contrast jitter

Brightness/contrast jitter is kept small because MR intensity ranges carry
real diagnostic signal.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from app.core.preprocessing import preprocess_pixels


def _augment_train(image: tf.Tensor, label: tf.Tensor):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def feature_pipeline(
    arrays,
    labels,
    batch_size: int,
    input_size: int,
    augment: bool,
    shuffle_buffer: int = 512,
    seed: int = 42,
):
    def _make(im, lab):
        scaled = tf.numpy_function(
            lambda arr: preprocess_pixels(arr, input_size),
            [im],
            tf.float32,
            name="canonical_preprocess",
        )
        scaled.set_shape((input_size, input_size, 3))
        return scaled, lab

    ds = tf.data.Dataset.from_tensor_slices((arrays.astype("uint8"), labels.astype("int32")))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_make, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def normalize_array(array, input_size: int) -> tf.Tensor:
    """Normalise a single HWC array using the shared canonical preprocessing."""
    tensor = preprocess_pixels(array, input_size)
    return tf.expand_dims(tf.convert_to_tensor(tensor, dtype=tf.float32), axis=0)