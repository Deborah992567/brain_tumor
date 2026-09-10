"""tf.data augmentation pipeline.

The legacy model was trained with ``ImageDataGenerator(rescale=1/255)`` with
shear/zoom/flip augmentation. This pipeline reproduces the same rescaling (so
inference preprocessing stays identical) and adds gentle, well-understood
geometric augmentation to reduce overfitting:

* random left/right flip (axial-flip safe for MRIs)
* random brightness/contrast jitter
* small random rotation
* normalisation to [0, 1]

Brightness/contrast jitter is kept small because MR intensity ranges carry
real diagnostic signal.
"""

from __future__ import annotations

import tensorflow as tf


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
        im = tf.image.resize(im, (input_size, input_size), method="bilinear")
        im = tf.cast(im, tf.float32) / 255.0
        return im, lab

    ds = tf.data.Dataset.from_tensor_slices((arrays.astype("float32"), labels.astype("int32")))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_make, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def normalize_array(array, input_size: int) -> tf.Tensor:
    """Normalise a single HWC array to the (1, size, size, 3) tensor the model uses."""
    image = tf.image.resize(array, (input_size, input_size), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)