# Adeethyia TODO: idk what this file is even supposed to do
from math import log2, sqrt
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from einops import rearrange

from celle.vae import VQGanVAE
from celle.transformer import Transformer, DivideMax
import csv

# Helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val

def is_empty(t):
    # Returns True if tensor has no elements.
    return tf.equal(tf.size(t), 0)

def masked_mean(t, mask, dim=1):
    # Compute an average over dim while ignoring values where mask is False.
    mask_exp = tf.cast(tf.expand_dims(mask, axis=-1), t.dtype)
    masked = tf.where(tf.equal(mask_exp, 0), tf.zeros_like(t), t)
    return tf.reduce_sum(masked, axis=dim) / tf.cast(tf.reduce_sum(mask, axis=dim, keepdims=True), t.dtype)

def set_requires_grad(model, value):
    # In TensorFlow the notion of "requires_grad" is handled by trainable flags.
    model.trainable = value

def eval_decorator(fn):
    # In TF we often pass a "training" flag; here we simply wrap the call.
    def inner(model, *args, **kwargs):
        # Force training=False for evaluation
        return fn(model, *args, **kwargs)
    return inner

# Sampling helpers
def log(t, eps=1e-20):
    return tf.math.log(t + eps)

def gumbel_noise(t):
    noise = tf.random.uniform(tf.shape(t), minval=0., maxval=1.)
    return -log(-log(noise))

def gumbel_sample(t, temperature=0.9, axis=-1):
    noisy = (t / temperature) + gumbel_noise(t)
    return tf.argmax(noisy, axis=axis)

def top_k(logits, thres=0.5):
    num_logits = tf.shape(logits)[-1]
    # Compute k as a scalar tensor (ensure at least 1).
    k_float = (1. - thres) * tf.cast(num_logits, tf.float32)
    k = tf.maximum(tf.cast(tf.math.round(k_float), tf.int32), 1)
    topvals, _ = tf.math.top_k(logits, k=k)
    kth_value = tf.reduce_min(topvals, axis=-1, keepdims=True)
    return tf.where(logits >= kth_value, logits,
                    tf.fill(tf.shape(logits), -float("inf")))

#...typical