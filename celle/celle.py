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

def typical(scores, mass=0.9, filter_value=-float("inf"), min_tokens_to_keep=1):
    normalized = tf.nn.log_softmax(scores, axis=-1)
    p = tf.exp(normalized)
    ent = -tf.reduce_sum(normalized * p, axis=-1, keepdims=True)
    shifted_scores = tf.abs(-normalized - ent)
    # Sort scores in ascending order.
    sorted_scores = tf.sort(shifted_scores, direction='ASCENDING', axis=-1)
    sorted_indices = tf.argsort(shifted_scores, direction='ASCENDING', axis=-1)
    sorted_logits = tf.gather(scores, sorted_indices, batch_dims=1)
    sorted_probs = tf.nn.softmax(sorted_logits, axis=-1)
    cumulative_probs = tf.cumsum(sorted_probs, axis=-1)
    bool_mask = tf.cast(cumulative_probs < mass, tf.int32)
    last_ind = tf.reduce_sum(bool_mask, axis=-1)
    # Gather kth sorted score for each batch.
    kth_sorted_score = tf.gather(sorted_scores, last_ind, batch_dims=1)
    kth_sorted_score = tf.expand_dims(kth_sorted_score, axis=-1)
    sorted_indices_to_remove = sorted_scores > kth_sorted_score
    if min_tokens_to_keep > 1:
        prefix = tf.zeros([tf.shape(sorted_indices_to_remove)[0],
                           min_tokens_to_keep], dtype=tf.bool)
        sorted_indices_to_remove = tf.concat(
            [prefix, sorted_indices_to_remove[:, min_tokens_to_keep:]], axis=-1)
    # Unsort: compute inverse permutation.
    inv_perm = tf.argsort(sorted_indices, axis=-1, stable=True)
    indices_to_remove = tf.gather(sorted_indices_to_remove, inv_perm, batch_dims=1)
    return tf.where(indices_to_remove,
                    tf.fill(tf.shape(scores), filter_value),
                    scores)

# Straight-through Gumbel softmax
def tf_gumbel_softmax(logits, tau, hard, axis):
    noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    y = tf.nn.softmax((logits + noise) / tau, axis=axis)
    if hard:
        y_hard = tf.one_hot(tf.argmax(y, axis=axis), depth=tf.shape(y)[axis])
        y = tf.stop_gradient(y_hard - y) + y
    return y

# SharedEmbedding: uses a Dense layer’s kernel as the embedding table.
class SharedEmbedding(tf.keras.layers.Layer):
    def __init__(self, linear, start_index, end_index, **kwargs):
        super().__init__(**kwargs)
        self.linear = linear
        self.start_index = start_index
        self.end_index = end_index
        self.emb_dim = self.linear.kernel.shape[1]

    def call(self, inputs):
        emb_weight = self.linear.kernel[self.start_index:self.end_index]
        return tf.nn.embedding_lookup(emb_weight, inputs)

# ModelExtender: wraps a pretrained model.
class ModelExtender(tf.keras.layers.Layer):
    def __init__(self, vocab, out_features, fixed_embedding=False, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        if vocab == "unirep":
            # TODO: fill these 3 out
            self.model = ...
            in_features = 1900
        elif vocab == "bert":
            self.model = ...
            in_features = 768
        elif vocab == "esm1b":
            self.model = ...
            in_features = 33
        self.out_features = out_features
        self.scale_layer = tf.keras.layers.Dense(out_features)
        self.fixed_embedding = fixed_embedding
        if self.fixed_embedding:
            self.model.trainable = False

    def call(self, x, training=False):
        if self.fixed_embedding:
            x = tf.stop_gradient(self.model(x))
        else:
            x = self.model(x)
        if ((self.vocab == "unirep" and self.out_features != 1900) or
            (self.vocab == "bert" and self.out_features != 768) or
            (self.vocab == "esm1b" and self.out_features != 1280)):
            x = self.scale_layer(x)
        return x

# Discrete VAE components
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, chan, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(chan, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(chan, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(chan, kernel_size=1, padding='same'),
        ])

    def call(self, x):
        return self.net(x) + x



