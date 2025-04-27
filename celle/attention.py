# TODO: IMPOTANT FUNCTIONS
# Import functions 

from inspect import isfunction

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers
import tensorflow.nn.functional as tff

from einops import rearrange, repeat

# rotary_embedding_torch is not part of PyTorch, so we keep it as is.
from rotary_embedding_torch import apply_rotary_emb





# Helper functions

def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def stable_softmax(t, dim=-1, alpha=32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))

