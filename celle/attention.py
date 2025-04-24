# Adeethyia TODO: write Attention(s)
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# Helpers
def exists(val): return val is not None
def uniq(arr): return list(set(el for el in arr))
def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def max_neg_value(t):
    dtype = t.dtype.as_numpy_dtype
    return -tf.constant(np.finfo(dtype).max, t.dtype)
