# Adeethyia TODO: write Attention(s)
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from rotary_embedding_tensorflow import apply_rotary_emb

## Helpers
def exists(val): return val is not None
def uniq(arr): return list(set(el for el in arr))
def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def max_neg_value(t):
    dtype = t.dtype.as_numpy_dtype
    return -tf.constant(np.finfo(dtype).max, t.dtype)

def apply_pos_emb(pos_emb, qkv):
    # qkv is a tuple of tensors (q, k, v)
    # Assume their penultimate dimension is the “sequence” dimension.
    n = tf.shape(qkv[0])[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(apply_rotary_emb(pos_emb, t) for t in qkv)

## Classes
class Attention(layers.Layer):
    def __init__(self,
                 dim,
                 seq_len,
                 causal=True,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 stable=False,
                 static_mask=None,
                 **kwargs):
        """
        Implements full attention. In this version:
          • `to_qkv` projects input into query, key, and value.
          • Optionally applies rotary position embeddings.
          • Supports caching (for autoregressive generation).
        """
        super().__init__(**kwargs)
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5
        self.stable = stable
        self.causal = causal
        # Save static_mask as a non-trainable constant (if provided)
        if static_mask is not None:
            self.static_mask = tf.constant(static_mask, dtype=tf.bool)
        else:
            self.static_mask = None
        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
        # Identity function; you might want to store/inspect attention weights here.
        self.save_attn = lambda x: x

    def call(self, x, mask=None, rotary_pos_emb=None, cache=None, cache_key=None):
        # x: [batch, n, dim]
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        h = self.heads

        offset = 0 if cache is None else cache.get("offset", 0)

        # Project to queries, keys, values then split into 3 parts.
        qkv = self.to_qkv(x)  # shape: [b, n, inner_dim*3]
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)  # each: [b, n, inner_dim]

        def reshape_to_heads(t):
            # from [b, n, h*d] to [b, heads, n, d]
            t = tf.reshape(t, [b, n, h, -1])
            return tf.transpose(t, perm=[0, 2, 1, 3])
        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        if exists(rotary_pos_emb):
            # Apply rotary embeddings; note we slice starting from "offset"
            q, k, v = apply_pos_emb(rotary_pos_emb[..., offset:, :], (q, k, v))
        q = q * self.scale

        if offset > 0 and cache is not None:
            k_top, v_top = cache[cache_key]
            # Concatenate along the sequence dimension (axis=2)
            k = tf.concat([k_top, k], axis=2)
            v = tf.concat([v_top, v], axis=2)
        if cache is not None:
            cache[cache_key] = (k, v)

        # Compute dot-product attention weights.
        dots = tf.einsum("bhid,bhjd->bhij", q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            # mask: [b, j] → [b, 1, 1, j]
            mask_exp = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
            dots = tf.where(tf.logical_not(mask_exp), mask_value, dots)

        if self.causal and offset == 0:
            # Create a causal mask. Here we mimic torch.triu_(j - i + 1)
            i = tf.shape(dots)[-2]
            j = tf.shape(dots)[-1]
            idx_i = tf.range(i)[:, None]
            idx_j = tf.range(j)[None, :]
            causal_mask = idx_j >= (idx_i + (j - i + 1))
            causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)  # shape [1, 1, i, j]
            dots = tf.where(causal_mask, mask_value, dots)

        if exists(self.static_mask):
            # self.static_mask assumed shape [seq_len, seq_len]
            static_mask_slice = self.static_mask[offset:offset + n, offset:offset + n]
            static_mask_slice = tf.expand_dims(tf.expand_dims(static_mask_slice, 0), 0)
            dots = tf.where(tf.logical_not(static_mask_slice), mask_value, dots)

        attn = tf.nn.softmax(dots)
        _ = self.save_attn(attn)

        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        # Rearrange back from [b, heads, n, d] to [b, n, heads * d]
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        b, n, h, d = tf.shape(out)[0], tf.shape(out)[1], tf.shape(out)[2], tf.shape(out)[3]
        out = tf.reshape(out, [b, n, h * d])
        return self.to_out(out)


class SparseConvCausalAttention(layers.Layer):
    def __init__(self,
                 dim,
                 seq_len,
                 image_size=32,
                 kernel_size=5,
                 dilation=1,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 stable=False,
                 **kwargs):
        """
        Sparse attention with a convolutional pattern.
        Note: uses tf.image.extract_patches as a replacement for PyTorch’s unfold
        """
        super().__init__(**kwargs)
        if kernel_size % 2: raise ValueError("kernel size must be odd")
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stable = stable
        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, mask=None, rotary_pos_emb=None):
        # x: [b, n, dim]
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        h = self.heads
        img_size = self.image_size
        kernel_size = self.kernel_size
        dilation = self.dilation
        seq_len = self.seq_len

        img_seq_len = img_size * img_size
        # text tokens: seq_len + 1 - image tokens
        text_len = seq_len + 1 - img_seq_len

        # Pad x to length (seq_len - n + 1) if needed.
        pad_len = seq_len - n + 1
        if pad_len > 0:
            x = tf.pad(x, paddings=[[0, 0], [0, pad_len], [0, 0]], constant_values=0)

        # Prepare mask.
        if mask is None:
            mask = tf.ones((b, text_len), dtype=tf.bool)
        else:
            mask = mask[:, :text_len]

        # Project to queries, keys, values.
        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        def reshape_for_heads(t):
            # from [b, n, h*d] → merge batch and heads: [b*h, n, d]
            t = tf.reshape(t, [b, -1, h, t.shape[-1] // h])
            t = tf.transpose(t, perm=[0, 2, 1, 3])
            return tf.reshape(t, [b * h, -1, t.shape[-1]])
        q = reshape_for_heads(q)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        q = q * self.scale

        # Split into text and image portions.
        q_text, q_img = q[:, :-img_seq_len, :], q[:, -img_seq_len:, :]
        k_text, k_img = k[:, :-img_seq_len, :], k[:, -img_seq_len:, :]
        v_text, v_img = v[:, :-img_seq_len, :], v[:, -img_seq_len:, :]

        # Text attention.
        dots_text = tf.einsum("bid,bjd->bij", q_text, k_text)
        mask_value = max_neg_value(dots_text)
        i = tf.shape(dots_text)[-2]
        j = tf.shape(dots_text)[-1]
        idx_i = tf.range(i)[:, None]
        idx_j = tf.range(j)[None, :]
        text_causal_mask = idx_j >= (idx_i + (j - i + 1))
        text_causal_mask = tf.cast(text_causal_mask, tf.bool)
        dots_text = tf.where(text_causal_mask, mask_value, dots_text)
        attn_text = tf.nn.softmax(dots_text, axis=-1)
        out_text = tf.einsum("bij,bjd->bid", attn_text, v_text)

        # Image attention for the image tokens; "convolutional" attention
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        pad_val = effective_kernel_size // 2

        # Helper using tf.image.extract_patches.
        def extract_patches_fn(tensor):
            # tensor: [b*h, img_seq_len, d] --> reshape to [b*h, img_size, img_size, d]
            d_val = tensor.shape[-1]
            tensor_reshaped = tf.reshape(tensor, [b * h, img_size, img_size, d_val])
            patches = tf.image.extract_patches(
                images=tensor_reshaped,
                sizes=[1, kernel_size, kernel_size, 1],
                strides=[1, 1, 1, 1],
                rates=[1, dilation, dilation, 1],
                padding='SAME'
            )
            # patches: [b*h, img_size, img_size, kernel_size*kernel_size*d]
            patches = tf.reshape(patches, [b * h, img_seq_len, kernel_size * kernel_size, d_val])
            return patches

        k_img_patches = extract_patches_fn(k_img)
        v_img_patches = extract_patches_fn(v_img)

        # Let image tokens attend to all text
        dots_image = tf.einsum("bid,bijd->bij", q_img, k_img_patches)
        dots_image_to_text = tf.einsum("bid,bjd->bij", q_img, k_text)

        # Causal mask for local convolution
        i_img = tf.shape(dots_image)[-2]
        j_img = tf.shape(dots_image)[-1]
        # Index patches from image sequence
        k_img_indices = tf.reshape(tf.cast(tf.range(img_seq_len), tf.float32), [img_size, img_size])
        k_img_indices = tf.expand_dims(k_img_indices, axis=0)  # [1, img_size, img_size]
        k_img_indices = tf.pad(k_img_indices,
                               paddings=[[0, 0], [pad_val, pad_val], [pad_val, pad_val]],
                               constant_values=tf.cast(img_seq_len, tf.float32))
        k_img_indices_patches = tf.image.extract_patches(
            images=tf.expand_dims(k_img_indices, axis=-1),
            sizes=[1, kernel_size, kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, dilation, dilation, 1],
            padding='VALID'
        )
        k_img_indices_patches = tf.reshape(k_img_indices_patches, [1, img_seq_len, kernel_size * kernel_size])
        q_img_indices = tf.reshape(tf.cast(tf.range(img_seq_len), tf.float32), [1, img_seq_len, 1])
        causal_mask = tf.less(q_img_indices, k_img_indices_patches)
        causal_mask = tf.tile(causal_mask, [b * h, 1, 1])

        # Prepare text mask for image tokens.
        mask_new = tf.tile(tf.expand_dims(mask, axis=1), [1, tf.shape(q_img)[1], 1])
        mask_new = tf.reshape(tf.tile(mask_new, [h, 1, 1]), [b * h, tf.shape(q_img)[1], -1])
        full_mask = tf.concat([tf.logical_not(mask_new), causal_mask], axis=-1)

        dots = tf.concat([dots_image_to_text, dots_image], axis=-1)
        dots = tf.where(full_mask, mask_value, dots)
        attn = tf.nn.softmax(dots, axis=-1)

        attn_image_to_text = attn[:, :, :text_len]
        attn_image = attn[:, :, text_len:]
        out_image_to_image = tf.einsum("bij,bijd->bid", attn_image, v_img_patches)
        out_image_to_text = tf.einsum("bij,bjd->bid", attn_image_to_text, v_text)
        out_image = out_image_to_image + out_image_to_text

        # Combine text and image outputs.
        out = tf.concat([out_text, out_image], axis=1)

        # Reshape back: first, restore heads then merge.
        # Here, out is [b*h, n, d]. Reshape to [b, h, n, d] then combine the heads.
        d_val = tf.shape(out)[-1]
        out = tf.reshape(out, [b, h, -1, d_val])
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, [b, -1, h * d_val])
        return self.to_out(out)[:, :n]


class SparseAxialCausalAttention(layers.Layer):
    def __init__(self,
                 dim,
                 seq_len,
                 image_size=32,
                 axis=0,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 stable=False,
                 **kwargs):
        """
        Sparse axial attention where the attention is applied along one axis (height or width).
        """
        super().__init__(**kwargs)
        if axis not in [0, 1]:
            raise ValueError("axis must be either 0 (along height) or 1 (along width)")
        self.axis = axis
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.stable = stable
        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, mask=None, rotary_pos_emb=None):
        # x: [b, n, dim]
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        h = self.heads
        img_size = self.image_size
        axis = self.axis
        seq_len = self.seq_len

        img_seq_len = img_size * img_size
        text_len = seq_len + 1 - img_seq_len

        pad_len = seq_len - n + 1
        if pad_len > 0:
            x = tf.pad(x, paddings=[[0, 0], [0, pad_len], [0, 0]], constant_values=0)

        if mask is None:
            mask = tf.ones((b, text_len), dtype=tf.bool)
        else:
            mask = mask[:, :text_len]

        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        def reshape_for_heads(t):
            t = tf.reshape(t, [b, -1, h, t.shape[-1] // h])
            t = tf.transpose(t, perm=[0, 2, 1, 3])
            return tf.reshape(t, [b * h, -1, t.shape[-1]])
        q = reshape_for_heads(q)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        q = q * self.scale

        q_text, q_img = q[:, :-img_seq_len, :], q[:, -img_seq_len:, :]
        k_text, k_img = k[:, :-img_seq_len, :], k[:, -img_seq_len:, :]
        v_text, v_img = v[:, :-img_seq_len, :], v[:, -img_seq_len, :]

        dots_text = tf.einsum("bid,bjd->bij", q_text, k_text)
        mask_value = max_neg_value(dots_text)
        i = tf.shape(dots_text)[-2]
        j = tf.shape(dots_text)[-1]
        idx_i = tf.range(i)[:, None]
        idx_j = tf.range(j)[None, :]
        text_causal_mask = idx_j >= (idx_i + (j - i + 1))
        text_causal_mask = tf.cast(text_causal_mask, tf.bool)
        dots_text = tf.where(text_causal_mask, mask_value, dots_text)
        attn_text = tf.nn.softmax(dots_text, axis=-1)
        out_text = tf.einsum("bij,bjd->bid", attn_text, v_text)

        # For axial attention, we first split the image tokens along the specified axis.
        if axis == 0:
            # reshape: "b (h w) c -> b h w c"
            def split_axis(t):
                return tf.reshape(t, [b, img_size, img_size, -1])
            q_img = split_axis(q_img)
            k_img = split_axis(k_img)
            v_img = split_axis(v_img)
        else:  # axis == 1
            def split_axis(t):
                t = tf.reshape(t, [b, img_size, img_size, -1])
                return tf.transpose(t, perm=[0, 2, 1, 3])
            q_img = split_axis(q_img)
            k_img = split_axis(k_img)
            v_img = split_axis(v_img)

        # Compute similarities.
        dots_image_to_image = tf.einsum("bxi d, bxj d->bxi j", q_img, k_img)
        dots_image_to_text = tf.einsum("bxi d, bjd->bxi j", q_img, k_text)
        dots = tf.concat([dots_image_to_text, dots_image_to_image], axis=-1)

        # Build a causal mask for the axial dimension.
        bh, x_dim, i_dim, _ = tf.unstack(tf.shape(dots))
        causal_mask = tf.linalg.band_part(tf.ones((i_dim, img_size), dtype=tf.bool), 0, -1)
        causal_mask = tf.tile(tf.expand_dims(causal_mask, 0), [bh, x_dim, 1, 1])
        mask_new = tf.tile(tf.expand_dims(mask, axis=1), [1, x_dim, 1])
        mask_new = tf.reshape(tf.tile(mask_new, [h, 1, 1]), [b * h, x_dim, i_dim, -1])
        full_mask = tf.concat([tf.logical_not(mask_new), causal_mask], axis=-1)

        dots = tf.where(full_mask, mask_value, dots)
        attn = tf.nn.softmax(dots, axis=-1)
        attn_image_to_text = attn[..., :text_len]
        attn_image_to_image = attn[..., text_len:]
        out_image_to_image = tf.einsum("bxi j, bxj d->bxi d", attn_image_to_image, v_img)
        out_image_to_text = tf.einsum("bxi j, bjd->bxi d", attn_image_to_text, v_text)
        out_image = out_image_to_image + out_image_to_text

        # Merge back the axial dimension.
        if axis == 0:
            out_image = tf.reshape(out_image, [b, -1, tf.shape(out_image)[-1]])
        else:
            out_image = tf.transpose(out_image, perm=[0,2,1,3])
            out_image = tf.reshape(out_image, [b, -1, tf.shape(out_image)[-1]])

        out = tf.concat([out_text, out_image], axis=1)
        # Restore heads.
        d_val = tf.shape(out)[-1]
        out = tf.reshape(out, [b, h, -1, d_val])
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, [b, -1, h * d_val])
        return self.to_out(out)[:, :n]
