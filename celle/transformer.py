# Adeethyia SUPER TODO: can't have a "protein localization transformer" without a transformer...
import tensorflow as tf
from einops import rearrange
from collections import deque
from functools import partial
from itertools import cycle, islice

from celle.reversible import ReversibleSequence, SequentialSequence
from celle.attention import Attention, SparseConvCausalAttention,
SparseAxialCausalAttention
from rotary_embedding_tensorflow import RotaryEmbedding, broadcat

# Helpers
def exists(val): return val is not None
def default(val, d): return val if exists(val) else d
def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    elif isinstance(val, tuple):
        return val
    else:
        return (val,) * depth


class DivideMax(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        maxes = tf.stop_gradient(tf.reduce_max(
            x, axis=self.dim, keepdims=True))
        return x / maxes


class NonCached(tf.keras.layers.Layer):
    """
    Wrapper for layers that don't themselves support inference cache
    Reconstructs full sequence before the layer and
    cuts suffix of outputs after layer
    """
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def call(self, x, *, cache=None, cache_key=None, **kwargs):
        # x has shape [batch, seq_length, ...], slice along axis -2
        n = tf.shape(x)[-2]
        if exists(cache):
            if cache_key in cache:
                x = tf.concat([cache[cache_key], x], axis=-2)
            cache[cache_key] = x
        out = self.fn(x, **kwargs)
        # Last n elements along the sequence dimension
        return out[..., -n:, :]


class CachedAs(tf.keras.layers.Layer):
    """
    Wrapper that defines a key for the inference cache
    """
    def __init__(self, cache_key, fn, **kwargs):
        super().__init__(**kwargs)
        self.cache_key = cache_key
        self.fn = fn

    def call(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)


class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim, depth, fn, **kwargs):
        super().__init__(**kwargs)
        if depth <= 18:
            init_eps = 0.1
        elif depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        # Create trainable scale with shape [1, 1, dim]
        self.scale = self.add_weight(
            "scale",
            shape=(1, 1, dim),
            initializer=tf.constant_initializer(init_eps),
            trainable=True,
        )
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, sandwich=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        if sandwich:
            self.norm_out = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.norm_out = lambda x: x
        self.fn = fn

    def call(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


class GEGLU(tf.keras.layers.Layer):
    def call(self, x):
        # Split tensor into two halves along the last (feature) axis
        x_part, gates = tf.split(x, num_or_size_splits=2, axis=-1)
        return x_part * tf.nn.gelu(gates)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, dropout=0.0, mult=4.0, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(dim * mult * 2),
            GEGLU(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dim * mult),
        ])
        self.project = tf.keras.layers.Dense(dim)

    def call(self, x, cache=None, cache_key=None):
        x = self.net(x)
        return self.project(x)


class PreShiftToken(tf.keras.layers.Layer):
    def __init__(self, fn, image_size, num_images, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        img_seq_len = ((image_size // num_images) ** 2) * num_images
        self.text_len = seq_len - img_seq_len + 1

    def call(self, x, cache=None, cache_key=None, **kwargs):
        seq_len = self.seq_len
        image_size = self.image_size
        text_len = self.text_len

        if exists(cache) and cache_key in cache:
            offset = cache["offset"]
            if offset >= text_len:
                raise ValueError("cached inference for text is not supported")
            q = cache[cache_key]
            # q should be a deque of length equal to image_size
            assert isinstance(q, deque) and len(q) == image_size

            # Split the last token into 4 parts along the feature axis
            x_last = x[:, -1]
            splits = tf.split(x_last, num_or_size_splits=4, axis=-1)
            x_top, x_left = splits[0], splits[1]
            x_pass = splits[2:]
            q.append((x_top, x_left))
            x_top_new = q.popleft()[0]
            x_left_new = q[-2][1]
            if (offset - text_len) % image_size == 0:
                x_left_new = tf.zeros_like(x_left_new)
            x_concat = tf.concat([x_top_new, x_left_new] + list(x_pass),
                                 axis=-1)
            return self.fn(tf.expand_dims(x_concat, axis=1), cache=cache,
                           **kwargs)

        n = tf.shape(x)[1]
        padding = seq_len - n + 1
        if n < text_len:
            return self.fn(x, **kwargs)

        # Split into text and image tokens
        x_text = x[:, :text_len]
        x_img = x[:, text_len:]
        x_img = tf.pad(x_img, paddings=[[0, 0], [0, padding], [0, 0]])
        x_img = rearrange(x_img, "b (h w) d -> b h w d", h=image_size)

        # Token shift for image tokens: split into 4 parts along feature axis
        splits = tf.split(x_img, num_or_size_splits=4, axis=-1)
        x_img_shift_top, x_img_shift_left = splits[0], splits[1]
        x_img_pass = splits[2:]
        # Shift left: prepend one row of zeros and remove the last row
        x_img_shift_left = tf.concat([
            tf.zeros_like(x_img_shift_left[:, :1, :, :]),
            x_img_shift_left[:, :-1, :, :]], axis=1)
        # Shift top: prepend one column of zeros and remove the last column
        x_img_shift_top = tf.concat(
            [tf.zeros_like(x_img_shift_top[:, :, :1, :]),
            x_img_shift_top[:, :, :-1, :]],
            axis=2
        )
        x_img_shifted = tf.concat([
            x_img_shift_top, x_img_shift_left] + list(x_img_pass),
            axis=-1)
        x_img_merged = rearrange(x_img_shifted, "b h w d -> b (h w) d")
        x_img_merged = x_img_merged[:, :-padding]
        x_combined = tf.concat([x_text, x_img_merged], axis=1)

        if exists(cache):
            dummy_splits = tf.split(x[:, -1], num_or_size_splits=4, axis=-1)
            dummy_top = tf.zeros_like(dummy_splits[0])
            dummy_left = tf.zeros_like(dummy_splits[1])
            q = deque()
            x_img_part = x_img_merged[:, -image_size:]
            r = tf.shape(x_img_part)[1]
            for _ in range(image_size - r):
                q.append((dummy_top, dummy_left))
            for i in range(r):
                token_splits = tf.split(x_img_part[:, i],
                                        num_or_size_splits=4, axis=-1)
                q.append((token_splits[0], token_splits[1]))
            cache[cache_key] = q

        return self.fn(x_combined, cache=cache, **kwargs)


class Transformer(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        reversible=False,
        causal=True,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        attn_types=None,
        image_fmap_size=None,
        num_images=None,
        stable=False,
        sandwich_norm=False,
        shift_tokens=False,
        rotary_emb=True,
        shared_attn_ids=None,
        shared_ff_ids=None,
        optimize_for_inference=False,  # use cache-friendly masked attention
        **kwargs
    ):
        super().__init__(**kwargs)
        layers = []
        self.seq_len = seq_len
        self.image_fmap_size = image_fmap_size

        attn_types = default(attn_types, ("full",))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        shared_attn_ids = cycle(default(shared_attn_ids, list(range(depth))))
        shared_ff_ids = cycle(default(shared_ff_ids, list(range(depth))))
        shared_attn_layers = {}
        shared_ff_layers = {}

        for ind, attn_type, attn_id, ff_id in zip(
            range(depth), attn_type_layer, shared_attn_ids, shared_ff_ids
        ):
            if attn_type == "full":
                attn_class = partial(Attention, stable=stable)
            elif attn_type == "axial_row":
                if optimize_for_inference:
                    attn_class = partial(
                        Attention,
                        stable=stable,
                        static_mask=self._get_attention_mask(attn_type),
                    )
                else:
                    attn_class = partial(
                        SparseAxialCausalAttention,
                        seq_len=seq_len,
                        axis=0,
                        image_size=image_fmap_size,
                        stable=stable,
                    )
            elif attn_type == "axial_col":
                if optimize_for_inference:
                    attn_class = partial(
                        Attention,
                        stable=stable,
                        static_mask=self._get_attention_mask(attn_type),
                    )
                else:
                    attn_class = partial(
                        SparseAxialCausalAttention,
                        seq_len=seq_len,
                        axis=1,
                        image_size=image_fmap_size,
                        stable=stable,
                    )
            elif attn_type == "conv_like":
                attn_class = partial(
                    SparseConvCausalAttention,
                    seq_len=seq_len,
                    image_size=image_fmap_size,
                    stable=stable,
                )
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            attn, reused_attn_type = shared_attn_layers.get(attn_id, (None, None))
            if not exists(attn):
                attn = attn_class(
                    dim,
                    causal=causal,
                    seq_len=seq_len,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=attn_dropout,
                )
                shared_attn_layers[attn_id] = (attn, attn_type)
            elif attn_type != reused_attn_type:
                raise ValueError(
                    "attn_types do not match shared_attn_ids "
                    f'(ind = {ind}, attn_type = "{attn_type}", '
                    f'reused_attn_type = "{reused_attn_type}")'
                )

            ff = shared_ff_layers.get(ff_id)
            if not exists(ff):
                ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                shared_ff_layers[ff_id] = ff

            if isinstance(attn, Attention):
                attn = CachedAs(f"attn_{ind}", attn)
            else:
                # At the moment, other attention classes don't support cache
                attn = NonCached(attn)

            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            if shift_tokens:
                attn = CachedAs(
                    f"preshift_attn_{ind}",
                    PreShiftToken(
                        attn,
                        image_size=image_fmap_size,
                        num_images=num_images,
                        seq_len=seq_len,
                    ),
                )
                ff = CachedAs(
                    f"preshift_ff_{ind}",
                    PreShiftToken(
                        ff,
                        image_size=image_fmap_size,
                        num_images=num_images,
                        seq_len=seq_len,
                    ),
                )

            layers.append(
                [
                    LayerScale(
                        dim,
                        ind + 1,
                        PreNorm(dim, attn, sandwich=sandwich_norm),
                    ),
                    LayerScale(
                        dim,
                        ind + 1,
                        PreNorm(dim, ff, sandwich=sandwich_norm),
                    ),
                ]
            )

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        route_all = ((True, True),) * depth
        attn_route_map = {
            "mask": route_attn,
            "rotary_pos_emb": route_attn,
            "cache": route_all,
        }
        self.layers = execute_type(layers, args_route=attn_route_map)

        # Generate positional embeddings for rotary
        pos_emb = None
        if rotary_emb:
            rot_dim = dim_head // 3
            img_seq_len = ((image_fmap_size // num_images) ** 2) * num_images
            text_len = seq_len - img_seq_len + 1

            text_pos_emb = RotaryEmbedding(dim=rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim=rot_dim, freqs_for="pixel")

            text_freqs = text_pos_emb(tf.range(text_len))
            img_to_text_freqs = text_pos_emb(tf.fill([img_seq_len], 8192))
            text_freqs = tf.concat([text_freqs, img_to_text_freqs], axis=0)

            img_freqs_axial = img_axial_pos_emb(
                tf.linspace(-1.0, 1.0, image_fmap_size)
            )

            if num_images > 1:
                split_img_freqs_axial = tf.split(
                    img_freqs_axial,
                    num_or_size_splits=image_fmap_size // num_images,
                    axis=0,
                )
                split_img_freqs = [
                    broadcat(
                        (
                            rearrange(
                                img_freqs_axial_per_image, "i d -> i () d"
                            ),
                            rearrange(
                                img_freqs_axial_per_image, "j d -> () j d"
                            ),
                        ),
                        dim=-1,
                    )
                    for img_freqs_axial_per_image in split_img_freqs_axial
                ]
                split_img_freqs = [
                    rearrange(img_freqs_per_image, "h w d -> (h w) d")
                    for img_freqs_per_image in split_img_freqs
                ]
                img_freqs = tf.concat(split_img_freqs, axis=0)
            elif num_images == 1:
                img_freqs = broadcat(
                    (
                        rearrange(img_freqs_axial, "i d -> i () d"),
                        rearrange(img_freqs_axial, "j d -> () j d"),
                    ),
                    dim=-1,
                )
                img_freqs = rearrange(img_freqs, "h w d -> (h w) d")
            else:
                raise ValueError("num_images must be int greater than 0")

            self.img_axial_pos_emb = img_axial_pos_emb
            self.text_pos_emb = text_pos_emb

            text_axial_freqs = img_axial_pos_emb(tf.fill([text_len], -10.0))
            text_axial_freqs = tf.concat(
                [text_axial_freqs, text_axial_freqs], axis=-1
            )

            img_freqs = tf.concat([text_axial_freqs, img_freqs], axis=0)
            pos_emb = tf.concat([text_freqs, img_freqs], axis=-1)
            pos_emb = rearrange(pos_emb, "n d -> () n d")
        self.pos_emb = pos_emb  # stored as a constant-like attribute

    def call(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb=self.pos_emb, **kwargs)

    def _get_attention_mask(self, attn_type):
        img_seq_len = self.image_fmap_size ** 2
        text_len = self.seq_len + 1 - img_seq_len
        static_mask = tf.zeros((self.seq_len, self.seq_len), dtype=tf.bool)
        mask_update = tf.ones((self.seq_len, text_len), dtype=tf.bool)
        static_mask = tf.concat([mask_update, static_mask[:, text_len:]], axis=1)
        if attn_type == "axial_row":
            for row in range(self.image_fmap_size):
                begin = text_len + row * self.image_fmap_size
                end = text_len + (row + 1) * self.image_fmap_size
                for i in range(begin, end):
                    for j in range(begin, end):
                        static_mask = tf.tensor_scatter_nd_update(
                            static_mask, [[i, j]], [True]
                        )
        elif attn_type == "axial_col":
            for col in range(self.image_fmap_size):
                begin = text_len + col
                idxs = [
                    [i, j]
                    for i in range(
                        begin, self.seq_len, self.image_fmap_size
                    )
                    for j in range(
                        begin, self.seq_len, self.image_fmap_size
                    )
                ]
                for idx in idxs:
                    static_mask = tf.tensor_scatter_nd_update(
                        static_mask, [idx], [True]
                    )
        else:
            raise ValueError(
                f'attention type "{attn_type}" can\'t be simulated with a '
                "static mask"
            )
        return static_mask