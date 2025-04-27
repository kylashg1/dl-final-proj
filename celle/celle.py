from math import log2, sqrt
import numpy as np

import tensorflow as tf
import keras
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

class DiscreteVAE(tf.keras.Model):
    def __init__(self, image_size=256, num_tokens=512, codebook_dim=512,
                 num_layers=3, num_resnet_blocks=0, hidden_dim=64, channels=3,
                 smooth_l1_loss=False, temperature=0.9,
                 straight_through=False, kl_div_loss_weight=0.0,
                 normalization=((0.5,)*3, (0.5,)*3), **kwargs):
        super().__init__(**kwargs)
        if not log2(image_size).is_integer():
            raise ValueError("image size must be a power of 2")
        if num_layers < 1:
            raise ValueError("number of layers must be >= 1")
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.channels = channels
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = tf.keras.layers.Embedding(num_tokens, codebook_dim)
        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))
        enc_chans = [channels] + enc_chans
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan] + dec_chans

        enc_chans_io = [(enc_chans[i], enc_chans[i+1])
                        for i in range(len(enc_chans)-1)]
        dec_chans_io = [(dec_chans[i], dec_chans[i+1])
                        for i in range(len(dec_chans)-1)]

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv2D(enc_out, kernel_size=4, strides=2,
                                       padding='same'),
                tf.keras.layers.ReLU(),
            ]))
            dec_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(dec_out, kernel_size=4, strides=2,
                                                padding='same'),
                tf.keras.layers.ReLU(),
            ]))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, tf.keras.layers.Conv2D(codebook_dim, kernel_size=1,
                                                        padding='same'))

        enc_layers.append(tf.keras.layers.Conv2D(num_tokens, kernel_size=1,
                                                 padding='same'))
        dec_layers.append(tf.keras.layers.Conv2D(channels, kernel_size=1,
                                                 padding='same'))

        self.encoder = tf.keras.Sequential(enc_layers)
        self.decoder = tf.keras.Sequential(dec_layers)

        if smooth_l1_loss:
            self.loss_fn = tf.keras.losses.Huber()
        else:
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.kl_div_loss_weight = kl_div_loss_weight
        self.normalization = normalization
        # _register_external_parameters omitted

    def norm(self, images):
        if not exists(self.normalization):
            return images
        means = tf.convert_to_tensor(self.normalization[0], dtype=images.dtype)
        stds = tf.convert_to_tensor(self.normalization[1], dtype=images.dtype)
        means = tf.reshape(means, [1, 1, 1, -1])
        stds = tf.reshape(stds, [1, 1, 1, -1])
        return (images - means) / stds

    @tf.function
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        indices = tf.argmax(logits, axis=1)
        return indices

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b = tf.shape(image_embeds)[0]
        n = tf.shape(image_embeds)[1]
        h = tf.cast(tf.sqrt(tf.cast(n, tf.float32)), tf.int32)
        image_embeds = rearrange(image_embeds, "b (h w) d -> b d h w", h=h, w=h)
        images = self.decoder(image_embeds)
        return images

    def call(self, img, return_loss=False, return_recons=False,
             return_logits=False, temp=None):
        image_size = self.image_size
        # Assume img shape [batch, height, width, channels]
        if tf.shape(img)[1] != image_size or tf.shape(img)[2] != image_size:
            raise ValueError(f"input must have the correct image size {image_size}")
        img = self.norm(img)
        logits = self.encoder(img)
        if return_logits:
            return logits
        temp = default(temp, self.temperature)
        soft_one_hot = tf_gumbel_softmax(logits, tau=temp,
                                         hard=self.straight_through, axis=1)
        sampled = einsum("b n h w, n d -> b d h w", soft_one_hot,
                         self.codebook.weights[0])
        out = self.decoder(sampled)
        if not return_loss:
            return out
        recon_loss = self.loss_fn(img, out)
        logits = rearrange(logits, "b n h w -> b (h w) n")
        log_qy = tf.nn.log_softmax(logits, axis=-1)
        log_uniform = tf.math.log(1.0 / tf.cast(self.num_tokens, logits.dtype))
        
        # KL-divergence calculation
        uniform_log_prob = tf.math.log(1.0 / tf.cast(self.num_tokens, logits.dtype))
        q = tf.exp(log_qy)
        kl_div = tf.reduce_mean(tf.reduce_sum(
            q * (log_qy - uniform_log_prob), axis=-1
        ))

        loss = recon_loss + self.kl_div_loss_weight * kl_div
        if not return_recons:
            return loss
        return loss, out

# Other utility classes
class OneHotEmbedding:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sequence):
        return tf.one_hot(sequence, depth=self.num_classes)

class aaDescriptors:
    def __init__(self, path):
        with open(path, mode="r") as infile:
            reader = csv.reader(infile)
            self.embeddings = {
                int(rows[0]): tf.convert_to_tensor(
                    np.array(rows[2:], dtype=np.float32)
                )
                for rows in reader if rows[0] != "Index"
            }

    def __call__(self, sequence):
        b = tf.shape(sequence)[0]
        l = tf.shape(sequence)[1]
        sequence_flat = tf.reshape(sequence, [-1])
        # Original pytorch implementation also does list comp
        embed_list = [self.embeddings[int(x.numpy())] for x in sequence_flat]
        emb = tf.stack(embed_list, axis=0)
        return tf.reshape(emb, [b, l, 66])

class CELLE(tf.keras.Model):
    def __init__(
        self,
        *,
        dim,
        vae,
        condition_vae=None,
        num_images=2,
        num_text_tokens=30,
        text_seq_len=1000,
        depth=16,
        heads=16,
        dim_head=64,
        reversible=False,
        attn_dropout=0.1,
        ff_dropout=0.1,
        attn_types=None,
        loss_cond_weight=1,
        loss_img_weight=7,
        stable=False,
        sandwich_norm=False,
        shift_tokens=True,
        rotary_emb=True,
        text_embedding="bert",
        fixed_embedding=True,
        shared_attn_ids=None,
        shared_ff_ids=None,
        share_input_output_emb=False,
        optimize_for_inference=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Ensure vae is an instance of DiscreteVAE or VQGanVAE
        self.text_embedding = text_embedding
        self.fixed_embedding = fixed_embedding
        self.num_text_tokens = num_text_tokens
        self.text_seq_len = text_seq_len
        self.num_images = num_images

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.image_size // (2 ** vae.num_layers)
        image_seq_len = image_fmap_size * image_fmap_size

        if not rotary_emb:
            self.text_pos_emb = keras.layers.Embedding(text_seq_len + 1, dim)
        else:
            self.text_pos_emb = lambda x: 0
        if not rotary_emb:
            self.image_pos_emb = AxialPositionalEmbedding(
                dim, axial_shape=(image_fmap_size, image_fmap_size)
            )
        else:
            self.image_pos_emb = lambda x: 0

        self.image_seq_len = image_seq_len
        self.num_image_tokens = num_image_tokens

        if exists(condition_vae):
            condition_size = condition_vae.image_size
            num_condition_tokens = condition_vae.num_tokens
            condition_fmap_size = condition_vae.image_size // (2 ** condition_vae.num_layers)
            condition_seq_len = condition_fmap_size * condition_fmap_size
            self.condition_emb = keras.layers.Embedding(num_condition_tokens, dim)
            if not rotary_emb:
                self.condition_pos_emb = AxialPositionalEmbedding(
                    dim, axial_shape=(condition_fmap_size, condition_fmap_size)
                )
            else:
                self.condition_pos_emb = lambda x: 0
        else:
            condition_fmap_size = 0
            condition_seq_len = 0
            num_condition_tokens = 0

        self.num_condition_tokens = num_condition_tokens
        self.condition_seq_len = condition_seq_len
        seq_len_total = text_seq_len + image_seq_len + condition_seq_len
        total_tokens = num_text_tokens + num_image_tokens + num_condition_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len_total

        self.vae = vae  # assume vae is already in evaluation mode if needed
        self.condition_vae = condition_vae

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len_total,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size + condition_fmap_size,
            num_images=num_images,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
            shared_attn_ids=shared_attn_ids,
            shared_ff_ids=shared_ff_ids,
            optimize_for_inference=optimize_for_inference,
        )
        self.stable = stable
        if stable:
            self.norm_by_max = DivideMax(dim=-1)
        self.to_logits = tf.keras.Sequential([
            keras.layers.LayerNormalization(axis=-1),
            keras.layers.Dense(total_tokens),
        ])
        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits.layers[1],
                                            0, num_text_tokens)
            self.image_emb = SharedEmbedding(self.to_logits.layers[1],
                                             num_text_tokens, total_tokens)
        else:
            if text_embedding in [None, "no_text"]:
                self.bos_token = num_text_tokens - 1
                self.text_emb = keras.layers.Embedding(num_text_tokens, dim)
            elif text_embedding == "unirep":
                self.bos_token = 24
                self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)
            elif text_embedding == "bert":
                self.bos_token = 2
                self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)
            elif text_embedding == "esm1b":
                self.bos_token = 0
                self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)
            elif text_embedding == "onehot":
                self.bos_token = 24
                self.text_emb = OneHotEmbedding(num_text_tokens)
            elif text_embedding == "aadescriptors":
                self.bos_token = 21
                self.text_emb = aaDescriptors(path="data/aaDescriptors.csv")
            self.image_emb = keras.layers.Embedding(num_image_tokens, dim)
        seq_range = tf.range(seq_len_total)
        logits_range = tf.range(total_tokens)
        seq_range = tf.reshape(seq_range, [1, -1, 1])
        logits_range = tf.reshape(logits_range, [1, 1, -1])
        if exists(condition_vae):
            logits_mask = tf.logical_or(
                tf.logical_and(seq_range >= text_seq_len,
                               logits_range < num_text_tokens),
                tf.logical_or(
                    tf.logical_and(
                        seq_range >= text_seq_len,
                        tf.logical_and(
                            seq_range < text_seq_len + condition_seq_len,
                            logits_range >= num_text_tokens + num_condition_tokens
                        )
                    ),
                    tf.logical_or(
                        tf.logical_and(
                            seq_range >= text_seq_len + condition_seq_len,
                            tf.logical_and(
                                logits_range >= num_text_tokens,
                                logits_range < num_text_tokens + num_condition_tokens
                            )
                        ),
                        tf.logical_and(
                            seq_range < text_seq_len,
                            logits_range >= num_text_tokens
                        )
                    )
                )
            )
        else:
            logits_mask = tf.logical_or(
                tf.logical_and(seq_range >= text_seq_len,
                               logits_range < num_text_tokens),
                tf.logical_and(seq_range < text_seq_len,
                               logits_range >= num_text_tokens)
            )
        self.logits_mask = logits_mask
        self.loss_img_weight = loss_img_weight
        self.loss_cond_weight = loss_cond_weight

    @tf.function
    @eval_decorator
    def generate_images(self, text, *, clip=None, filter_thres=0.5,
                        temperature=1.0, condition=None, img=None,
                        num_init_img_tokens=None, return_logits=False,
                        filter_method="top_k", progress=False,
                        cond_scale=1, use_cache=False):
        # Unpack variables.
        vae = self.vae
        cond_vae = self.condition_vae
        text_seq_len = self.text_seq_len
        image_seq_len = self.image_seq_len
        condition_seq_len = self.condition_seq_len
        num_condition_tokens = self.num_condition_tokens
        num_text_tokens = self.num_text_tokens

        # Set vae models to evaluation mode if needed.
        total_len = text_seq_len + image_seq_len + condition_seq_len
        text = text[:, :text_seq_len]
        out = text
        if exists(condition) and (not is_empty(condition)) and exists(cond_vae):
            cond_indices = cond_vae.get_codebook_indices(condition)
            cond_indices = cond_indices[:, :num_condition_tokens]
            out = tf.concat([out, cond_indices], axis=-1)
        if exists(img) and (not is_empty(img)) and exists(vae):
            img_indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, 
                                     int(0.4375 * image_seq_len))
            img_indices = img_indices[:, :num_img_tokens]
            out = tf.concat([out, img_indices], axis=-1)
        prev_cache = None
        cache = {} if use_cache else None
        full_logits = []
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len
            if is_image:
                is_not_condition = cur_len >= (text_seq_len + condition_seq_len)
            t = out[:, :text_seq_len]
            cond = out[:, text_seq_len : text_seq_len + condition_seq_len]
            im = out[:, text_seq_len + condition_seq_len :]
            if cond_scale != 1 and use_cache:
                prev_cache = cache.copy()
            logits = self(text=t, condition=cond, image=im, cache=cache)
            if cond_scale != 1:
                null_cond_logits = self(text=t, image=im,
                                        null_cond_prob=1.0, cache=prev_cache)
                logits = null_cond_logits + (logits - null_cond_logits) * cond_scale
            if use_cache:
                if not full_logits:
                    full_logits = logits
                else:
                    full_logits = tf.concat([full_logits, logits], axis=1)
            else:
                full_logits = logits
            logits = logits[:, -1, :]
            if filter_method == "top_k":
                filtered_logits = top_k(logits, thres=filter_thres)
            elif filter_method == "typical":
                filtered_logits = typical(logits, min_tokens_to_keep=20)
            sample = gumbel_sample(filtered_logits, temperature=temperature, axis=-1)
            if is_image:
                sample -= num_text_tokens
            if is_image and is_not_condition:
                sample -= num_condition_tokens
            sample = tf.expand_dims(sample, axis=1)
            out = tf.concat([out, sample], axis=-1)
        text_seq = out[:, :text_seq_len]
        condition_seq = out[:, text_seq_len : text_seq_len + condition_seq_len]
        img_seq = out[:, text_seq_len + condition_seq_len :]
        images = vae.decode(img_seq)
        if return_logits:
            return images, full_logits
        return images

    @tf.function
    def call(self, text, condition=None, image=None, return_loss=False,
             return_encoding=False, null_cond_prob=0, cache=None):
        if tf.shape(text)[-1] != self.text_seq_len:
            raise ValueError(
            f"the length {tf.shape(text)[-1].numpy()} of the text tokens you "
            f"passed in does not have the correct length ({self.text_seq_len})")
        batch = tf.shape(text)[0]
        total_seq_len = self.total_seq_len

        # (In TF you would implement null_cond_prob masking as needed.)
        if null_cond_prob > 0:
            null_mask = tf.random.uniform([batch]) < null_cond_prob
            null_mask = tf.expand_dims(tf.logical_not(null_mask), axis=1)
            text = text * tf.cast(null_mask, text.dtype)

        self.image = image
        self.condition = condition

        # Add <bos> token on the left.
        bos_tokens = tf.fill([batch, 1], self.bos_token)
        text = tf.concat([bos_tokens, text], axis=1)
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(tf.range(tf.shape(text)[1]))
        seq_len_current = tf.shape(tokens)[1]

        if exists(condition) and (not is_empty(condition)) and exists(self.condition_vae):
            is_raw_image = (tf.rank(condition) == 4)
            if is_raw_image:
                condition = self.condition_vae.get_codebook_indices(condition)
            condition_len = tf.shape(condition)[1]
            condition_emb = self.condition_emb(condition)
            condition_emb += self.condition_pos_emb(condition_emb)
            tokens = tf.concat([tokens, condition_emb], axis=1)
            seq_len_current += condition_len

        if exists(image) and (not is_empty(image)) and exists(self.vae):
            is_raw_image = (tf.rank(image) == 4)
            if is_raw_image:
                image_size = self.vae.image_size
                image = self.vae.get_codebook_indices(image)
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            image_len = tf.shape(image)[1]
            tokens = tf.concat([tokens, image_emb], axis=1)
            seq_len_current += image_len

        if tf.shape(tokens)[1] > total_seq_len:
            tokens = tokens[:, :-1]

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tf.stop_gradient(tokens) * (1 - alpha)

        if exists(cache) and (cache.get("offset") is not None):
            tokens = tokens[:, -1:]
        out = self.transformer(tokens, cache=cache)
        if self.stable:
            out = self.norm_by_max(out)
        if return_encoding:
            return out
        logits = self.to_logits(out)
        logits_mask = self.logits_mask[:, :seq_len_current]
        if exists(cache) and (cache.get("offset") is not None):
            logits_mask = logits_mask[:, -1:]
        max_neg_value = -tf.experimental.numpy.finfo(logits.dtype).max
        logits = tf.where(logits_mask, max_neg_value, logits)
        if exists(cache):
            cache["offset"] = cache.get("offset", 0) + tf.shape(logits)[1]
        if not return_loss:
            return logits
        labels = text[:, 1:]
        if exists(condition) and exists(self.condition_vae):
            offsetted_condition = condition + self.num_text_tokens
            labels = tf.concat([labels, offsetted_condition], axis=1)
        if not exists(image):
            raise ValueError("when training, image must be supplied")
        offsetted_image = image + self.num_text_tokens + self.num_condition_tokens
        labels = tf.concat([labels, offsetted_image], axis=1)
        logits_re = rearrange(logits, "b n c -> b c n")
        loss_text = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels[:, :self.text_seq_len],
                logits_re[:, :, :self.text_seq_len],
                from_logits=True
            )
        )
        loss_cond = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels[:, self.text_seq_len:self.text_seq_len+self.condition_seq_len],
                logits_re[:, :, self.text_seq_len:self.text_seq_len+self.condition_seq_len],
                from_logits=True
            )
        )
        loss_img = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels[:, self.text_seq_len+self.condition_seq_len:],
                logits_re[:, :, self.text_seq_len+self.condition_seq_len:],
                from_logits=True
            )
        )
        loss_dict = {
            "loss_text": loss_text,
            "loss_cond": loss_cond,
            "loss_img": loss_img,
        }
        loss = (loss_text + self.loss_cond_weight * loss_cond +
               self.loss_img_weight * loss_img) / (self.loss_img_weight +
               self.loss_cond_weight + 1)
        return loss, loss_dict, logits