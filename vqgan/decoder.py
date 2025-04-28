import tensorflow as tf

# Decided to implement as a function instead of a class because we're using
# the encoder as an intermediary component, not as a full model

# CNN Decoder for proteign image rexontr
# def decoder(latent_dim=128):
#     decoder =  tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(16, 16, latent_dim)),  # Depends on encoder compression
#         tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
#         tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
#         tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation=None)  # 1 output channel as logits
#     ], name="Decoder")

#     return decoder
def decoder(latent_dim=128):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, latent_dim)),
        tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),  # 16 -> 32
        tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),   # 32 -> 64
        tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),   # 64 -> 128
        tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),   # 128 -> 256
        tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation=None)  # final layer for 1 channel
    ], name="Decoder")
