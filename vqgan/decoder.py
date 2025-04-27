import tensorflow as tf

def build_decoder(latent_dim=128):
    """
    Simple CNN decoder for grayscale protein images.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, latent_dim)),  # Depends on encoder downsampling
        tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation=None)  # 1 output channel, logits
    ], name="Decoder")