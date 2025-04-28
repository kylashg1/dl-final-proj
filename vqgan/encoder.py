import tensorflow as tf

# Decided to implement as a function instead of a class because we're using
# the encoder as an intermediary component, not as a full model

# CNN encoder for proteign images
def encoder(latent_dim=128):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 1)),  # Assuming 256x256 thresholded input and is grayscale
        tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(latent_dim, kernel_size=4, strides=1, padding='same') 
    ], name="Encoder")
    
    return encoder
