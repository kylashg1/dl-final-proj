import tensorflow as tf

class EmbeddingToDensity(tf.keras.Model):
    def __init__(self):
        super(EmbeddingToDensity, self).__init__()
        self.dense_proj = tf.keras.layers.Dense(64*64, activation='relu')  # reduce sequence length
        self.reshape_layer = tf.keras.layers.Reshape((64, 64, 1))
        self.upsample = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),  # 64->128
            tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),  # 128->256
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')  # output map
        ])
    
    def call(self, x):
        x = tf.reduce_mean(x, axis=-1)  # (batch_size, seq_len) -> summarize embeddings
        x = self.dense_proj(x)           # (batch_size, 64*64)
        x = self.reshape_layer(x)        # (batch_size, 64, 64, 1)
        x = self.upsample(x)             # (batch_size, 256, 256, 1)
        return x
