import tensorflow as tf

class TransformerModel(tf.keras.Model):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4, dropout_rate=0.1):
        super(TransformerModel, self).__init__()

        # encoder layers
        self.encoder_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
            for _ in range(num_layers)
        ]

        # self attention layers
        self.self_attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            for _ in range(num_layers)
        ]

        # feed forward layers
        self.ffn_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='relu'),
                tf.keras.layers.Dense(embed_dim),
            ])
            for _ in range(num_layers)
        ]

        # dropout layers
        self.dropout_layers = [
            tf.keras.layers.Dropout(dropout_rate)
            for _ in range(num_layers)
        ]

        # Embedding to Density
        self.dense_proj = tf.keras.layers.Dense(64*64, activation='relu')  # reduce sequence length
        self.reshape_layer = tf.keras.layers.Reshape((64, 64, 1))
        self.upsample = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),  # 64->128
            tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),  # 128->256
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')  # output map
        ])

    def call(self, x, training=False):
        for norm, attn, ffn, drop in zip(self.encoder_layers, self.self_attention_layers, self.ffn_layers, self.dropout_layers):
            # normalization layer
            x_norm = norm(x)
            # self-attention layer
            attn_output = attn(x_norm, x_norm)
            x = x + drop(attn_output, training=training)

            # feed forward layer
            x_norm = norm(x)
            ffn_output = ffn(x_norm)
            x = x + drop(ffn_output, training=training)

        # Embedding to Density
        x = tf.reduce_mean(x, axis=-1)
        x = self.dense_proj(x)
        x = self.reshape_layer(x)
        x = self.upsample(x)

        return x
