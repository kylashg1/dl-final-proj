import tensorflow as tf

class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, codebook_size, code_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size # number of discrete vectors
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost # determines how much econder sticks vs. moving around
        self.codebook = self.add_weight(shape=(codebook_size, code_dim), initializer='uniform', trainable=True, name='codebook')

    def call(self, inputs, training=False):
        # Flatten input to become (B, H, W, 128) --> (BHW, 128)
        flat_inputs = tf.reshape(inputs, [-1, self.code_dim])

        # Compute distances to codebook vectors
        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat_inputs, self.codebook, transpose_b=True)
            + tf.reduce_sum(self.codebook**2, axis=1)
        )

        # For each vector, finding the closet codebook vector
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.codebook_size)
        quantized = tf.matmul(encodings, self.codebook)

        # Reshape back to input shape
        quantized = tf.reshape(quantized, tf.shape(inputs))

        # Compute commitment and codebook loss
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2) # Loss of moving the embeddings closer to the inputs
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2) # Loss of moving the encoder output closer to its nearest embedding.
        total_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized, total_loss

class VQGAN(tf.keras.Model):
    def __init__(self, latent_dim=128, input_shape=256, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            # Assuming 2-channel image: nucleus and thresholded grayscale inputs
            tf.keras.layers.Input(shape=(input_shape, input_shape, 2)),
            tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(latent_dim, kernel_size=4, strides=1, padding='same')
        ], name="Encoder")
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16, 16, latent_dim)),  # Depends on encoder compression
            tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2, 3, strides=1, padding='same', activation=None) # 1 output channel as logits
        ], name="Decoder")
        self.vquantizer = VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=0.25)

    def call(self, input, training=True):
        # Encoding the images
        encoder_output = self.encoder(input, training=training)

        # Getting vector quantizer output (codebook) and loss
        vquantizer_output, codebook_loss = self.vquantizer(encoder_output, training=training)

        # Recontructing the images via decoder
        decoder_output = self.decoder(vquantizer_output, training=training)

        # Added loss
        self.add_loss(codebook_loss)

        # Returning reconstructed image
        return decoder_output
