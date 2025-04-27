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