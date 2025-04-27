import tensorflow as tf

# Creating class for VQGan model


class VQGAN(tf.keras.Model):
    def __init__(self, encoder, decoder, vquantizer, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.vquantizer = vquantizer

    def call(self, inputs, training=False):
        # Encoding the images
        encoder_output = self.encoder(inputs, training=training)

        # Getting vector quantizer output (codebook) and loss
        vquantizer_output, codebook_loss = self.vquantizer(encoder_output, training=training)

        # Recontructing the images via decoder
        decoder_output = self.decoder(vquantizer_output, training=training)

        # Added loss
        self.add_loss(codebook_loss)

        # Returning reconstructed image
        return decoder_output