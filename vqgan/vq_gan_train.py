import tensorflow as tf
from vqgan import VQGAN
from encoder import econder
from decoder import decoder
from vector_quantizer import VectorQuantizer


def main():
    # Build models
    encoder = encoder(latent_dim=128)
    decoder = encoder(latent_dim=128)
    quantizer = VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=0.25)

    # Initialize VQGAN model
    model = VQGAN(encoder=encoder, decoder=decoder, quantizer=quantizer)

    # Compile model with optimizer, loss, and metrics 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
    )

    # Loading datasets
    train_dataset = None
    val_dataset = None

    # Train
    model.fit(train_dataset, epochs=100, validation_data=val_dataset)

    # Test
    # test_loss, test_accuracy = model.evaluate()

if __name__ == "__main__":
    main()