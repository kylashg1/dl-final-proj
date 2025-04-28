# Import everything needed
import dataloader as data
from vqgan import VQGAN, VectorQuantizer
import tensorflow as tf
from transformer_new_new_new import TransformerModel
from embed_to_density import EmbeddingToDensity
from heatmap import overlay_density_on_image
import matplotlib.pyplot as plt
import os

# Creating + training the model - training data should be in (batch_size, 64, 64, 1) normalized to [-1, 1]
def train_vqgan(train_dataset, target_dataset):
    vqgan = VQGAN(latent_dim=128)
    # Compiling the model with optimizer and loss
    vqgan.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(), # Reconstruction loss
    )
    history = vqgan.fit(train_dataset, target_dataset, epochs=100, verbose=2) # Training the model

    # Plot VQGAN training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('VQGAN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('vqgan_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    vqgan.save('vqgan_model') # Saving the trained model

def main():
    # Preprocess
    if not os.path.exists('data.csv':)
        # downloads images from OpenCell and put them into a csv and processed data folders
        processed_data = data.dataloader()
        limit = 100
        processed_data.download_cell_images(limit=limit) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, None means download ALL images
        processed_data.populate_inputs("unprocessed_data", "processed_data") # splits .tiff images into .png images
        processed_data.populate_csv("data.csv") # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv

    # creates the vectorized dataset to pass into VQGAN
    dataset = data.OpenCellLoaderTF("data.csv", crop_size=256).get_dataset()

    # Map nucleus and threshold images into two-channel images
    def vqgan_two_channel_inputs(data):
        return (tf.concat([data['nucleus'], data['threshold']], axis=-1)) * 2

    vqgan_dataset = dataset.map(vqgan_two_channel_inputs).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    # VQGAN
    # Loading in trained vqgan
    train_vqgan(vqgan_dataset, vqgan_dataset)
    vqgan = tf.keras.models.load_model('vqgan_model', custom_objects={'VectorQuantizer': VectorQuantizer})

    nucleus_threshold_data = vqgan(vqgan_dataset)
    print(f"VQGAN output: {nucleus_threshold_data.shape}")

    nucleus_data, threshold_data = tf.split(combined_output, num_or_size_splits=2, axis=-1)
    print(f"Nucleus data: {nucleus_data.shape}")
    print(f"Threshold data: {threshold_data.shape}")

    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.squeeze(sequence, axis=1)

    nucleus_data = tf.cast(nucleus_data, tf.float32)
    nucleus_data = tf.squeeze(nucleus_data, axis=3)

    threshold_data = tf.cast(threshold_data, tf.float32)
    threshold_data = tf.squeeze(threshold_data, axis=3)
    # Concatenate
    concat_input = tf.concat([sequence, nucleus_data, threshold_data], axis=1)

    print(concat_input.shape)
    # Forward pass
    output = transformer_model(concat_input)

    print(output.shape)  # Should be (3, 1513, 256)

    transformer_output = transformer_model(concat_input)  # (3, 1513, 256)

    # Generate density map
    density_model = EmbeddingToDensity()
    density_maps = density_model(transformer_output)

    print(density_maps.shape)  # (3, 256, 256, 1)

    nucleus_image = nucleus_data[0]# (256, 256, 1)
    density_map = density_maps[0]  # (256, 256, 1)

    overlay_density_on_image(nucleus_image, density_map, alpha=0.5)

if __name__ ==  "__main__":
    main()
