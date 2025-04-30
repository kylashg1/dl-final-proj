# Import everything needed
import dataloader as data
from vqgan import VQGAN, VectorQuantizer
import tensorflow as tf
from transformer import TransformerModel
from utilities import *
import matplotlib.pyplot as plt
import os

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Creating + training the model - training data should be in (batch_size, 64, 64, 1) normalized to [-1, 1]
def train_vqgan(train_dataset, target_dataset):
    vqgan = VQGAN(latent_dim=128)
    # Compiling the model with optimizer and loss
    vqgan.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(), # Reconstruction loss
    )
    history = vqgan.fit(train_dataset, target_dataset, epochs=10, verbose=2) # Training the model

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


# # Example Usage:
# prob_map = np.random.rand(256, 256)  # Example probability heat map
# ground_truth = np.random.rand(256, 256)  # Example ground truth image

# emd_result = compute_emd(prob_map, ground_truth)
# print(f"Earth Mover's Distance: {emd_result}")

def main():
    # --Example usage--
    # Preprocessing data
    # Downloads cell images from OpenCell dataset, comment out if already downloaded

    # Model creation

    # Metrics 





    # Preprocess

    # # downloads images from OpenCell and put them into a csv and processed data folders
    # processed_data = data.preprocess()
    # limit = 100
    # # processed_data.download_cell_imgages(limit=limit) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, None means download ALL images
    # processed_data.populate_inputs("unprocessed_data", "processed_data") # splits .tiff images into .png images
    # processed_data.populate_csv("data.csv") # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv

    # creates the vectorized dataset to pass into VQGAN
    dataset = data.OpenCellLoaderTF("data.csv", crop_size=256).get_dataset()

    # Get data in four separate tensors
    nucleus_list = []
    target_list = []
    threshold_list = []
    sequence_list = []

    for batch in dataset:
        nucleus_list.append(batch["nucleus"])
        target_list.append(batch["target"])
        threshold_list.append(batch["threshold"])
        sequence_list.append(batch["sequence"])

    nucleus_tensor = tf.stack(nucleus_list)
    target_tensor = tf.stack(target_list) # Probably not useful
    threshold_tensor = tf.stack(threshold_list)
    sequence_tensor = tf.stack(sequence_list)

    # VQGAN
    
    # Traning vqgan
    combined_list = nucleus_list + threshold_list
    combined_tensor = tf.stack(combined_list)
    train_vqgan(combined_tensor, combined_tensor)
    vqgan = tf.keras.models.load_model('vqgan_model', custom_objects={'VectorQuantizer': VectorQuantizer})


    nucleus_data = vqgan(nucleus_tensor)
    print(f"nucleus data output {nucleus_data}")
    print(f"Nucleus data shape: {nucleus_data.shape}")  

    # For the threshold image
    threshold_data = vqgan(threshold_tensor)
    print(f"threshold data output {threshold_data}")
    print(f"threshold data shape: {threshold_data.shape}")  


    # Formatting all the data before concatenation
    sequence = tf.cast(sequence_tensor, tf.float32)
    sequence = tf.squeeze(sequence, axis=1)

    nucleus_data = tf.cast(nucleus_data, tf.float32)
    nucleus_data = tf.squeeze(nucleus_data, axis=3)

    threshold_data = tf.cast(threshold_data, tf.float32)
    threshold_data = tf.squeeze(threshold_data, axis=3)



    # Concatenate
    concat_input = tf.concat([sequence, nucleus_data, threshold_data], axis=1)
    print(f"Concatentation shape: {concat_input.shape}")



    # Transformer Model
    transformer_model = TransformerModel(embed_dim=256, num_heads=8, ff_dim=512, num_layers=4)
    transformer_output = transformer_model(concat_input)  # (3, 1513, 256)
    print(f"Transoformer output shape: {transformer_output.shape}")  # Should be (3, 1513, 256)


    # Density Map
    density_model = EmbeddingToDensity()
    density_maps = density_model(transformer_output)
    print(f"Density map shape: {density_maps.shape}")  # (3, 256, 256, 1)

    # overlay_density_on_image(nucleus_image, density_map, alpha=0.5)

    # emd_value = compute_emd(density_map, threshold_image)
    # print(f"emd value: {emd_value}")


    emd_scores = compute_emd(nucleus_list, density_maps).numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(emd_scores, marker='o')
    plt.title('Earth Mover\'s Distance (Batch)')
    plt.xlabel('Sample Index')
    plt.ylabel('EMD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('emd.png')
    plt.close()
    
    if not os.path.exists('images'):
        os.makedirs('images')

    for i, imgs in enumerate(zip(threshold_list, nucleus_list, target_list, density_maps)):
        t_img, n_img, tar_img, d_map = imgs
        overlay_density_on_image(image=t_img, density_map=d_map, fname=f'threshold_density{i}', alpha=0.5)
        save_img(t_img, f'threshhold{i}')
        overlay_density_on_image(image=n_img, density_map=d_map, fname=f'nucleus_density{i}', alpha=0.5)
        save_img(n_img, f'nucleus{i}')
        overlay_density_on_image(image=tar_img, density_map=d_map, fname=f'target_density{i}', alpha=0.5)
        save_img(tar_img, f'target{i}')

if __name__ ==  "__main__":
    main()