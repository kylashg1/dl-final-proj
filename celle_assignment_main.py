# Import libraries
import dataloader as data
from vqgan import VQGAN, VectorQuantizer
import tensorflow as tf
from transformer import TransformerModel
from utilities import *
import matplotlib.pyplot as plt
import os
import pickle

# Setting paramaters for plots
os.makedirs('images', exist_ok=True)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def train_vqgan(train_dataset, target_dataset, input_shape):
    '''
    Creating + training the model
    '''
    vqgan = VQGAN(latent_dim=128, input_shape=input_shape)
    # Compiling the model with optimizer and loss
    vqgan.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(), # Reconstruction loss
    )
    history = vqgan.fit(train_dataset, target_dataset, epochs=10) # Training the model

    # Plot VQGAN training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('VQGAN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/vqgan_training_loss.png')
    plt.close()

    vqgan.save('vqgan_model') # Saving the trained model

def train_transformer(train_dataset, target_dataset):
    transformer_model = TransformerModel(embed_dim=256, num_heads=4, ff_dim=512, num_layers=2)
    # Compiling the model with optimizer and loss
    transformer_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.MeanSquaredError()
    )
    history = transformer_model.fit(train_dataset, target_dataset, epochs=3)

    # Plot Transformer training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Transformer Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/transformer_training_loss.png')
    plt.close()

    transformer_model.save('transformer_model') # Saving the trained model

def main(data_csv: str='data.csv', limit: int=0, crop_size: int=256):
    """
    Parameters
    data_csv path to data.csv
    limit TOTAL number of images to download from OpenCell
    """

    # Preprocessing data ----------------------------

    # # downloads images from OpenCell and put them into a csv and processed data folders
    if not os.path.exists(data_csv):
        processed_data = data.Preprocess()
        processed_data.download_cell_images(limit=limit) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, 0 means download ALL images
        processed_data.populate_image_inputs("unprocessed_data", "processed_data") # splits .tiff images into .png images
        processed_data.populate_csv(data_csv) # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv

    # creates the vectorized dataset to pass into VQGAN
    dataset = data.OpenCellLoaderTF(data_csv, crop_size).get_dataset()

    if not os.path.exists('tensors.pkl'):
        # Get data in four separate tensors
        nucleus_list = []
        target_list = []
        threshold_list = []
        sequence_list = []

        # Adding batches to respective lists
        for batch in dataset:
            nucleus_list.append(batch["nucleus"])
            target_list.append(batch["target"])
            threshold_list.append(batch["threshold"])
            sequence_list.append(batch["sequence"])

        # Making lists into tensors
        nucleus_tensor = tf.stack(nucleus_list)
        target_tensor = tf.stack(target_list) # Probably not useful
        threshold_tensor = tf.stack(threshold_list)
        sequence_tensor = tf.stack(sequence_list)

        # Pickle tensors
        with open('tensors.pkl', 'wb') as file:
            pickle.dump((nucleus_tensor, target_tensor, threshold_tensor,
            sequence_tensor), file)

    with open('tensors.pkl', 'rb') as file:
        nucleus_tensor, target_tensor, threshold_tensor, sequence_tensor = \
        pickle.load(file)

    # VQGAN ----------------------------
    
    # Training vqgan
    if not os.path.exists('vqgan_model'):
        combined_tensor = tf.concat([nucleus_tensor, threshold_tensor], axis=0)
        train_vqgan(combined_tensor, combined_tensor, input_shape=crop_size)
    vqgan = tf.keras.models.load_model('vqgan_model', custom_objects={'VectorQuantizer': VectorQuantizer})

    # Feeding in nucleus and threshold images into vqgan
    nucleus_vqgan_output = []
    threshold_vqgan_output = []
    batch_size = 69
    for i in range(0, nucleus_tensor.shape[0], batch_size):
        nucleus_vqgan_output.append(vqgan(nucleus_tensor[i:i+batch_size]))
        threshold_vqgan_output.append(vqgan(threshold_tensor[i:i+batch_size]))
    nucleus_data = tf.concat(nucleus_vqgan_output, axis=0)
    threshold_data = tf.concat(threshold_vqgan_output, axis=0)
    print(f"Nucleus data shape: {nucleus_data.shape}")
    print(f"Threshold data shape: {threshold_data.shape}")

    # Formatting all the data before concatenation
    sequence = tf.squeeze(tf.cast(sequence_tensor, tf.float32))
    nucleus_data = tf.squeeze(tf.cast(nucleus_data, tf.float32))
    threshold_data = tf.squeeze(tf.cast(threshold_data, tf.float32))

    # Concatenating embeddings
    concat_input = tf.concat([sequence, nucleus_data, threshold_data], axis=1)
    print(f"Concatentation shape: {concat_input.shape}")

    # Transformer ----------------------------

    # Transformer + Embedding to Density Model
    if not os.path.exists('transformer_model'):
        train_transformer(concat_input, target_tensor)
    transformer_model = tf.keras.models.load_model('transformer_model')

    density_map_list = []
    batch_size = 437
    for i in range(0, concat_input.shape[0], batch_size):
        density_map_list.append(transformer_model(concat_input[i:i+batch_size]))
    density_maps = tf.concat(density_map_list, axis=0)
    print(f"Density map shape: {density_maps.shape}")

    # Computing EMD values and plotting
    emd_scores = compute_emd(target_tensor, density_maps).numpy()
    save_emd(emd_scores)

    # Saving images
    for i in range(density_maps.shape[0]):
        t_img, n_img, tar_img, d_map = [
            tf.squeeze(x[i]) for x in [
                threshold_tensor, nucleus_tensor, target_tensor, density_maps
            ]
        ]
        os.makedirs(f'images/{i}', exist_ok=True)
        overlay_density_on_image(image=t_img, density_map=d_map, fname=f'{i}/threshold_density', alpha=0.5)
        save_img(t_img, f'{i}/threshold')
        overlay_density_on_image(image=n_img, density_map=d_map, fname=f'{i}/nucleus_density', alpha=0.5)
        save_img(n_img, f'{i}/nucleus')
        overlay_density_on_image(image=tar_img, density_map=d_map, fname=f'{i}/target_density', alpha=0.5)
        save_img(tar_img, f'{i}/target')
        save_density_map(d_map, f'{i}/density_map')

if __name__ ==  "__main__":
    main()
