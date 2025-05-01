# Import libraries
import dataloader as data
from vqgan import VQGAN, VectorQuantizer
import tensorflow as tf
from transformer import TransformerModel
from utilities import *
import matplotlib.pyplot as plt
import os

# Setting paramaters for plots
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def train_vqgan(train_dataset, target_dataset, input_shape):
    '''
    Creating + training the model - training data should be in (batch_size, 64, 64, 1) normalized to [-1, 1]
    '''
    vqgan = VQGAN(latent_dim=128, input_shape=input_shape)
    # Compiling the model with optimizer and loss
    vqgan.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
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
    plt.savefig('images/vqgan_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    vqgan.save('vqgan_model') # Saving the trained model

def tensor_to_image(tensor):
    '''
    Converts tensor to iamge
    '''
    tensor = tf.convert_to_tensor(tensor)

    # Normalize if needed (optional, depending on how your tensor looks)
    if tf.reduce_max(tensor) > 1.0:
        tensor = tensor / tf.reduce_max(tensor)

    # If grayscale, expand dims to 3 channels
    if len(tensor.shape) == 2:
        tensor = tf.expand_dims(tensor, axis=-1)

    tensor = tf.clip_by_value(tensor, 0.0, 1.0) # make sure values are between 0-1
    return tensor

def compute_emd(nucleus_data_list, density_maps):
    '''
    Computes Earth Mover's Distance - a new type of metric
    '''
    # Stack the list into a proper tensor
    nucleus_data = tf.stack(nucleus_data_list, axis=0)  # Shape: [batch, 256, 256]

    nucleus_flat = tf.reshape(nucleus_data, [tf.shape(nucleus_data)[0], -1])
    density_flat = tf.reshape(density_maps, [tf.shape(density_maps)[0], -1])

    nucleus_flat /= tf.reduce_sum(nucleus_flat, axis=1, keepdims=True) + 1e-8
    density_flat /= tf.reduce_sum(density_flat, axis=1, keepdims=True) + 1e-8

    nucleus_cdf = tf.cumsum(nucleus_flat, axis=1)
    density_cdf = tf.cumsum(density_flat, axis=1)

    emd_per_sample = tf.reduce_mean(tf.abs(nucleus_cdf - density_cdf), axis=1)

    return emd_per_sample

def save_img(img, fname):
    '''
    Saves images to images/ folder
    '''
    image = tensor_to_image(img)
    plt.imshow(tf.squeeze(image), cmap='gray') # squeeze removes extra channel if needed
    plt.axis('off')
    plt.savefig(f'images/{fname}.png')

def main(data_csv: str='data.csv', limit: int=100, crop_size: int=256):
    """
    Parameters
    data_csv path to data.csv
    limit TOTAL number of images to download from OpenCell
    """

    # Preprocessing data ----------------------------

    # # downloads images from OpenCell and put them into a csv and processed data folders
    if not os.path.exists(data_csv) and not os.path.exists("processed_data") and not os.path.exists("unprocessed_data"):
        processed_data = data.dataloader()
        processed_data.download_cell_images(limit=limit) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, None means download ALL images
        processed_data.populate_inputs("unprocessed_data", "processed_data") # splits .tiff images into .png images
        processed_data.populate_csv(data_csv) # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv

    # creates the vectorized dataset to pass into VQGAN
    dataset = data.OpenCellLoaderTF(data_csv, crop_size).get_dataset()

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

    # VQGAN ----------------------------
    
    # Traning vqgan
    if not os.path.exists('vqgan_model'):
        combined_list = nucleus_list + threshold_list
        combined_tensor = tf.stack(combined_list)
        train_vqgan(combined_tensor, combined_tensor, input_shape=crop_size)
    vqgan = tf.keras.models.load_model('vqgan_model', custom_objects={'VectorQuantizer': VectorQuantizer})


    # Feeding in nucleus images into vqgan
    nucleus_data = vqgan(nucleus_tensor)
    print(f"nucleus data output {nucleus_data}")
    print(f"Nucleus data shape: {nucleus_data.shape}")  

    ## Feeding in threshold images into vqgan
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



    # Concatenating embeddings
    concat_input = tf.concat([sequence, nucleus_data, threshold_data], axis=1)
    print(f"Concatentation shape: {concat_input.shape}")

    # Transformer ----------------------------

    # Transformer Model
    transformer_model = TransformerModel(embed_dim=256, num_heads=8, ff_dim=512, num_layers=4)
    transformer_output = transformer_model(concat_input)  # (3, 1513, 256)
    print(f"Transoformer output shape: {transformer_output.shape}")  # Should be (3, 1513, 256)


    # Density Map
    density_model = EmbeddingToDensity()
    density_maps = density_model(transformer_output)
    print(f"Density map shape: {density_maps.shape}")  # (3, 256, 256, 1)


    # Computing EMD values and plotting rsule
    emd_scores = compute_emd(nucleus_list, density_maps).numpy()
    plt.figure(figsize=(6, 6))
    plt.plot(emd_scores, marker='o')
    plt.title('Earth Mover\'s Distance (Batch)')
    plt.xlabel('Sample Index')
    plt.ylabel('EMD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/emd.png')
    plt.close()


    # Saving images
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
