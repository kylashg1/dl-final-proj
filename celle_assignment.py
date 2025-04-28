# Import everything needed
import dataloader as data
from vqgan import VQGAN, VectorQuantizer
import tensorflow as tf
from transformer_new_new_new import TransformerModel
from embed_to_density import EmbeddingToDensity
from heatmap import overlay_density_on_image

# Creating + training the model - training data should be in (batch_size, 64, 64, 1) normalized to [-1, 1]
def train_vqgan(train_dataset, target_dataset):
    vqgan = VQGAN(latent_dim=128)
    # Compiling the model with optimizer and loss
    vqgan.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(), # Reconstruction loss
    )
    vqgan.fit(train_dataset, target_dataset, epochs=1) # Training the model
    vqgan.save('vqgan_model') # Saving the trained model

def main():
    # Preprocess

    # downloads images from OpenCell and put them into a csv and processed data folders
    # processed_data = data.dataloader()
    # limit = 100
    # # processed_data.download_cell_imgages(limit=limit) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, None means download ALL images
    # processed_data.populate_inputs("unprocessed_data", "processed_data") # splits .tiff images into .png images
    # processed_data.populate_csv("data.csv") # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv

    # creates the vectorized dataset to pass into VQGAN
    dataset_tensor = data.OpenCellLoaderTF("data.csv", crop_size=256)
    # dataset_tensor = "data.csv"
    dataset_tensor = dataset_tensor.get_dataset()
    dataset_tensor = dataset_tensor.batch(5)

    input = None
    threshold = None
    sequence = None
    #syntax for accessing each vectorized dataset - need to rework
    for batch in dataset_tensor:
        input = batch['nucleus']  # (batch_size, 256, 256, 1) # VQGAN
        target = batch['target']    # (batch_size, 256, 256, 1) # this one is NOT passed into VQGAN
        threshold = batch['threshold']  # (batch_size, 256, 256, 1) #VQGAN
        sequence = batch['sequence']  # (batch_size, 1, 1001, 256) # already tokenized, just needs to be concatted

    # checking shape of each tensor
    # print(nucleus.shape)  # (batch_size, 256, 256, 1)
    print(target.shape)  # (batch_size, 256, 256, 1)
    print(threshold.shape)  # (batch_size, 256, 256, 1)
    print(sequence.shape)  # (batch_size, 1, 1001, 256)

    # VQGAN
    # Loading in trained vqgan

    train_vqgan(input, input)
    vqgan = tf.keras.models.load_model('vqgan_model', custom_objects={'VectorQuantizer': VectorQuantizer})

    # For the nucleus
    # nucleus_vqgan = VQGAN() # VQGAN(encoder(), decoder(), VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=.25))

    nucleus_data = vqgan(input)
    print(f"output {nucleus_data}")

    print(nucleus_data.shape)  

    # For the threshold image
    # threshold_vqgan = VQGAN() # VQGAN(encoder(), decoder(), VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=.25))
    threshold_data = vqgan(threshold)
    print(f"output {threshold_data}")


    print(threshold_data.shape)  

    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.squeeze(sequence, axis=1)

    nucleus_data = tf.cast(nucleus_data, tf.float32)
    nucleus_data = tf.squeeze(nucleus_data, axis=3)

    threshold_data = tf.cast(threshold_data, tf.float32)
    threshold_data = tf.squeeze(threshold_data, axis=3)
    # Concatenate
    concat_input = tf.concat([sequence, nucleus_data, threshold_data], axis=1)

    print(concat_input.shape)
#     # Feed into transformer

# # # Embedding/Concatenating everything together: concat order -> AA + nucleus + threshold
#     num_layers = 4
#     d_model = 128
#     num_heads = 8
#     dff = 512
# #     vocab_size = 10000  # You can adjust this depending on your data
# #     max_len = 1000  # Adjust this based on the sequence length you're using
# #     num_classes = 10  # For classification, adjust accordingly

#     transformer = Transformer(
#         num_layers,
#         d_model,
#         num_heads,
#         dff,
#         input_vocab_size,
#         target_vocab_size
#     )

#     inputs = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=input_vocab_size)
#     targets = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=target_vocab_size)

#     look_ahead_mask = None
#     padding_mask = None

#     output = transformer((inputs, targets), training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

#     # Print the output shape
#     print(output.shape)  # (batch_size, num_classes) or whatever your output shape is
    transformer_model = TransformerModel(embed_dim=256, num_heads=8, ff_dim=512, num_layers=4)
    # Create model

    # Example input


    # Forward pass
    output = transformer_model(concat_input)

    print(output.shape)  # Should be (3, 1513, 256)

    transformer_output = transformer_model(concat_input)  # (3, 1513, 256)

    # Generate density map
    density_model = EmbeddingToDensity()
    density_maps = density_model(transformer_output)

    print(density_maps.shape)  # (3, 256, 256, 1)

    nucleus_image = input[0]       # (256, 256, 1)
    density_map = density_maps[0]  # (256, 256, 1)

    overlay_density_on_image(nucleus_image, density_map, alpha=0.5)









# Transformer




if __name__ ==  "__main__":
    main()
