# Import everything needed
import dataloader as data
from vqgan.vqgan import VQGAN
from vqgan.encoder import encoder
from vqgan.decoder import decoder
from vqgan.vector_quantizer import VectorQuantizer
import tensorflow as tf

def main():
    # Preprocess

    # downloads images from OpenCell and put them into a csv and processed data folders
    processed_data = data.dataloader()
    # processed_data.download_cell_imgages(3) # download function for OpenCell AWS server - only argument is how many TOTAL images you want to download, None means download ALL images
    # processed_data.populate_csv("data.csv") # puts processed .png images into the csv and searched up the corresponding amino acid sequence and also puts into data.csv


    # creates the vectorized dataset to pass into VQGAN
    dataset_tensor = data.OpenCellLoaderTF("data.csv", crop_size=64)
    dataset_tensor = dataset_tensor.get_dataset()
    dataset_tensor = dataset_tensor.batch(3)


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
    # For the nucleus
    nucleus_vqgan = VQGAN(encoder(), decoder(), VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=.25))
    nucleus_data = nucleus_vqgan(input)
    print(f"output {nucleus_data}")

    # For the threshold image
    threshold_vqgan = VQGAN(encoder(), decoder(), VectorQuantizer(codebook_size=512, code_dim=128, commitment_cost=.25))
    threshold_data = threshold_vqgan(threshold)
    print(f"output {threshold_data}")

    # Concatenate
    tf.concat([sequence, nucleus_data, threshold_data])

    # Feed into transformer












# Embedding/Concatenating everything together: concat order -> AA + nucleus + threshold








# Transformer




if __name__ ==  "__main__":
    main()
