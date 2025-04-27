# Import everything needed
import dataloader as data


def main():
    # Preprocess

    # downloads images from OpenCell and 
    processed_data = data.dataloader()
    # processed_data.download_cell_imgages(3)
    # processed_data.populate_csv("data.csv")

    # creates the vectorized dataset to pass into VQGAN
    dataset_tensor = data.OpenCellLoaderTF("data.csv", crop_size=64)
    dataset_tensor = dataset_tensor.get_dataset()
    dataset_tensor = dataset_tensor.batch(3)
    for batch in dataset_tensor:
        nucleus = batch['nucleus']  # (batch_size, 256, 256, 1)
        target = batch['target']    # (batch_size, 256, 256, 1)
        threshold = batch['threshold']  # (batch_size, 256, 256, 1)
        sequence = batch['sequence']  # (batch_size, 1, 1001, 256)
        
        # checking shape of each tensor
        print(nucleus.shape)  # (batch_size, 256, 256, 1)
        print(target.shape)  # (batch_size, 256, 256, 1)
        print(threshold.shape)  # (batch_size, 256, 256, 1)
        print(sequence.shape)  # (batch_size, 1, 1001, 256)


    # VQGAN


    # Embedding/Concatenating everything together


    # Transformer


if __name__ ==  "__main__":
    main()