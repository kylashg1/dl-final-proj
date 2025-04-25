import tensorflow as tf
import dataloader as data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import numpy as np
import os


def main():
    # dataloader = data.dataloader()
    # dataloader.download_cell_imgages()
    # dataloader.populate_inputs("unprocessed_data", "processed_data")
    # dataloader.populate_csv("data.csv")
    
    processed_dataset = data.OpenCellLoaderTF("data.csv")
    dataset = processed_dataset.get_dataset()
    dataset = dataset.batch(3)
    for batch in dataset:
        nucleus = batch['nucleus']  # Shape: (batch_size, 256, 256, 1)
        target = batch['target']    # Shape: (batch_size, 256, 256, 1)
        threshold = batch['threshold']  # Shape: (batch_size, 256, 256, 1)
        sequence = batch['sequence']  # Shape: (batch_size, 1, 1001, 256)
        
        # Now you can use the batch data for training or evaluation
        print(nucleus.shape)  # Should print (batch_size, 256, 256, 1)
        print(target.shape)  # Should print (batch_size, 256, 256, 1)
        print(threshold.shape)  # Should print (batch_size, 256, 256, 1)
        print(sequence.shape)  # Should print (batch_size, 1, 1001, 256)

    
if __name__ ==  "__main__":
    main()


