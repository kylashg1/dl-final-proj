from argparse import ArgumentParser
import tensorflow as tf
import dataloader as data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import numpy as np
import os


def parse_args():
    """
    Command-line argument parsing
    """
    parser = ArgumentParser(description='“Transforming” Protein sequences into images humans can understand!')
    parser.add_argument('--data', required=True, type=str, help='File path to data csv')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs used in training')
    return parser.parse_args()

def main(args):
    print(f'Data CSV file path: {args.data}')
    print(f'Epochs: {args.epochs}')
    print(f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}')

    # dataset = processed_dataset.get_dataset()
    # dataset = dataset.batch(3)
    # for batch in dataset:
    #     nucleus = batch['nucleus']  # Shape: (batch_size, 256, 256, 1)
    #     target = batch['target']    # Shape: (batch_size, 256, 256, 1)
    #     threshold = batch['threshold']  # Shape: (batch_size, 256, 256, 1)
    #     sequence = batch['sequence']  # Shape: (batch_size, 1, 1001, 256)
        
    #     # Now you can use the batch data for training or evaluation
    #     print(nucleus.shape)  # Should print (batch_size, 256, 256, 1)
    #     print(target.shape)  # Should print (batch_size, 256, 256, 1)
    #     print(threshold.shape)  # Should print (batch_size, 256, 256, 1)
    #     print(sequence.shape)  # Should print (batch_size, 1, 1001, 256)

    
if __name__ ==  "__main__":
    main(parse_args())
