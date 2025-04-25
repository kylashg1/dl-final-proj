import tensorflow as tf
import dataloader as data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import os


def main():
    dataloader = data.dataloader()
    dataloader.download_cell_imgages()
    dataloader.populate_inputs("unprocessed_data", "processed_data")
    dataloader.populate_csv("data.csv")
    
    
if __name__ ==  "__main__":
    main()


