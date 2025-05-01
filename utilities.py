import re
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import tensorflow as tf
import matplotlib.pyplot as plt


# path to folder that holds .tif images from OpenCell
TARGET_DIR = Path("../dl-final-proj/unprocessed_data")

# OpenCell dataset website
BUCKET = "czb-opencell"
PREFIX = "microscopy/raw/"

def download_from_OpenCell(limit=0):
    """
    dowloads cell images from OpenCell dataset
    limit: the max number of images this script download, for local downloading/testing purposes; a limit of 0 means download everything!!!
    """
    # can limit the amount of images downlaoded from OpenCell dataset for local testing purposes
    print(f"connecting to S3 bucket '{BUCKET}' to download 1 image per unique ENSG_id (up to limit:{limit})...")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    ensg_seen = set()
    count = 0
    ensg_regex = re.compile(r'(ENSG\d{11})')

    if not limit:  # catches limit=None or limit=0
        limit = float('inf')

    for pg in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in pg.get("Contents", []):
            key = obj["Key"]

            if not key.endswith("_proj.tif"):
                continue

            # attempts to extrac ENSG_id from filenames
            match = ensg_regex.search(key)
            if not match:
                continue

            ensg_id = match.group(1)
            if ensg_id in ensg_seen:
                continue

            # new ENSG_id -> download file
            filename = key.split("/")[-1]
            target_path = TARGET_DIR / filename
            s3.download_file(BUCKET, key, str(target_path))

            print(f"downloaded {ensg_id}: {filename}")
            ensg_seen.add(ensg_id)
            count += 1

            if count >= limit:
                print(f"downloaded {count} unique ENSG files.")
                return

    # print(f"downloading done!!! -> {count} unique ENSG_id cell images downloaded from OpenCell!!!.")


class EmbeddingToDensity(tf.keras.Model):
    """
    model that turns image + AA sequence embeddings to density plot maps
    """
    def __init__(self):
        super(EmbeddingToDensity, self).__init__()
        self.dense_proj = tf.keras.layers.Dense(64*64, activation='relu')  # reduce sequence length
        self.reshape_layer = tf.keras.layers.Reshape((64, 64, 1))
        self.upsample = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),  # 64->128
            tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),  # 128->256
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')  # output map
        ])
    

    def call(self, x):
        x = tf.reduce_mean(x, axis=-1)  # (batch_size, seq_len) -> summarize embeddings
        x = self.dense_proj(x)           # (batch_size, 64*64)
        x = self.reshape_layer(x)        # (batch_size, 64, 64, 1)
        x = self.upsample(x)             # (batch_size, 256, 256, 1)
        return x


def overlay_density_on_image(image, density_map, fname, alpha=0.4, cmap='jet'):
    """
    overlays a density map onto an image.

    Args:
        image: Tensor of shape (256, 256, 1) or (256, 256, 3)
        density_map: Tensor of shape (256, 256, 1)
        alpha: Transparency of the density map (0 = only image, 1 = only density)
        cmap: Colormap to use for the density map
    """
    # convert tensors to numpy 
    image = image.numpy().squeeze()
    density_map = density_map.numpy().squeeze()

    # normalize images to 0-1
    image = (image - image.min()) / (image.max() - image.min())

    # normalize density map to 0-1
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())

    # plots the overlayed density maps and images
    plt.imshow(image, cmap='gray')
    plt.imshow(density_map, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.savefig(f'images/{fname}.png', dpi=300, bbox_inches='tight')


def tensor_to_image(tensor):
    """
    converts a tensor to an image
    """
    tensor = tf.convert_to_tensor(tensor)

    # normalize if needed (optional, depending on how your tensor looks)
    if tf.reduce_max(tensor) > 1.0:
        tensor = tensor / tf.reduce_max(tensor)

    # if img is grayscale, expand dims to 3 channels
    if len(tensor.shape) == 2:
        tensor = tf.expand_dims(tensor, axis=-1)

    tensor = tf.clip_by_value(tensor, 0.0, 1.0) # make sure values are between 0-1
    return tensor



def compute_emd(nucleus_data_list, density_maps):
    """
    computes Earth Mover's Distance given a dataset and density maps (metric to compare two density distributions)
    """
    # stack the list into a proper tensor
    nucleus_data = tf.stack(nucleus_data_list, axis=0)  # (batch, 256, 256)

    nucleus_flat = tf.reshape(nucleus_data, [tf.shape(nucleus_data)[0], -1])
    density_flat = tf.reshape(density_maps, [tf.shape(density_maps)[0], -1])

    nucleus_flat /= tf.reduce_sum(nucleus_flat, axis=1, keepdims=True) + 1e-8
    density_flat /= tf.reduce_sum(density_flat, axis=1, keepdims=True) + 1e-8

    nucleus_cdf = tf.cumsum(nucleus_flat, axis=1)
    density_cdf = tf.cumsum(density_flat, axis=1)

    emd_per_sample = tf.reduce_mean(tf.abs(nucleus_cdf - density_cdf), axis=1)

    return emd_per_sample


def save_img(img, fname):
        """
        saves images to a file! 
        """
        image = tensor_to_image(img)
        plt.imshow(tf.squeeze(image), cmap='gray') # squeeze removes extra channel if needed
        plt.axis('off')
        plt.savefig(f'images/{fname}.png')
