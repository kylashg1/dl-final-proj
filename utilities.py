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

def compute_emd(nucleus_tensor, density_maps):
    '''
    Computes Earth Mover's Distance - a new type of metric
    '''
    nucleus_flat = tf.reshape(nucleus_tensor, [tf.shape(nucleus_tensor)[0], -1])
    density_flat = tf.reshape(density_maps, [tf.shape(density_maps)[0], -1])

    nucleus_flat /= tf.reduce_sum(nucleus_flat, axis=1, keepdims=True) + 1e-8
    density_flat /= tf.reduce_sum(density_flat, axis=1, keepdims=True) + 1e-8

    nucleus_cdf = tf.cumsum(nucleus_flat, axis=1)
    density_cdf = tf.cumsum(density_flat, axis=1)

    emd_per_sample = tf.reduce_mean(tf.abs(nucleus_cdf - density_cdf), axis=1)

    return emd_per_sample

def save_emd(emd_scores):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Sorted line plot
    import numpy as np
    axs[0].plot(np.sort(emd_scores))
    axs[0].set_title('Sorted EMD')
    axs[0].set_ylabel('EMD')

    # 2. Histogram
    axs[1].hist(emd_scores, bins=30, color='teal', edgecolor='black', alpha=0.75)
    axs[1].set_title('EMD Score Distribution')
    axs[1].set_xlabel('EMD Value')
    axs[1].set_ylabel('Frequency')

    # 3. Boxplot
    axs[2].boxplot(emd_scores, vert=True)
    axs[2].set_title('EMD Boxplot')
    axs[2].set_ylabel('EMD')

    plt.tight_layout()
    plt.savefig('images/emd.png')
    plt.close()

def save_img(img, fname):
    '''
    Saves images to images/ folder
    '''
    image = tensor_to_image(img)
    plt.imshow(tf.squeeze(image), cmap='gray') # squeeze removes extra channel if needed
    plt.axis('off')
    plt.savefig(f'images/{fname}.png')

def save_density_map(density_map, fname, cmap='jet'):
    """
    Save just the density map
    """
    density_map = density_map.numpy().squeeze()
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    plt.imshow(density_map, cmap=cmap)
    plt.axis('off')
    plt.savefig(f'images/{fname}.png', dpi=300, bbox_inches='tight')
    plt.close()
