import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import json
import random

CHAR_LIST = [
    "-", "M", "R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "U", "G",
    "P", "A", "V", "I", "F", "Y", "W", "L", "O", "X", "Z", "B", "J"
]
CHAR_DICT = {char: i for i, char in enumerate(CHAR_LIST)}

DESCRIPTOR_DICT = {
    "<pad>": 0, "M": 1, "R": 2, "H": 3, "K": 4, "D": 5, "E": 6, "S": 7,
    "T": 8, "N": 9, "Q": 10, "C": 11, "G": 12, "P": 13, "A": 14, "V": 15,
    "I": 16, "F": 17, "Y": 18, "W": 19, "L": 20, "<cls>": 21
}

def simple_conversion(seq):
    return tf.convert_to_tensor([CHAR_DICT[char] for char in seq], dtype=tf.int64)

def convert_descriptor(seq, descriptor_lookup):
    seq = seq.upper()
    descriptor_list = []
    for aa in seq:
        if aa in descriptor_lookup:
            descriptor_list.append(descriptor_lookup[aa])
        else:
            descriptor_list.append(np.zeros(descriptor_lookup["A"].shape))
    return tf.convert_to_tensor(descriptor_list, dtype=tf.float32)

class OpenCellLoaderTF:
    def __init__(self, data_path, crop_size=600, crop_method="random", sequence_mode="simple", text_seq_len=0, descriptor_path=None):
        self.df = pd.read_csv(data_path)

        self.crop_size = crop_size
        self.crop_method = crop_method
        self.sequence_mode = sequence_mode
        self.text_seq_len = text_seq_len
        self.root = os.path.dirname(data_path)

        self.descriptor_lookup = None
        if self.sequence_mode == "aadescriptors":
            if descriptor_path is None:
                raise ValueError("descriptor_path must be provided for aadescriptors mode")
            descriptor_df = pd.read_csv(descriptor_path).set_index("AA")
            self.descriptor_lookup = descriptor_df.to_dict(orient="index")
            self.descriptor_lookup = {
                aa: np.array(list(values.values()), dtype=np.float32)
                for aa, values in self.descriptor_lookup.items()
            }

    def _load_image(self, path):
        """
        loads and converts a cell image into normalized pixel values and returns them as a tensor
        """
        image = Image.open(path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        return image[..., np.newaxis]  # Add channel dim

    def _crop_and_augment(self, nucleus, target):
        if self.crop_method == "random":
            h, w, _ = nucleus.shape
            i = np.random.randint(0, h - self.crop_size + 1)
            j = np.random.randint(0, w - self.crop_size + 1)
            nucleus = nucleus[i:i+self.crop_size, j:j+self.crop_size, :]
            target = target[i:i+self.crop_size, j:j+self.crop_size, :]

            if random.random() > 0.5:
                nucleus = np.fliplr(nucleus)
                target = np.fliplr(target)
            if random.random() > 0.5:
                nucleus = np.flipud(nucleus)
                target = np.flipud(target)
        return nucleus, target

    def _encode_sequence(self, seq):
        seq = seq[:self.text_seq_len]
        if self.sequence_mode == "simple":
            return simple_conversion(seq)
        elif self.sequence_mode == "aadescriptors":
            return convert_descriptor(seq, self.descriptor_lookup)
        else:
            raise ValueError("Unsupported sequence mode")

    def _generator(self):
        for _, row in self.df.iterrows():
            nucleus_path = os.path.join(self.root, row["nucleus_image_path"])
            target_path = os.path.join(self.root, row["protein_image_path"])

            nucleus = self._load_image(nucleus_path)
            target = self._load_image(target_path)

            nucleus, target = self._crop_and_augment(nucleus, target)
            threshold = (target > target.mean()).astype(np.float32)

            seq = row["protein_sequence"]
            seq_tensor = self._encode_sequence(seq)

            yield {
                "nucleus": nucleus.astype(np.float32),
                "target": target.astype(np.float32),
                "threshold": threshold.astype(np.float32),
                "sequence": seq_tensor.numpy()
            }

    def get_dataset(self):
        if self.sequence_mode == "aadescriptors":
            sequence_spec = tf.TensorSpec(shape=(None, len(next(iter(self.descriptor_lookup.values())))), dtype=tf.float32)
        else:
            sequence_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64)

        output_signature = {
            "nucleus": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32),
            "target": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32),
            "threshold": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32),
            "sequence": sequence_spec
        }
        return tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
