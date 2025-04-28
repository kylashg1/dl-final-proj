import tensorflow as tf
import keras
import pandas as pd
from PIL import Image
import requests
import os
import re
import numpy as np
import random
import downloader as cell_downloader
from transformers import BertTokenizer, TFBertModel
import torch

class dataloader():
    def __init__(self):
        # dict that maps amino acid sequence to nucleus_img_path and protein_img_path
        self.data = {}
        self.ENSG = set()


    def download_cell_imgages(self, limit=3):
        """
        Script to download cell images from the OpenCell dataset
        """
        cell_downloader.download_from_OpenCell(limit=3)


    def cell_img_splitter(self, img_path: str, output_folder:str):
        """
        Converts the cell images in tiff image format to png files

        img_path: file path to cell images in tiff format; each tiff img file contains 2 frames,
            the first frame corresponds to nucleus cell imgs and the second frame represents the
            protein cell imgs
        output_path: file path to store the png files
        """
        relative_path =  img_path
        folder = output_folder

        file_name = os.path.basename(relative_path)
        img = Image.open(relative_path)

        pattern = r"ENSG\d+"
        ensg_id = re.findall(pattern, file_name)
        if ensg_id[0] not in self.data.keys():
            self.data[ensg_id[0]] = []
            self.ENSG.add(ensg_id[0])

        for i in range(img.n_frames):
            img.seek(i)
            if i==0: # nuclus img
                new_file_name = f"{folder}/nucleus_img_path_{file_name}.png"
                new_file_name = new_file_name.replace(".tif", "")
                if new_file_name not in output_folder:
                    print(f"new nucleus img converted")
                    img.save(new_file_name)
                    self.data[ensg_id[0]].append(new_file_name)
            elif i==1: # protein img
                new_file_name = f"{folder}/protein_img_path_{file_name}.png"
                new_file_name = new_file_name.replace(".tif", "")
                if new_file_name not in output_folder:
                    print(f"new protein img converted")
                    img.save(new_file_name)
                    self.data[ensg_id[0]].append(new_file_name)


    def populate_inputs(self, input_folder_path: str, output_folder_path:str):
        folder_path = input_folder_path
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(folder_path, filename)
                # print(full_path)
                self.cell_img_splitter(img_path=full_path, output_folder=output_folder_path)


    def populate_csv(self, csv_path:str):
        """
        Main func to preprocess data ->
        Fills the model's csv input file with nuclus_img_path, protein_img_path, and amino_acid_sequence
        """
        for ensg_id in self.ENSG:
            amino_acid_sequence = self.get_amino_acid_sequence(ensg_id)
            self.data[ensg_id].append(amino_acid_sequence)
        
        # print(self.data)
        df = pd.DataFrame(self.data)
        df = df.transpose()
        df.columns = ['nucleus_img_path', 'protein_img_path', 'amino_acid_seq']

        df.to_csv(csv_path, index=False)
        print(f"{csv_path} populated!")


    def get_amino_acid_sequence(self, ENSG_ID: str):
        """
        retrieves amino acid sequence from an ENSG_ID for a given protein from the emsembl database
        returns a string of the full amino acid sequence
        """
        base_url = "https://rest.ensembl.org"
        url = f"{base_url}/sequence/id/{ENSG_ID}?content-type=application/json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            amino_acid_sequence = data.get('seq', None)

            if amino_acid_sequence:
                # print(f"aa seq {amino_acid_sequence}")
                return amino_acid_sequence


class OpenCellLoaderTF():
    def __init__(self, data_path, crop_size=64):
        self.df = pd.read_csv(data_path)
        self.crop_size = crop_size
        self.root = os.path.dirname(data_path)

        # 20 regular amino acids + 7 'special' ambiguous amino acid characters
        char_list = [
        "-", "M", "R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "U", "G",
        "P", "A", "V", "I", "F", "Y", "W", "L", "O", "X", "Z", "B", "J"
        ]
        self.char_dict = {char: i for i, char in enumerate(char_list)}

        # commonly used tokenizer and model for AA sequences
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert") 
        self.bert_model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)


    def _crop_and_augment(self, nucleus, target):
        """
        randomly crops a pair of nucleus and target images (to prevent model overfitting); also randomly flips images
        """
        # randomly crops nucleus and target images
        h, w, _ = nucleus.shape
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)
        nucleus = nucleus[i:i+self.crop_size, j:j+self.crop_size, :]
        target = target[i:i+self.crop_size, j:j+self.crop_size, :]

        # randomly flips images
        if random.random() > 0.5:
            nucleus = np.fliplr(nucleus)
            target = np.fliplr(target)
        if random.random() > 0.5:
            nucleus = np.flipud(nucleus)
            target = np.flipud(target)

        return nucleus, target
    

    def _encode_sequence(self, seq):
        """
        encodes an amino acid sequence into tokens (numerical representation that the model can understand)
        :seq: input amino acid sequence
        :return: tensor of a tokenized amino acid sequence with dimensions (1, 1001, 256)
        """
        processed_seq = " ".join(seq)
        encoded_seq = self.tokenizer(processed_seq, padding="max_length", truncation=True, max_length=1001, return_tensors="tf") 
        protein_embeddings = self.bert_model(encoded_seq.input_ids).last_hidden_state # (1, 1001, 1024)

        # need to project to 256-dim embedding to match VQGAN embeddings
        projection_layer = tf.keras.layers.Dense(64)
        protein_embeddings = projection_layer(protein_embeddings) # (1, 1001, 256)
        
        return protein_embeddings

    
    def _load_image(self, path):
        """
        loads and converts a cell image into normalized pixel values and returns them as a tensor
        """
        image = Image.open(path) # img are already in greyscale
        image = np.array(image, dtype=np.float32) / 65535.0 # original imgs are 16-bit
        return image[..., np.newaxis]  # add channel dim


    def _generator(self):
        """
        generates the vectorized dataset that the model will use
        """
        for _, row in self.df.iterrows():
            nucleus_path = os.path.join(self.root, row["nucleus_img_path"])
            target_path = os.path.join(self.root, row["protein_img_path"])

            nucleus = self._load_image(nucleus_path)
            target = self._load_image(target_path)

            nucleus, target = self._crop_and_augment(nucleus, target)
            threshold = (target > target.mean()).astype(np.float32)

            seq = row["amino_acid_seq"]
            seq_tensor = self._encode_sequence(seq)

            yield {
                "nucleus": nucleus.astype(np.float32),
                "target": target.astype(np.float32),
                "threshold": threshold.astype(np.float32), 
                "sequence": seq_tensor.numpy()
            }


    def get_dataset(self):
        """
        returns the vectorize dataset the model will use
        """
        output_signature = {
            "nucleus": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32, name="nucleus"),
            "target": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32, name="target"),
            "threshold": tf.TensorSpec(shape=(self.crop_size, self.crop_size, 1), dtype=tf.float32, name="threshold"),
            "sequence": tf.TensorSpec(shape=(1, 1001, 64), dtype=tf.int64, name="sequence")
        }
        return tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)