import tensorflow as tf
import pandas as pd
from PIL import Image
from collections import defaultdict
import requests
import os
import re
import downloader as cell_downloader

class dataloader():
    def __init__(self):
        # dict that maps amino acid sequence to nucleus_img_path and protein_img_path
        self.data = {}
        self.ENSG = set()


    def download_cell_imgages(self):
        """
        Script to download cell images from the OpenCell dataset
        """
        cell_downloader.download_from_OpenCell(limit=10)


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
        df.columns = ['nuclus_img_path', 'protein_img_path', 'amino_acid_seq']

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


    # def preprocess_data():
    #     """
    #     crops images to fit the model's parameters
    #     """