# dl-final-proj
**Title**: Protein to Image Transformer: Amino Acid “Translation” for Computer Science Concentrators \
**Who**: Kylash Ganesh (kganesh), Adeethyia Shankar (ashank10), Andy Le (andyhuyle) \
**Team** Name: Neural Net-Workers

## About this project
From carrying oxygen to our brain to regulating how our DNA is expressed, proteins are an essential part of how living things operate. While this is the case, they still remain one of the great frontiers of biology that we yet to fully decode. After the introduction of AlphaFold in 2018, an abundance of protein structures have been proposed. A question from this stepping stone is where do these novel protein structures act within cells. Our project reimplements the research paper “CELL-E: A Text-to-Image Transformer for Protein Image Prediction” (Khwaja, E., Song, Y.S., Huang, B. (2024)) using TensorFlow instead of its original PyTorch implementation. The objective of this paper is to predict a refined representation of protein localization given an amino acid sequence and nucleus image, essentially allowing us to see where these novel proteins work.

## Requirements
``` bash
python3 -m venv dl-final-proj-venv && \
source dl-final-proj-venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

## Dataset
We used OpenCell, which can be downloaded from https://opencell.sf.czbiohub.org/download.

## Usage
```
python main.py --data <data CSV file path> --epochs <number of epochs>
```

## Public implementations
https://github.com/BoHuangLab/Protein-Localization-Transformer/
