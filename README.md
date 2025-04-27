# dl-final-proj
“Transforming” Protein sequences into images humans can understand!

## Requirements
``` bash
python3 -m venv dl-final-proj && source dl-final-proj/bin/activate && pip install --upgrade pip && \
pip install numpy pandas matplotlib tensorflow[and-cuda] rotary-embedding-tensorflow einops boto3 transformers
```

## Dataset
We used OpenCell, which can be downloaded from https://opencell.sf.czbiohub.org/download. Helper script download_tifs_proj.sh included to download images.
Need to produce csv file with the columns ```nucleus_image_path```, ```protein_image_path```, ```metadata_path```, and ```split``` (train or val).

## Usage
```
python main.py --data <data CSV file path> --epochs <number of epochs>
```

## Public implementations
https://github.com/BoHuangLab/Protein-Localization-Transformer/
