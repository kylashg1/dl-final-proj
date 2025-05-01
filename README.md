# dl-final-proj
“Transforming” Protein sequences into images humans can understand!

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
