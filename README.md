# dl-final-proj
“Transforming” Protein sequences into images humans can understand!

## Requirements
``` bash
python3 -m venv dl-final-proj && source dl-final-proj/bin/activate && \
pip install --upgrade pip && pip install numpy pandas matplotlib tensorflow[and-cuda]
```

## Dataset
We used (Human Protein Atlas?), which can be downloaded from (https://www.proteinatlas.org/about/download).
Need to produce csv file with the columns ```nucleus_image_path```, ```protein_image_path```, ```metadata_path```, and ```split``` (train or val).

## Usage
``` bash
python main.py --data <data CSV file path> --epochs <number of epochs>
```
