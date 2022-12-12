# Lang-8 Preprocessing

This repo contains preprocessing scripts for extracting English correction corpus from Lang-8 Learner Corpora (<https://sites.google.com/site/naistlang8corpora/>). Please use Python >= 3.9 and install the following dependencies:
```
pip install joblib pycountry fasttext nltk tqdm
```
and you need a fasttext model
```
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
or light version
```
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```