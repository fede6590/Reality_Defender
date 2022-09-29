# RD-Data-Takehome-Federico-Ferreyra

MobileNetV2 model used for Real VS. Fake faces classification.
The data has been subseted from a Kaggle similar (and much bigger) dataset to a 140 images subset: 70 real faces and 70 fakes faces, placed in 2 two different folders (named real and fake).

## ENVIRONMENT

Pleasure be sure you have all the dependencies needed by creating and activating the required environment from the .yaml file.

Use the following command in the terminal from the src of the repo:

conda env create --file environment.yaml

That creates the required environment from environment.yaml (provided).

Then:

conda activa rd_env

## TRAINNING

Please use the terminal to run the following command from the src of the repo:

python3 model/train.py model/config.yml


## EVALUTION

Notebook in model folder named Evaluation_Test_DS.ipynb
Open it and just run from start to end.