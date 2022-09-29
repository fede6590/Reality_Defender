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

## DATASET

In question_01, there is a notebook not made to be RUN but to explain the simple subsetting process.
The "data" folder, should contain the subset folder (provided by link and mail) as well as the rd_test_dataset folder (from the provided .zip).

REPO
--- data
------ subset
--------- real
--------- fake
------ rd_test_dataset
--------- 001.jpg
--------- 002.jpg
--------- 003.jpg
--------- 004.jpg
...

## TRAINNING

Please use the terminal to run the following command from the src of the repo:

python3 model/train.py model/config.yml


## EVALUTION

Notebook in model folder named Evaluation_Test_DS.ipynb
Open it and just run from start to end to obtain accuracy on test and the test_df.csv file in question_03 folder.