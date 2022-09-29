# 1. Design & build a small dataset (about 100+ images) to differentiate between real and fake face images. Please explain:

## a. Considerations that went into deciding what data to collect.

* A 50/50 distribution of real/fake faces seems to be a good mesure to start with a balanced dataset. As the asked dataset is small, there is no need to add an issue of unbalanced data.
* The real and fake faces must have some kind of relation, otherwise it's very unlikely the classifier (working as a discriminator) will work.
* Evoiding some kind of bias (this one is very dificult). The dataset will have to be balanced in more than one label (real/fake) but also regarding common facial features between the variety of faces: ethnia, eye colour/shape, tone of skin, etc... At the very least, the model will have to be used for classify faces with the same kind of facial features appearing in the dataset.

## b. How you went about collecting the data.

First of all, I start searching for available data that might be usefull for this assignment.
I found a Kaggle dataset with 70k real faces and 1M fake faces generated with StyleGAN (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). Instead of building the dataset on my own and take the risk of high bias, I subset this dataset using:
* 50 real images from train
* 50 fake images from train
* 10 real images from test
* 10 fake images from test
* 10 real images from valid
* 10 fake images from valid


## c. Besides fake/real labels, what other labels would you consider? Explain a simple method to sample a uniform dataset in the i.i.d sense, given the labels.

As I said before, the dataset would ideally have to be balanced in certain facial features to evoid biases (wich can be seen as racism depending on the use cases). This balance is very important to build the dataset, but not that much for the classifier. At least not in the output layer but indeed important for the feature maps for example.

## d. What API (e.g Pandas, etc.) you used to store and organize meta information about the dataset.

Pandas.

## e. Please share your mini-dataset as a zip file.

The subset has to be extract in a folder named data at the source of the repository. 
A notebook is available in question_01 folder (data_sampling.ipynb) to see the basic process of subsetting the original dataset.

repo
--- data
------ subset
---------- real
---------- fake

The .zip file for the obtained subset will be sent by email.
