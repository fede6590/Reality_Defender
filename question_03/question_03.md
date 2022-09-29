# 3. Write a simple (supervised) deep classifier to train and test using the dataset collected in Q1.

NOTE: I used Transfer Learning on a MobileNetV2 model using Keras API for this purpose.

## a. How will you divide your dataset into training and test sets.

With a larger dataset, I will choose something like 20% of the dataset to test, but in this case the subset is pretty small so 10% could be enough to have an idea of the model behaviour.

## b. What data-augmentation techniques will to use for out-of-distribution (unseen) images?

In model/config.yml, we can see all the implementation and settings customizable for the model. That includes data-augmentation. From random zoom, contrast, rotation and flip, to dropout rate (not all mentionned are active in this case).

## c. Please test accuracy on the attached, rd_test_dataset zipped face images, and save the output to a .csv file.

https://drive.google.com/file/d/1jcdByJPkAGq9JsgsdLqeyLwI4Yl6plOf/view?usp=sharing

## d. Explain your accuracy scores, and add analysis based on what you proposal in Q2.

Like said in the Q2, 91% accuracy is good (for a small MobileNetV2 model at least), but 2 out of 22 mistakes can sometimes be a real problem (as mentioned for security or health). There is then more balancing and improvement to do on the training dataset.