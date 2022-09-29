# 2. You are given a classifier that reports high accuracy on the validation set:

## a. Would you be happy with these results or would you like to do more analysis.

* Maybe, but accuracy is not everything. Although it can be the main metric to look up if we have a nice and balanced dataset, it's not a perfect representation of the behaviour of the model and it can be interesting to do some other observations. If those check ups are good then yes, I would be happy.

## b. If so, what type of analysis would you perform?

* The first thing I will do in order to confirm the good results in accuracy, it's to contrast with the F1 Score wich takes in count the Recall of the model. This way, we give more importance to False Negatives, wich are mucho more "dangerous" if those predictions lead to health or security related decisions for example.

* Besides that, there is always the problem with the dataset: in a way, there is always room for improvement and trying to balance every feature to evoiding bias.