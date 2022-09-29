# 4. Now, consider the case where you had to manage a dataset with millions images rather than a few hundred. How will you change your dataset building and storing methods for:


## a. Faster access, given that data data lives on the cloud infrastructure like S3

One way to make more agile the access is to use a Cloud Service with the different funcionalities provided. For example, training and testing on AWS EC2 Servers using S3 Storage at the same time is easy to handle and quick to execute. You can also work with batches of data from your millions of images in order to obtain i.i.d parameters on separate parts of your data.

## b. Faster data re-sampling, to create custom datasets

I supose Styles GANs can help for this. There are a lot of ways to create Custom Dataset from synthetic data but I don't know much about this topic.

## c. Faster data-loader access for faster training

I don't know a particular technique to implement that, but the bottleneck training has a strong relation with hardware. So, beeing update about new applying technologies is always a plus. Trying different kind of access like passing from secuential to random access can lead to faster training as well. And finally, adapting long and heavy pipelines to low languages like C++ it's also a good way to evolve as I see it.