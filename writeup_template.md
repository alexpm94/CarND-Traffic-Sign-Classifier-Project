# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image_distribution]: ./examples/distribution_1.png
[image_random]: ./examples/random.png
[image_hls]: ./examples/hls.png
[image_clahe]: ./examples/clahe.png
[image_augmentation]: ./examples/augmentation.png
[image_newDistribution]: ./distribution_2.png
[images_web]: ./examples/web.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution per class. 

![alt text][image_distribution]
It's clear that the image set is heavily unbalanced.

I also plot 18 random images in order to see similarities and differences before starting the image preporcess.

![alt text][image_random]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to homogenize lighting conditions. I firstly decided to equalize the image in RGB. To achieve this, I converted the image to HLS, I equalize the L component an then revert it back to RGB.

Here is an example of a traffic sign image before and after equalization.

![alt text][image_hls]

However, I decided to convert to grayscale because Traffic signs do not rely on the colors but on the shape of each figure. 
I've worked with object detection and the tensorflow API for object detection. In my experience, Contrast Limit Adaptative Histogram Equalization (CLAHE) gives a good performance. Here is an image of the 18 images shown above after CLAHE. 

![alt text][image_clahe]

I decided to generate data augmentation because the data set is clearly unbalanced, ther are classes with up to 2000 images whereas there are some with less than 200.

To add more data to the the data set, I used 4 different techniques
1. Image rotation (between [-25,25] degrees)
2. Random noise.
3. Image translation.
4. Mix of the 3 above.

Here is an example of an original image and an augmented image:

![alt text][image_augmentation]

I calculated the mean distribution per class, every class that is under the mean was augmented by a factor of 2.5.
The data set was augmented by N images. Now we can see the following distribution.

![alt text][image_newDistribution]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten				| Input 400										|
| Fully connected		| Output 200 									|
| RELU					|												|
| Dropout				| 0.6 when training								|
| Fully connected		| Output 90    									|
| RELU					|												|
| Dropout				| 0.6 when training								|
| Softmax				| Output 43    									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer which is similar to Stochastic Gradient descent. I used a batch size of 256, I found out that with a bigger batch I reached a better performance but the training was slower. I decided to take 20 epochs as if I increase the number to 50 epochs, it tended to reach an asymptote around 0.97.About the learning rate, I took different values, I started with 0.001 but it is too small, it requieres up to 50 epochs to reach a good performance. I

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I tried LeNet architecture, from the course. From the beginning I realize that color on images was not the most relevant and I knew I would work with grayscale. As the data set was in gray scale and the images were the same resolution, I decided to take the same architecture. 

I tried to tweak different parameters from the traing data to the drop out. Firstly, without data augmentation I got a validation accuracy of 0.93 after 20 epochs. Then I tried data augmentation, I double the images of the classes that were under the mean and only with ratation, I got almost the same 0.934. Then I increase the data by factor of 3 instead of 2 and added different data transformation (noise, translatiom, mix oh them) and I got 0.96 after 50 epochs. 

The biggest difference was when I applied dropout to the fully connected layers. I reached 0.96 in the first 10 epochs. As mentioned in the course, I started with a dropout of 0.5, however the accuracy was very small at the first epoch, I decided to drop less characteristics by setting the dropout at 0.6

My final model results were:
* validation set accuracy of 0.971
* test set accuracy of 0.949


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web and resized to 32x32x3.

![alt text][images_web]

The first image might be difficult to classify it shows 2 signals, so the principal sign "General caution", is smaller than the ones used for training. The second one seems a bit distorted afer resizing. The rest seem ok.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution  		| Roundabout madatory   						| 
| Turn right ahead 		| Priority Road 								|
| Slippery Road			| Beware of ice									|
| Right of way   		| Right of way 					 				|
| Yield     			| Yield               							|
| Bumpy Road  			| Bumpy Road          							|
| Priority Road 		| Priority Road      							|
| Speed limit 30Km/h    | Speed limit 30Km/h      						|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 0.625%. This is far from the testing and validation accuracy. The classes that wer well identified had an initial daset up to the mean distribution per class, they was not augmented. For classes that were augmented, maybe they need more real data instead of fake. In addition, I generated fake data by a factor of 3, so if the random transformations were not different enough, it may cause overfitting for these particular classes.

Another difficulty is the resolution of the images, in the case of slippery road, It's hard to identify even by my eye. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I will show only the top 3 probabilities, as the 1st probability is 0.9 in most cases.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99 e-1     			| 40   Roundabout mandatory						| 
| 7.08 e-6    			| 37   Go straight or left						|
| 3.39 e-6  			| 12   Priority road							|


For the second image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.64 e-1     			| 12  Priority road								| 
| 5.50 e-2  			| 33  Turn right ahead  						|
| 2.85 e-2				| 35  Ahead only								|


For the third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.37 e-1     			| 30  Beware of ice/snow						| 
| 2.91 e-1  			| 29  Bicycles crossing							|
| 1.37 e-1				| 40  Roundabout mandatory						|


For the fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99 e-1     			| 11  Right-of-way at the next intersection		| 
| 5.18 e-7  			| 30  Beware of ice/snow						|
| 9.14 e-8				| 21  Double curve								|

For the fifth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99 e-1     			| 13  Yield  									| 
| 1.15 e-7  			| 35  Ahead only								|
| 2.35 e-8				| 15  No vehicles								|


For the sixth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.42 e-1     			| 22  Bumpy road								| 
| 3.78 e-2  			| 26  Traffic signals							|
| 4.87 e-3				| 29  Bicycles crossing							|


For the seventh image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99 e-1     			| 12  Priority road								| 
| 1.10 e-5  			| 40  Roundabout mandatory						|
| 2.43 e-6				| 7	  Speed limit (100km/h)						|


For the egiht image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.85 e-1     			| 1  Speed limit (30km/h)						| 
| 1.17 e-2  			| 2  Speed limit (50km/h)						|
| 1.78 e-3				| 5	 Speed limit (80km/h)						|




