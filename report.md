# **Traffic Sign Recognition** 
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

[image1]: ./Data/visual_my.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./Data/aug_example.png "Rotate"
[image4]: ./Data/TS2_01.jpeg "Traffic Sign 1"
[image5]: ./Data/TS2_02.jpeg "Traffic Sign 2"
[image6]: ./Data/TS2_03.jpeg "Traffic Sign 3"
[image7]: ./Data/TS2_04.jpeg "Traffic Sign 4"
[image8]: ./Data/TS2_05.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gcjyzdd/SDC-P2/blob/master/Traffic_Sign_Classifier_v2.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram of train, validation, and test data set. From the figure below, we can see that 

* all three datasets have similar distribution of traffic sign classes
* and the size of train dataset is the largest while validation dataset is the smallest
* from the distribution of the histogram, it also shows that the size of training images with class ID around 30 is smaller than others. 

Because the size of images with class ID around 30 is smaller, our trained deep neural network performs worse with cooresponding test images.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to use RGB color images as they are because three channels contains more information than greyscale. Therefore, three channels gives more freedom to the DNN to fit the dataset.

Next, I normalized the image data because pixel values could be 255, which is very big. When the input values are large, the neural network is more determined as the sigmoid function magnifies big values. Becase we want to train the DNN, it should be undetermined from the beginning. As a result, I normalized the image pixel value space to the section of [0.1, 0.9].

I decided to generate additional data because the test images of traffic sign could be rotated.

To add more data to the the data set, I used the following techniques to rotate the training images:

```
# Set upper and lower bounds of random variables
lower, upper = -30, 30
# Set the mean and standard deviation. Make the std small so that the ditribution is more uniformed
mu, sigma = 0, 0.4
# Get a random number generator
gen_rand = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
```

As putted in the comments above, the traing image is rotated randomly between -30 degree and 30 degree. The size of augmented dataset is `n_aug = np.floor(0.4*n_train).astype(int)`, i.e., 40% of the original dataset.

Here is an example of an original image and an augmented image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3     | 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9  				|
| Convolution 5x5x9	    | 1x1 stride, valid pading, outputs 10x10x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| 400x160, outputs 160   						|
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | 160x84, outputs 84                            |
| RELU                  |                                               |
| Dropout               |                                               |
| Softmax				| etc.        									|
| Fully connected       | 84x43, outputs 43                             |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]{width=200px} ![alt text][image5]{width=200px} ![alt text][image6]{width=200px} 
![alt text][image7]{width=200px} ![alt text][image8]{width=200px}

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


