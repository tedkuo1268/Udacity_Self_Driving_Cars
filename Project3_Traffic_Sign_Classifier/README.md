# **Traffic Sign Classifier** 
---

[//]: # (Image References)

[image1]: ./images/data_hist.png "dataset histogram"
[image2]: ./images/gan.pdf "GAN architecture"
[image3]: ./images/fake_image1.jpg "Fake image 1"
[image4]: ./images/fake_image2.jpg "Fake image 2"
[image5]: ./images/fake_image3.jpg "Fake image 3"
[image6]: ./images/fake_image4.jpg "Fake image 4"
[image7]: ./images/fake_image5.jpg "Fake image 5"
[image8]: ./images/original_image.jpg "image without preprocessing"
[image9]: ./images/preprocessed_image.jpg "precessed image"
[image10]: ./images/new_test_images.png "New test images"

### Data Set Summary & Exploration
---

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Below are summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Below is the histogram of the training dataset. We can see that the dataset is highly imbalanced, and this negatively affects the accuracy of the classifier. To overcome this, we can train the classifier on an augmented training dataset which can lead to higher classifying accuracy. The detail on how to generate augmented dataset will be explained in the next part.

![alt text][image1]

### Design and Test a Model Architecture
---

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As mentioned in the above section, the dataset is highly imbalanced, so we need to generate additional data to the minority classes to balance the dataset. Here I implement deep convolutional generative adversarial networks (DCGANs) to generate fake images to the dataset. The basic GAN architecture is as the following figure. The generator takes Gaussian noise and created samples matching the dimension of the training samples. The discriminator takes samples from training dataset as well as generated samples and attempts to recognize if a sample is real (i.e. coming from training set) or fake (i.e. generated one). 

![alt text][image2]

To balance the dataset, I augmented the dataset by generating fake images to all the minority classes until they have 1000 examples. Here are some examples of augmented images:

![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]
![alt text][image7] 

For data preprocessing, I did the following steps:

* Convert the image from RGB color space to YUV color space, and use only Y channel for image recognition. The result in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) shows that using only Y channel performs better than using RGB colored image.

* Increase the global contrast of images by using histogram equalization. 

* Standardize the images to have zero mean and standard deviation of one.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image8] ![alt text][image9]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        		  	            | 
|:---------------:|:-------------------------------------------:|
| Input         	| 32x32x3 RGB image   						         	  | 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x38 |
| ReLU			      |										                 	      	|
| Dropout					|	keep probability = 0.6				           		|
| Max pooling	    | 2x2 stride,  outputs 14x14x38 			       	|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x64 |
| ReLU				   	|												                      |
| Dropout				 	|	keep probability = 0.6					           	|
| Max pooling	    | 2x2 stride,  outputs 5x5x64             	  |
| Fully connected	| inputs: 400, outputs: 120                   |
| Fully connected	| inputs: 120, outputs: 84                    |
| Fully connected	| inputs: 84, outputs: 43                     |

On top of that, I also implemented boosting to get an ensemble model to achieve higher accuracy. The boosting algorithm here is really simple: multiply the softmax probabilities of each class from all the "weak learners" and choose the largest probability to be the final prediction.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, which is proved to be more computationally efficient and has little memory requirements from the [paper](https://arxiv.org/pdf/1412.6980.pdf). 

Other hyperparameters are as follows:
* batch size: 128
* number of epochs: 30
* learning rate: 0.001

For model boosting:
* number of weak learners: 5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results (with boosting) were:
* training set accuracy of 99.7%
* validation set accuracy of 97.8%
* test set accuracy of 96.5%

To get the final model, I compared the following three different network architectures. 

* Original LeNet-5 architecture: Two convolutional layers and three fully connected layers
* Modified LeNet-5 architecture: Also two convolutional layers and three fully connected layers, but in the convolutional layers, the feature sizes are 38 and 64.
* Two-stage architecture: Two convolutional layers and two fully connected layers, and both the features' output from first and second convolutional layers are concatenated and fed to the fully connected layer.The feature sizes of two convolutional layers are also 38 and 64. This architecture is based on this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Here are the accuracies of three architectures:

| Architecture    	     | Training Accuracy | Validation Accuracy | 
| ---------------------- |:-----------------:| -------------------:|
| Original LeNet-5       |  	   98.1%       |        94.2%        |
| Modified LeNet-5       |       99.7%       |        96.7%        |
| Two-stage architecture |			 99.9%       |        94.8%        |

From the above table, we can see that the validation accuracy of the original LeNet-5 architecture is just over 94%, and the training accuracy is also just about 98%. This underfitting may due to the insufficient extracted features, so I modified the architecture to have deeper filter depth to extract more features from the image. The result shows a significant improvement of extracting more features, which has the training accuracy of 99.7%, and validation accuracy of 96.7%. Following this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I construct the final architecture that concatenate the outputs of two convolutional layers and feeds to the fully connected layer. In doing so, the classifier is fed with both  "global" shape and structure from the second stage and "local" motifs with more precise detail from the first stage. 

To prevent overfitting, I added dropout layers to all the layers with keep probability of 0.6.

Here are the accuracies of three architectures with **boosting algorithm**:

| Architecture    	     | Training Accuracy | Validation Accuracy | 
| ---------------------- |:-----------------:| -------------------:|
| Original LeNet-5       |  	   99.2%       |        96.4%        |
| Modified LeNet-5       |       99.9%       |        97.8%        |
| Two-stage architecture |			 100.0%      |        96.3%        |

The result shows that by using ensemble methods, the model achieve higher validation accuracy than a single "weak learner" does. The boosting algorithm is explained in section 2.

Based on the above table, I chose **Modified LeNet-5** as the final model, and the accuracy on the test dataset is **96.5%**.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]

The first image might be difficult to classify because of its orientation. The last image might also be difficult to classify because the sign is covered with snow.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			           |     Prediction	      | 
|:--------------------:|:--------------------:| 
| Turn right ahead	   | Speed limit (50km/h) | 
| Speed limit (30km/h) | Speed limit (30km/h) |
| Stop sign			    	 | Stop sign    			  |
| Keep right	         | Keep right           |
| Slippery road        | Speed limit (80km/h) |

The test accuracy on new images is 60%, which is below the accuracy on the test set. One of the main reason is that there are only five images so the accuracy is likely to be very different from the accuracy on large dataset. Also, the wrongly predicted images are the images that are difficult to classify as mentioned in the previous part.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

* First image (True label: Turn right ahead)

| Probability | Prediction	                          | 
|:-----------:|:-------------------------------------:| 
| 12.1%       | Speed limit (50km/h)                  | 
| 8.9%    	  | Children crossing                     |
| 8.0%				| Speed limit (80km/h)		  					  |
| 6.1%	   		| Beware of ice/snow                    |
| 5.7%			  | Right-of-way at the next intersection |


* Second image (True label: Speed limit (30km/h))

| Probability | Prediction	          | 
|:-----------:|:---------------------:| 
| 38.1%       | Speed limit (30km/h)  | 
| 22.0%    	  | Speed limit (20km/h)  |
| 14.9%				| Speed limit (80km/h)	|
| 6.8%	   		| Speed limit (70km/h)  |
| 5.7%			  | Speed limit (120km/h) |


* Third image (True label: Stop sign)

| Probability | Prediction	          | 
|:-----------:|:---------------------:| 
| 40.7%       | Stop sign             | 
| 10.1%    	  | Speed limit (120km/h) |
| 4.1%				| Speed limit (80km/h)	|
| 3.4%	   		| Speed limit (60km/h)  |
| 2.9%			  | No vehicles    			  |


* Fourth image (True label: Keep right)

| Probability | Prediction	                                 | 
|:-----------:|:--------------------------------------------:| 
| 99.8%       | Keep right                                   | 
| 0.1%    	  | Turn left ahead                              |
| 0.0%				| No vehicles							                  	 |
| 0.0%	   		| Dangerous curve to the right                 |
| 0.0%			  | No passing for vehicles over 3.5 metric tons |


* Fifth image (True label: Slippery road)

| Probability | Prediction	          | 
|:-----------:|:---------------------:| 
| 8.1%        | Speed limit (80km/h)  | 
| 7.0%    	  | Speed limit (120km/h) |
| 5.6%				| Traffic signals			  |
| 4.9%	   		| Speed limit (50km/h)  |
| 4.9%			  | Children crossing     |
