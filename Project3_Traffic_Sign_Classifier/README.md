# **Traffic Sign Classifier** 

## Writeup


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

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Below are summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Below is the histogram of training dataset. We can see that the dataset is highly imbalanced, and this negatively affects the accuracy of the classifier. To overcome this, we can train the classifier on an augmented training dataset which can lead to higher classifying accuracy. The detail on how to generate augmented dataset will be explained in the next part.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As mentioned in the above section, the dataset is highly imbalanced, so we need to generate additional data to the minority classes to balance the dataset. Here I implement deep convolutional generative adversarial networks (DCGANs) to generate fake images to the dataset. The basic GAN architecture is as the following figure. The generator takes Gaussian noise and created samples matching the dimension of the training samples. The discriminator takes samples from training dataset as well as generated samples and attempts to recognize if a sample is real (i.e. coming from training set) or fake (i.e. generated one). Finally, I augmented the dataset by generating fake images to all the minority classes until they have 1000 examples. Here are some examples of an original image and augmented image:

![alt text][image3] ![alt text][image3]
![alt text][image3] ![alt text][image3]
![alt text][image3] ![alt text][image3]

After augmenting the dataset, 

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        			               		| 
|:---------------:|:-------------------------------------------:|
| Input         		| 32x32x3 RGB image   						                 	| 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6  |
| ReLU					       |										                                 		|
| Dropout					    |	keep probability = 0.6									           		|
| Max pooling	    | 2x2 stride,  outputs 14x14x6 			           	|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| ReLU				       	|												                                 |
| Dropout				    	|	keep probability = 0.6					           						|
| Max pooling	    | 2x2 stride,  outputs 5x5x16             				|
| Fully connected	| inputs: 400, outputs: 120                   |
| Fully connected	| inputs: 120, outputs: 84                    |
| Fully connected	| inputs: 84, outputs: 43                     |

On top of that, I also implemented boosting to get an ensemble model to achieve higher accuracy. The boosting algorithm here is really simple: multiply the softmax probabilities of each class from all the "weak learners" and choose the largest probability to be the final prediction.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, which is proved to be more computationally efficient and has little memory requirements from the ![https://arxiv.org/pdf/1412.6980.pdf]papaer. 

Other hyperparameters are as follows:
* batch size: 128
* number of epochs: 25
* learning rate: 0.001

For model boosting:
* number of weak learners: 10

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results (with boosting) were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

To get the final model, I compared the following three different network architectures. 

* Original LeNet-5 architecture: Two convolutional layers and three fully connected layers
* Modified LeNet-5 architecture: Also two convolutional layers and three fully connected layers, but in the convolutional layers, the feature sizes are 38 and 64.
* Two-stage architecture: Two convolutional layers and two fully connected layers, and both the features' output from first and second convolutional layers are concatenated and fed to the fully connected layer.The feature sizes of two convolutional layers are also 38 and 64. This architecture is based on this ![http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]paper.

Here are the accuracies of three architectures:
| Architecture    	      | Training Accuracy | Validation Accuracy	| 
|:-----------------------|:-----------------:|--------------------:|
| Original LeNet-5       |  	                |                     |
| Modified LeNet-5       |                   |                     |
| Two-stage architecture	|			                |                     |

From the above table, we can see that the validation accuracy of the original LeNet-5 architecture is about 93%, and the training accuracy is also just about %. This underfitting may due to the insufficient extracted features, so I modified the architecture to have deeper filter depth to extract more features from the image. The result shows a significant improvement of extracting more features, which has the training accuracy of %, and validation accuracy of %. Following this ![http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]paper. I construct the final architecture that concatenate the outputs of two convolutional layers and feeds to the fully connected layer. In doing so, the classifier is fed with both  "global" shape and structure from the second stage and "local" motifs with more precise detail from the first stage.

To prevent overfitting, I added dropout layers to all the layers.

Here are the accuracies of three architectures with boosting algorithm:
| Architecture    	      | Training Accuracy | Validation Accuracy	| 
|:-----------------------|:-----------------:|--------------------:|
| Original LeNet-5       |  	                |                     |
| Modified LeNet-5       |                   |                     |
| Two-stage architecture	|			                |                     |

The result shows that by using ensemble methods, the model achieve higher validation accuracy than a single "weak learner" does. The boosting algorithm is explained in section 2.

So I chose      as the final model, and the accuracy on the test dataset is  %.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

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
