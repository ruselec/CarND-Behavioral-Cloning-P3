# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./examples/hist1.jpg
[image2]: ./examples/hist2.jpg
[image3]: ./examples/preprocess.jpg
[image4]: ./examples/augment1.jpg
[image5]: ./examples/augment2.jpg
[image6]: ./examples/predict.jpg
[image7]: ./examples/loss.jpg

## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* video.mp4 for a video of the car using model.h5 on the first track

To run the code start the simulator in autonomous mode, open shell and type

<code>python drive.py model.h5</code>

The code in model.py uses a Python generator to generate data for training rather than storing the training data in memory. 

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is based on comma.ai model of convolution neural network with 8x8 and 5x5 filter sizes and depths between 16 and 64 (model.py lines 164-196). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 167).

### 2. Attempts to reduce overfitting in the model

To prevent overfitting in the model I used:

* train/validation splits with test size 0.1 (code lines 75-76)
* shuffling of training data (code line 130)
* comparing training and validation set loss by plotting results after each epoch ending and choosing the checkpoint for epoch where validation loss is not increasing (last checkpoint was chosen that matching 14 on the plot below).

![alt text][image7]

### 3. Model parameter tuning

Adam optimizer is used as it has given good results, the learning rate of 0.001 works good (code line 193).

### 4. Appropriate training data

Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track) (code lines 23-70). 
Data rows have been chosen where speed > 0. Data rows with angle = 0 have beebn downsampled approx. by 90 % to balance dataset.

Here is histogram of angles before downsampling angles with zero value:

![alt text][image1]

And after downsampling angles with zero value:

![alt text][image2]

Also I used left and right camera images with angle's correction:
```
  correction = 0.2*(1 + random.random()/2.0) 
  angles.append(angle)
  angles.append(angle+correction)
  angles.append(angle-correction)
```

## Model Architecture and Training Strategy

### 1. Solution Design Approach

For testing model Python Notebook file Visualisation_Data.ipynb was created. For solution design itterated process was used:
* set parameters of model (number of layers, activation functions), colorspace (RGB, HSV, YUV), correction for rignt and left camera images (0.05-0.4)
* training of model for 15 epochs and saving checkpoint for each epoch
* testing model chekpoints in simulator and save the best results
* come back to first par. and repeat process again

### 2. Final Model Architecture

Here is the final model architecture:

| Layer         		      |     Description	        					            | 
|:---------------------:  |:---------------------------------------------:| 
| Input         		      | 64x64x3 RGB image   							            |
| Lambda        		      | Normalization (-1, 1)  							          |
| Convolution 8x8     	  | 4x4 stride, same padding, 16 depth            |
| RELU					          |												                        |
| Convolution 5x5	        | 2x2 stride, same padding, 32 depth          	|
| RELU					          |												                        |
| Convolution 5x5	        | 2x2 stride, same padding, 64 depth       	    |
| Flatten 					      |	                                              |	
| RELU					          |												                        |
| Dense           	      | 1024 neurons							                    |
| RELU					          |												                        |
| Dropout           	    | 0.5            							                 	|
| Dense           	      | 1 neuron   							                   	  |

### 3. Creation of the training dataset and training process 

To generate more instances of training data augmentation was used (code lines 84-113, 122-123).

Here are examples of augmentation:

![alt text][image4]

![alt text][image5]

#### Image Preprocessing 

The image is cropped above the horizon and below the car to reduce the amount of information the network is required to learn. Next the image is resized to further reduce required processing (code lines 78-82, 124).

Here are examples of image preprocessing:

![alt text][image3]

#### Prediction of angle stearing for test images

Here are predictions of angle stearing by trained model:

![alt text][image6]

# Simulation

The final model was tested in simulator. The speed was limited by value 12 on sharp turns in drive.py file. 
The car went around the first and second tracks almost on the centre of the road. 
