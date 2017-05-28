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
[image1]: ./examples/hist2.jpg

## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* video.mp4 for a video of the car using model.h5 on the first track

To run the code start the simulator in autonomous mode, open shell and type
<code>python drive.py model.h5<code>
The code in model.py uses a Python generator to generate data for training rather than storing the training data in memory. 

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is based on Nvidia end-to-end model of convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 164-196). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 167).

### 2. Attempts to reduce overfitting in the model

Train/validation splits have been used with test size 0.1 (code lines 75-76).

### 3. Model parameter tuning

Adam optimizer is used (code line 193).

### 4. Appropriate training data

Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track) (code lines 23-70). 
Data rowa have been chosen where speed > 20. Data rows with angle = 0 have beebn downsampled approx. by 90 % to balance dataset.

Here is histogram of angles before downsampling angles with zero value:

![alt text][image1]

And after downsampling angles with zero value:

![alt text][image2]

Also I used left and right camera images with angle's correction:
<code> 
correction = 0.2*(1 + random.random()/2.0) 
angles.append(angle)
# Add correction for steering for left images
angles.append(angle+correction)
# Minus correction for steering for right images
angles.append(angle-correction)
<code>

