# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn.png "Model Visualization"
[image2]: ./examples/a0.jpg "Middle of the road"
[image3]: ./examples/a1.jpg "Recovery Image"
[image4]: ./examples/a2.jpg "Recovery Image"
[image5]: ./examples/a3.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline based on well-known nvidia model for behavior cloning I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on nvidia model and consists of a convolution neural network with 5 times 5x5 filter sizes and two times 3x3 and depths between 24 and 64 (model.py lines 74-80) 

The model includes ELU layers to introduce nonlinearity, and the data is first normalized in the model using a Keras lambda layer and (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 25-34). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have convolution layers that capable of recognising differenr characteristics of the road.

My first step was to use a convolution neural network model based on the nvidia model. I thought this model might be appropriate because my researches in the internet show that it is a well-known algoritm for behavioral cloning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I collected data using two tracks in different directions. Then I recorded sideway-rescueing attempts to teach the network recentering. 

The final step was to run the simulator to see how well the car was driving around track one. The tests seem very nice.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The NVIDIA model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![NVIDIA Model][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![At the middle][image2]
![At the middle][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![Recovery 1][image3]
![Recovery 2][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images horizontally together with multiplying the steering angle with -1.

After the collection process, I had 15000 number of data points. 
I finally randomly shuffled the data set and put 20% of the data into a validation set.  I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced during the training. I used an adam optimizer so that manually training the learning rate wasn't necessary.
