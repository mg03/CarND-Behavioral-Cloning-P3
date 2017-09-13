**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/timeseries.jpg
[image2]: ./output_images/anglehistogram.jpg
[image3]: ./output_images/augmentedimage.jpg
[image4]: ./output_images/sampleimagecenterleftright.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* dataexploration.ipynb help visualize the data

2. This was an interesting project since I had an oppurtunity to play with the sim and collect my own data. My own data was jerky, so fall back was udacity provided data. The udacity provided data had a huge bias towards 0 or near 0 steering angles. See images below

Sample Image taken from Center, Left and Right
![alt text][image4]

Steering Angle Time series
![alt text][image1]

Steering Angle Histogram Distibution
![alt text][image2]

To overcome data imbalance, I did up sampling of data with non-zero steering angles by random selection of either center, left or right images and then based on np.random decide to flip image or not.

With this the car went ok on straight roads but didnt do well on turns, hence captured recovery data at the turns and some tricky patches of the test track. Did capture a few times, ended up using the capture which had smoother recovery in it, the jerkier the data the worse the sim did in auto mode.

3. In addition to above, I did some image modification like cropping image to remove sky and hood, random brightness changes in HSV space.

Below image shows different image augmentations utilized

![alt text][image3]


4. I am using function data_generator(), to generate train and validation sets on the fly. This is to help with memory utilization and gpu efficiency by allowing to augment images while training the model in parallel.

5. Since I used adam optimizer, I didnt feel the need to change the Learning rate from the default of 0.001. I did test with 0.0001 and 0.0005, but didnt like what I saw hence reverted to the default. Infact didnt deviate much from the defaults for most of the parameters for the adam optimizer. 

6. Architecture

Used Nvidia Model as given in [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
This paper helped me get clarity on the Nvidia architecture , but obviously this project does not incorporate all its suggestions/directions.

Layers of Nvidia Architecture

- Input Layer is 66 x 220 x 3
- Normalization Layer
- Convolutional Layer 1: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 3 x 66 x 200
- Convolutional Layer 2: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 24 x 31 x 98
- Convolutional Layer 3: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 36 x 14 x 47
- Convolutional Layer 4: 5 x 5 Kernel, 3 x 3 stride, valid padding, Output: 48 x 5 x 22
- Convolutional Layer 5: 5 x 5 Kernel, 3 x 3 stride, valid padding, Output: 64 x 3 x 20
- Flatten Layer 1: Output: 64 x 1 x 18
- Fully-Connected Layer 1: Output 1152 neurons
- Fully-Connected Layer 2: Output 100 neurons
- Fully-Connected Layer 3: Output 50 neurons
- Fully-Connected Layer 3: Output 10 neurons
- Final Output: 1

7. Training

Data is split into train and validation sets ( 80/20 split). Overfitting has been avoided by doing the following:

- Random selection of data from each row
- Random change in brightness of image
- Random flip of image and change sign of steering angle to simulate the opposite turn

While training, I did see that I was getting low error on training and validation sets, but the sim would get off track. I tried tuning various parameters like: epoch size, batch size, etc. Nothing helped as much as capturing and using the recovery data.

8. The output video of sim in auto mode on Track One

- [Drive Track One](https://youtu.be/R8C0DUNY4O8). Recording was done on mac os using Quicktime recording software.

