# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:01:25 2016

@author: ztan6
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()

#define the initialize function
def weight_variable(shape):
    #Outputs random values from a truncated normal distribution.
    #follow a normal distribution with specified mean(default 0.0) and standard deviation(here 0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #Creates a constant tensor the value can also be a value list
    #slightly positive initial bias to avoid "dead neurons"
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


#define the convolution and pooling function
def conv2d(x,W):
    #tf.nn.conv2d(input, filter, strides, padding, ...)
    #Computes a 2-D convolution  the input and filter are 4-D
    #4-Dfilter[filter_height, filter_width, in_channels, out_channels]
    #2-Dfilter[filter_height * filter_width * in_channels, output_channels]
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    #tf.nn.max_pool(value, ksize, strides, padding, ..)
    #value 4-D Tensor[batch, height, width, channels]
    #ksize the size of window for each dimention
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
    
    
#initialize the input
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


#First Convolution Layer
#[patchsize, patchsize, input channels, output channels]
#the requirement 5 x 5 filter, 32 deep, 1 in channel
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#reshape the x to 4-D tensor 28*28=284 the last 1 is the input channels
x_image = tf.reshape(x,[-1, 28, 28, 1])

#apply the convolution and ReLU function
#tf.nn.relu(features, name=None) Computes rectified linear: max(features, 0).
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#Second Convolution Layer
#the requirement 5 x 5 filter, 64 deep, 32 in channels
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Densely Connected Layer
#initialize the variable
W_fc1 = weight_variable([7*7*64, 256])
b_fc1 = bias_variable([256])

#reshape for multiply
h_pool2_re = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_re, W_fc1)+b_fc1)

#dropout
#placeholder so could turn on when training and turn off when testing
keep_prob = tf.placeholder(tf.float32)
#tf.nn.dropout With probability keep_prob, outputs the input element scaled up by 1 / keep_prob
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Readout Layer(softmax layer)
W_fc2 = weight_variable([256, 10]) #correspond to the dimension of W_fc1 and the output dimention
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)


#the loss and the train step
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predict = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


#train the model
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob:0.5})
    

#test the model
print("test accuracy %g"%sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))