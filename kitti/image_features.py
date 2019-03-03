import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2, os
import numpy as np
import math
import time
from random import shuffle
import sys
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

#####
#Training setting

BIN, OVERLAP = 2, 0.1
W = 1.
NORM_H, NORM_W = 224, 224

#### Placeholder
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D bounding box')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'test')
    parser.add_argument('--image',dest = 'image',help='Image path')
    parser.add_argument('--label',dest = 'label',help='Label path')
    parser.add_argument('--box2d',dest = 'box2d',help='2D detection path')
    parser.add_argument('--output',dest = 'output',help='Output path', default = './validation/result_2/')
    parser.add_argument('--model',dest = 'model')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args


def build_model():

  #### build some layer
  def LeakyReLU(x, alpha):
      return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

  def orientation_loss(y_true, y_pred):
      # Find number of anchors
      anchors = tf.reduce_sum(tf.square(y_true), axis=2)
      anchors = tf.greater(anchors, tf.constant(0.5))
      anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

      # Define the loss
      loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
      loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss,axis=0))) / anchors

      return tf.reduce_mean(loss)

  #####
  #Build Graph
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    conv5 = tf.contrib.layers.flatten(net)

    #dimension = slim.fully_connected(conv5, 512, scope='fc7_d')
    dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
    dimension_int = LeakyReLU(dimension, 0.1)
    dimension = slim.dropout(dimension_int, 0.5, scope='dropout7_d')
    #dimension = slim.fully_connected(dimension, 3, scope='fc8_d')
    dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')
    #dimension = LeakyReLU(dimension, 0.1)

    #loss_d = tf.reduce_mean(tf.square(d_label - dimension))
    loss_d = tf.losses.mean_squared_error(d_label, dimension)

    #orientation = slim.fully_connected(conv5, 256, scope='fc7_o')
    orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_o')
    orientation_int = LeakyReLU(orientation, 0.1)
    orientation = slim.dropout(orientation_int, 0.5, scope='dropout7_o')
    #orientation = slim.fully_connected(orientation, BIN*2, scope='fc8_o')
    orientation = slim.fully_connected(orientation, BIN*2, activation_fn=None, scope='fc8_o')
    #orientation = LeakyReLU(orientation, 0.1)
    orientation = tf.reshape(orientation, [-1, BIN, 2])
    orientation = tf.nn.l2_normalize(orientation, dim=2)
    loss_o = orientation_loss(o_label, orientation)

    #confidence = slim.fully_connected(conv5, 256, scope='fc7_c')
    confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_c')
    confidence_int = LeakyReLU(confidence, 0.1)
    confidence = slim.dropout(confidence_int, 0.5, scope='dropout7_c')
    confidence = slim.fully_connected(confidence, BIN, activation_fn=None, scope='fc8_c')
    loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= confidence))

    confidence = tf.nn.softmax(confidence)
    #loss_c = tf.reduce_mean(tf.square(c_label - confidence))
    #loss_c = tf.losses.mean_squared_error(c_label, confidence)

    return dimension_int, orientation_int, confidence_int, conv5

class ImageFeatureExtractor:
    def __init__(self, model):
        ### buile graph
        self.dimension_int, self.orientation_int, self.confidence_int, self.conv5 = build_model()

        ### GPU config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Restore model
        saver = tf.train.Saver()
        saver.restore(self.sess, model)

    def get_features(self, img, bbox):

        [xmin,ymin,xmax,ymax] = bbox
        patch = img[max(0,math.floor(ymin)):min(math.floor(ymax),img.shape[0]), max(0,math.floor(xmin)):min(math.floor(xmax),img.shape[1])]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = patch.astype(np.float32)
        patch = cv2.resize(patch, (NORM_H, NORM_W))

        # print(patch)
        # test = patch.astype(np.uint8)
        # plt.imshow(test)
        # plt.show()

        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)
        dim_int, orient_int, conf_int, conv_int = self.sess.run([self.dimension_int, self.orientation_int, self.confidence_int, self.conv5], feed_dict={inputs: patch})

        return [dim_int, orient_int, conf_int, conv_int]
