import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19Factory:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19Factory)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print vgg19_npy_path

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        self.param_dict = {}
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        class VGG19Object: pass
        vgg19 = VGG19Object()

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        vgg19.conv1_1 = self.conv_layer(bgr, "conv1_1")
        vgg19.conv1_2 = self.conv_layer(vgg19.conv1_1, "conv1_2")
        vgg19.pool1 = self.max_pool(vgg19.conv1_2, 'pool1')

        vgg19.conv2_1 = self.conv_layer(vgg19.pool1, "conv2_1")
        vgg19.conv2_2 = self.conv_layer(vgg19.conv2_1, "conv2_2")
        vgg19.pool2 = self.max_pool(vgg19.conv2_2, 'pool2')

        vgg19.conv3_1 = self.conv_layer(vgg19.pool2, "conv3_1")
        vgg19.conv3_2 = self.conv_layer(vgg19.conv3_1, "conv3_2")
        vgg19.conv3_3 = self.conv_layer(vgg19.conv3_2, "conv3_3")
        vgg19.conv3_4 = self.conv_layer(vgg19.conv3_3, "conv3_4")
        vgg19.pool3 = self.max_pool(vgg19.conv3_4, 'pool3')

        vgg19.conv4_1 = self.conv_layer(vgg19.pool3, "conv4_1")
        vgg19.conv4_2 = self.conv_layer(vgg19.conv4_1, "conv4_2")
        vgg19.conv4_3 = self.conv_layer(vgg19.conv4_2, "conv4_3")
        vgg19.conv4_4 = self.conv_layer(vgg19.conv4_3, "conv4_4")
        vgg19.pool4 = self.max_pool(vgg19.conv4_4, 'pool4')

        vgg19.conv5_1 = self.conv_layer(vgg19.pool4, "conv5_1")
        vgg19.conv5_2 = self.conv_layer(vgg19.conv5_1, "conv5_2")
        vgg19.conv5_3 = self.conv_layer(vgg19.conv5_2, "conv5_3")
        vgg19.conv5_4 = self.conv_layer(vgg19.conv5_3, "conv5_4")
        vgg19.pool5 = self.max_pool(vgg19.conv5_4, 'pool5')

        vgg19.fc6 = self.fc_layer(vgg19.pool5, "fc6")
        assert vgg19.fc6.get_shape().as_list()[1:] == [4096]
        vgg19.relu6 = tf.nn.relu(vgg19.fc6)

        vgg19.fc7 = self.fc_layer(vgg19.relu6, "fc7")
        vgg19.relu7 = tf.nn.relu(vgg19.fc7)

        vgg19.fc8 = self.fc_layer(vgg19.relu7, "fc8")

        vgg19.prob = tf.nn.softmax(vgg19.fc8, name="prob")

        return vgg19

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        key = name + '/filter'
        if key not in self.param_dict:
            self.param_dict[key] = tf.constant(self.data_dict[name][0], name="filter")
        return self.param_dict[key]

    def get_bias(self, name):
        key = name + '/biases'
        if key not in self.param_dict:
            self.param_dict[key] = tf.constant(self.data_dict[name][1], name="biases")
        return self.param_dict[key]

    def get_fc_weight(self, name):
        key = name + '/weights'
        if key not in self.param_dict:
            self.param_dict[key] = tf.constant(self.data_dict[name][0], name="weights")
        return self.param_dict[key]
