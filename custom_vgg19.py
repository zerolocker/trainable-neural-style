import os
import sys
import tensorflow as tf

import numpy as np
import time
from tensorflow_vgg import vgg19_factory

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19Factory(vgg19_factory.Vgg19Factory):
    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, rgb, train=False):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        class VGG19Object: pass
        vgg19 = VGG19Object()

        vgg19.conv1_1 = self.conv_layer(bgr, "conv1_1")
        vgg19.conv1_2 = self.conv_layer(vgg19.conv1_1, "conv1_2")
        vgg19.pool1 = self.avg_pool(vgg19.conv1_2, 'pool1')

        vgg19.conv2_1 = self.conv_layer(vgg19.pool1, "conv2_1")
        vgg19.conv2_2 = self.conv_layer(vgg19.conv2_1, "conv2_2")
        vgg19.pool2 = self.avg_pool(vgg19.conv2_2, 'pool2')

        vgg19.conv3_1 = self.conv_layer(vgg19.pool2, "conv3_1")
        vgg19.conv3_2 = self.conv_layer(vgg19.conv3_1, "conv3_2")
        vgg19.conv3_3 = self.conv_layer(vgg19.conv3_2, "conv3_3")
        vgg19.conv3_4 = self.conv_layer(vgg19.conv3_3, "conv3_4")
        vgg19.pool3 = self.avg_pool(vgg19.conv3_4, 'pool3')

        vgg19.conv4_1 = self.conv_layer(vgg19.pool3, "conv4_1")
        vgg19.conv4_2 = self.conv_layer(vgg19.conv4_1, "conv4_2")
        vgg19.conv4_3 = self.conv_layer(vgg19.conv4_2, "conv4_3")
        vgg19.conv4_4 = self.conv_layer(vgg19.conv4_3, "conv4_4")
        vgg19.pool4 = self.avg_pool(vgg19.conv4_4, 'pool4')

        vgg19.conv5_1 = self.conv_layer(vgg19.pool4, "conv5_1")
        vgg19.conv5_2 = self.conv_layer(vgg19.conv5_1, "conv5_2")
        vgg19.conv5_3 = self.conv_layer(vgg19.conv5_2, "conv5_3")
        vgg19.conv5_4 = self.conv_layer(vgg19.conv5_3, "conv5_4")
        vgg19.pool5 = self.avg_pool(vgg19.conv5_4, 'pool5')

        return vgg19
