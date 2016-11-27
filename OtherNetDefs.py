import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Lib

def buildTransformNetBeforeDeconvBugFixAfterPadding(img, expected_shape, useNonHackScaling=False):
    assert expected_shape == [d.value for d in img.get_shape()]   # now we fix the image size and assume 
      # img has an already-defined shape. Otherwise Tensorflow will fail to infer the output shape of the layer
        # which is very inconvinent for debugging
    
    img = tf.pad(img, [[0,0],[40,40],[40,40],[0,0]], mode='REFLECT') # the same as the paper's supp material PDF.
            # I don't know why fast-style-transfer repo doesn't do the same, which is:
            #    (1) have this padding step
            #    (2) not use padding in the residual blocks' conv layers
    with tf.variable_scope("transNet"):
        conv1 = Lib.conv_layer(img, n_in_channel=3, n_out_channel=32, filter_size=9, stride=1, hasRelu=True)
        conv2 = Lib.conv_layer(conv1, n_in_channel=32, n_out_channel=64, filter_size=3, stride=2, hasRelu=True)
        conv3 = Lib.conv_layer(conv2, n_in_channel=64, n_out_channel=128, filter_size=3, stride=2, hasRelu=True)
        res1 = Lib.residual_block(conv3, n_in_channel=128, n_out_channel=128)
        res2 = Lib.residual_block(res1, n_in_channel=128, n_out_channel=128)
        res3 = Lib.residual_block(res2, n_in_channel=128, n_out_channel=128)
        res4 = Lib.residual_block(res3, n_in_channel=128, n_out_channel=128)
        res5 = Lib.residual_block(res4, n_in_channel=128, n_out_channel=128)
        deconv1 = de_conv_layer_beforeInstnormReluBugFix(res5, n_in_channel=128, n_out_channel=64, filter_size=3, stride=2)
        deconv2 = de_conv_layer_beforeInstnormReluBugFix(deconv1, n_in_channel=64, n_out_channel=32, filter_size=3, stride=2)
        convColor = Lib.conv_layer(deconv2, n_in_channel=32, n_out_channel=3, filter_size=9, stride=1, hasRelu=False) # if hasRelu, then x>0, then tanh(x) always>0
        if useNonHackScaling:
            tanh = tf.nn.tanh(convColor)
            scaled_01 = tanh / 2 + 0.5
        else:
            tanh = tf.nn.tanh(convColor) * 150 + 255./2  # TODO: why tanh * 150 + 255/2 is good?
            scaled_01 = tanh / 255.0
    return scaled_01
def de_conv_layer_beforeInstnormReluBugFix(input, n_in_channel, n_out_channel, filter_size, stride):
    filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_out_channel, n_in_channel], stddev=.1))
    in_shape = [s.value for s in input.get_shape()]
    out_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, n_out_channel]
    output = tf.nn.conv2d_transpose(input, filt, output_shape=out_shape, strides=[1,stride, stride, 1])
    print("deconv layer, output size: %s" % ([i.value for i in output.get_shape()]))
    return output
