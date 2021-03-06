import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def load_image_as_batch_with_optional_resize(path, newH=None, newW=None):
    img = skimage.io.imread(path)
    img = img / 255.0
    
    oldH, oldW = img.shape[0], img.shape[1] # assume h,w is shape[0] and [1] respectively
    if newH != None or newW != None:
        if newW is None:
            newW = int(oldW * float(newH) / oldH)
        elif newH is None:
            newH = int(oldH * float(newW) / oldW)
        img = skimage.transform.resize(img, (newH, newW))
    
    # delete the Alpha channel if the image is RGBA to make sure # channel is correct
    if img.shape[2]==4:
        img = img[:,:,0:3]
        
    # add another dimension to make it a batch , bacause our vgg19 def takes a batch
    img = img.reshape((1,)+img.shape)
    return img

def buildStyconNet(content, style, useNonHackScaling=False):
    # currently deconv layer needs an already known shape 
    for d in content.get_shape().as_list(): assert d is not None 
    for d in style.get_shape().as_list(): assert d is not None 

    content = tf.pad(content, [[0,0],[40,40],[40,40],[0,0]], mode='REFLECT')
    style = tf.pad(style, [[0,0],[40,40],[40,40],[0,0]], mode='REFLECT')


    with tf.variable_scope("styconNet"):
        with tf.variable_scope("con"):       
            conv1 = conv_layer(content, n_in_channel=3, n_out_channel=32, filter_size=9, stride=1, hasRelu=True)
            conv2 = conv_layer(conv1, n_in_channel=32, n_out_channel=64, filter_size=3, stride=2, hasRelu=True)
            conv3 = conv_layer(conv2, n_in_channel=64, n_out_channel=128, filter_size=3, stride=2, hasRelu=True)
            res1 = residual_block(conv3, n_in_channel=128, n_out_channel=128)
            res2_con = residual_block(res1, n_in_channel=128, n_out_channel=128)
        with tf.variable_scope("sty"):
                conv1 = conv_layer(style, n_in_channel=3, n_out_channel=32, filter_size=9, stride=1, hasRelu=True)
                conv2 = conv_layer(conv1, n_in_channel=32, n_out_channel=64, filter_size=3, stride=2, hasRelu=True)
                conv3 = conv_layer(conv2, n_in_channel=64, n_out_channel=128, filter_size=3, stride=2, hasRelu=True)
                res1 = residual_block(conv3, n_in_channel=128, n_out_channel=128)
                res2_sty = residual_block(res1, n_in_channel=128, n_out_channel=128)
        with tf.variable_scope("decoder"):
            combined = tf.mul(res2_con, res2_sty)
            res3 = residual_block(combined, n_in_channel=128, n_out_channel=128)
            res4 = residual_block(res3, n_in_channel=128, n_out_channel=128)
            res5 = residual_block(res4, n_in_channel=128, n_out_channel=128)
            deconv1 = de_conv_layer(res5, n_in_channel=128, n_out_channel=64, filter_size=3, stride=2)
            deconv2 = de_conv_layer(deconv1, n_in_channel=64, n_out_channel=32, filter_size=3, stride=2)
            convColor = conv_layer(deconv2, n_in_channel=32, n_out_channel=3, filter_size=9, stride=1, hasRelu=False) # if hasRelu, then x>0, then tanh(x) always>0
            if useNonHackScaling:
                tanh = tf.nn.tanh(convColor)
                scaled_01 = tanh / 2 + 0.5
            else:
                tanh = tf.nn.tanh(convColor) * 150 + 255./2  # TODO: why tanh * 150 + 255/2 is good?
                scaled_01 = tanh / 255.0
    return scaled_01



def buildTransformNet(img, expected_shape, useNonHackScaling=False):
    assert expected_shape == [d.value for d in img.get_shape()]   # now we fix the image size and assume 
      # img has an already-defined shape. Otherwise Tensorflow will fail to infer the output shape of the layer
        # which is very inconvinent for debugging
    
    img = tf.pad(img, [[0,0],[40,40],[40,40],[0,0]], mode='REFLECT') # the same as the paper's supp material PDF.
            # I don't know why fast-style-transfer repo doesn't do the same, which is:
            #    (1) have this padding step
            #    (2) not use padding in the residual blocks' conv layers
    with tf.variable_scope("transNet"):
        conv1 = conv_layer(img, n_in_channel=3, n_out_channel=32, filter_size=9, stride=1, hasRelu=True)
        conv2 = conv_layer(conv1, n_in_channel=32, n_out_channel=64, filter_size=3, stride=2, hasRelu=True)
        conv3 = conv_layer(conv2, n_in_channel=64, n_out_channel=128, filter_size=3, stride=2, hasRelu=True)
        res1 = residual_block(conv3, n_in_channel=128, n_out_channel=128)
        res2 = residual_block(res1, n_in_channel=128, n_out_channel=128)
        res3 = residual_block(res2, n_in_channel=128, n_out_channel=128)
        res4 = residual_block(res3, n_in_channel=128, n_out_channel=128)
        res5 = residual_block(res4, n_in_channel=128, n_out_channel=128)
        deconv1 = de_conv_layer(res5, n_in_channel=128, n_out_channel=64, filter_size=3, stride=2)
        deconv2 = de_conv_layer(deconv1, n_in_channel=64, n_out_channel=32, filter_size=3, stride=2)
        convColor = conv_layer(deconv2, n_in_channel=32, n_out_channel=3, filter_size=9, stride=1, hasRelu=False) # if hasRelu, then x>0, then tanh(x) always>0
        if useNonHackScaling:
            tanh = tf.nn.tanh(convColor)
            scaled_01 = tanh / 2 + 0.5
        else:
            tanh = tf.nn.tanh(convColor) * 150 + 255./2  # TODO: why tanh * 150 + 255/2 is good?
            scaled_01 = tanh / 255.0
    return scaled_01
    
def conv_layer(input, n_in_channel, n_out_channel, filter_size, stride, hasRelu=True, padding='SAME'):
    # TODO conv layer without adding bias ( bias is not used in paper either). but I could try it later if time permitted. tf.nn.bias_add
    filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_in_channel, n_out_channel], stddev=.1))
    output = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding=padding)
    output = _instance_norm(output) # TODO read what is instance normalization 
    if hasRelu:
        output = tf.nn.relu(output)
    print("conv layer, output size: %s" % ([i.value for i in output.get_shape()]))
    return output

def de_conv_layer(input, n_in_channel, n_out_channel, filter_size, stride):
    filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_out_channel, n_in_channel], stddev=.1))
    in_shape = [s.value for s in input.get_shape()]
    out_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, n_out_channel]
    output = tf.nn.conv2d_transpose(input, filt, output_shape=out_shape, strides=[1,stride, stride, 1])
    output = _instance_norm(output)
    output = tf.nn.relu(output)
    print("deconv layer, output size: %s" % ([i.value for i in output.get_shape()]))
    return output

def residual_block(input, n_in_channel, n_out_channel, name="n/a"):
    print("START residual_block ")
    output = conv_layer(input, n_in_channel, n_out_channel, filter_size=3, stride=1, hasRelu=True, padding='VALID')
    output = conv_layer(output,n_out_channel, n_out_channel, filter_size=3, stride=1, hasRelu=False, padding='VALID')
    bsize, h, w, ch = [i.value for i in input.get_shape()]
    output = input[:,2:h-2, 2:w-2, :] + output
    print("END residual_block")
    return output
    

# TODO Read what is instance normalization. The following code is copied, I don't know how it works
def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape), name="instnorm_shift")
    scale = tf.Variable(tf.ones(var_shape), name="instnorm_scale")
    epsilon = 1e-3
    normalized = (net-mu)/tf.sqrt(sigma_sq + epsilon)
    return scale * normalized + shift

def gram_matrix(feat_map_batch):
    # We don't need to handle unknown shape here. Otherwise: we can use tf.cast(tf.shape(feat_map_batch), tf.int32)
    bsize, h, w, ch  = feat_map_batch.get_shape().as_list()
    F = tf.reshape(feat_map_batch, [bsize, -1, ch])

    # TODO if m<n, compute feat_map*feat_map, else compute feat_map'*feat_map 
    gram = tf.batch_matmul(F, F, adj_x=True)/ (h * w * ch) # not sure why  we have "/ch". if not, the style_loss is too big
    return gram

def compute_style_loss(gram_target, feat_map_batch):
    bsize, h, w, ch  = feat_map_batch.get_shape().as_list()
    # TODO : test broadcasting and shape is correct for all tensors
    G1, G2_batch = gram_target, gram_matrix(feat_map_batch)
    style_loss =  tf.nn.l2_loss(G1-G2_batch) / bsize / (ch**2) # ch^2 is #element in G1
    return style_loss

def compute_content_loss(feat_map_target, feat_map_batch):
    bsize, h, w, ch  = feat_map_batch.get_shape().as_list()  # TODO I think we could just use tf.reduce_mean() to avoid getting the shape
    content_loss = tf.nn.l2_loss(feat_map_target-feat_map_batch)/(bsize*h*w*ch)
    return content_loss

def compute_tv_loss(img_gen):
    bsize, h , w, ch = img_gen.get_shape().as_list()
    y_tv = tf.nn.l2_loss(img_gen[:,1:,:,:] - img_gen[:,:h-1,:,:])
    x_tv = tf.nn.l2_loss(img_gen[:,:,1:,:] - img_gen[:,:,:w-1,:])
    tv_loss = (x_tv + y_tv)/(bsize*h*w*ch)
    return tv_loss
