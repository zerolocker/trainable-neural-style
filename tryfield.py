
# coding: utf-8

# In[1]:

import time

import tensorflow as tf
import numpy as np

import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import custom_vgg19


# In[2]:

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


# In[3]:

class ARG:pass
arg = ARG()
arg.gen_img_height=500
arg.styl

styleimg = load_image_as_batch_with_optional_resize('./picasso_selfport1907.jpg')
print(styleimg.shape)
contentimg = load_image_as_batch_with_optional_resize('./brad_pitt.jpg', newH=arg.gen_img_height)
print(contentimg.shape)

arg.gen_img_width = contentimg.shape[1] # computed from aspect ratio of content_img
# show image
# skimage.io.imshow(contentimg[0])
# plt.show()


# In[4]:

sess=tf.Session()
img_pl = tf.placeholder(tf.float32)
vgg19factory = custom_vgg19.Vgg19Factory()
vgg19 = vgg19factory.build(img_pl)

print(styleimg.shape)
conv31feat = sess.run(vgg19.conv3_1, feed_dict={img_pl:styleimg})
print(conv31feat.shape)


# In[5]:

def gram_matrix(feat_map):
    assert isinstance(feat_map, tf.Tensor)
    shape = tf.cast(tf.shape(feat_map), tf.float32)
    _, h, w, ch = shape[0], shape[1], shape[2], shape[3]
    F = tf.reshape(feat_map, [-1, tf.cast(ch,tf.int32)])
    
    # TODO: if m<n, compute feat_map*feat_map, else compute feat_map'*feat_map 
    gram = tf.matmul(F, F, transpose_a=True) / h / w / ch # not sure why  we have"/ ch". if not, the style_loss is too big
    return gram

def compute_style_loss(gram_of_feat_map1, feat_map2):
    shape = tf.cast(tf.shape(feat_map2), tf.float32)
    _, h, w, ch = shape[0], shape[1], shape[2], shape[3]
    
    G1, G2 = gram_of_feat_map1, gram_matrix(feat_map2)
    style_loss = tf.nn.l2_loss((G1-G2))/ (ch**2) # ch^2 is #element in G1, G2
    return style_loss

def compute_content_loss(feat_map1, feat_map2):
    shape = tf.cast(tf.shape(feat_map1), tf.float32)
    _, h, w, ch = shape[0], shape[1], shape[2], shape[3]
    
    content_loss = tf.nn.l2_loss(feat_map1-feat_map2)/h/w/ch
    return content_loss


# In[51]:

img_gen = tf.Variable(tf.truncated_normal(contentimg.shape,  mean=0.5, stddev=0.1))
sess.run(img_gen.initializer)

vgg19_for_img_gen = vgg19factory.build(img_gen)


# In[52]:

# precompute style image's stuffs
contentimg_feat_map = tf.Variable(vgg19.conv2_2,validate_shape=False, trainable=False)
styleimg_gram = tf.Variable(gram_matrix(vgg19.conv3_1), validate_shape=False, trainable=False)
sess.run(contentimg_feat_map.initializer, feed_dict={img_pl:contentimg})
sess.run(styleimg_gram.initializer, feed_dict={img_pl:styleimg})

style_loss = compute_style_loss(styleimg_gram, vgg19_for_img_gen.conv3_1)
[style_loss_np] =sess.run([style_loss])
print('initial style loss = %f' % style_loss_np)

content_loss = compute_content_loss(contentimg_feat_map, vgg19_for_img_gen.conv2_2)
[content_loss_np] =sess.run([content_loss])
print('initial content loss = %f' % content_loss_np)

total_loss =  1000*style_loss + content_loss
[total_loss_np] = sess.run([total_loss])
print('initial total loss = %f' % total_loss_np)


# In[53]:

temp = set(tf.all_variables())
train_op = tf.train.AdamOptimizer(0.02).minimize(total_loss)
#I honestly don't know how else to initialize ADAM in TensorFlow.
sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

# start optimization
iter = 0
MAX_ITER = 200

while iter < MAX_ITER:
    sess.run(train_op)
    iter += 1
print(sess.run([style_loss, content_loss, total_loss]))


# In[54]:

img_gen_clipped = tf.clip_by_value(img_gen, 0,1) #  the range of values in generated image will fall out of [0,1].
            # If you scale it to [0,1] instead of clipping it to [0,1], the image will look "grey"
img_gen_np = np.squeeze(sess.run(img_gen_clipped), 0)
skimage.io.imshow(img_gen_np)
plt.show()


# In[28]:




# In[ ]:



