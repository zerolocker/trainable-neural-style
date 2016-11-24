
import time,os, argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import custom_vgg19, Lib, InputPipeline
from InputPipeline import printdebug

BATCH_SIZE = 15
input_shape = [BATCH_SIZE, 256, 256, 3]
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1')
CONTENT_LAYER = 'conv4_2' # I can get good result with relu3_2 with slow neural-style with same weight. maybe I can try here
CONTENT_WEIGHT = 15
STYLE_WEIGHT = 100
NEW_H, NEW_W = 256, 256

parser = argparse.ArgumentParser()
parser.add_argument('--style', type=str, dest='style', help='style image path', required=True)
parser.add_argument('--train-path', type=str, dest='train_path', help='path to training images folder', required=True)
parser.add_argument('--epochs', type=int, dest='epochs', help='num epochs', default=2)
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='dir to save checkpoint in', default='chkpts/')
parser.add_argument('--model-prefix', type=str, dest='model_prefix', help='filename prefix of saved checkpoint', default='')
options = parser.parse_args()

paramStr = "%s_s%d_c%d" % (os.path.basename(options.style),int(STYLE_WEIGHT),int(CONTENT_WEIGHT))
logfile = open('out/'+paramStr+'.log','w+')
InputPipeline.logfile=logfile # let InputPipeline print some log, too
styleimg = Lib.load_image_as_batch_with_optional_resize(options.style)
img_train, NUM_EXAMPLES = InputPipeline.create_input_pipeline(batch_size=BATCH_SIZE, img_dir_path=options.train_path, NEW_H=NEW_H, NEW_W=NEW_W)


# Now we can go ahead and extract content features and style features
sess=tf.Session()
styleimg_ph = tf.placeholder(tf.float32, shape=styleimg.shape)
vgg19factory = custom_vgg19.Vgg19Factory()
vgg19_pretrain = vgg19factory.build(styleimg_ph)

# sanity check: make sure the layer names are correct
try:
    style_layers_pretrain = [getattr(vgg19_pretrain, name) for name in STYLE_LAYERS]
    content_layer_pretrain = getattr(vgg19_pretrain, CONTENT_LAYER)
except Exception as ex:
    print ex,  "incorrect layer name. Note: all layer named 'conv' is relu. e.g. 'conv1_1' is actually 'relu1_1'"
    sys.exit(1)

styleimg_grams = [Lib.gram_matrix(l) for l in style_layers_pretrain]
styleimg_grams_np = sess.run(styleimg_grams, feed_dict={styleimg_ph:styleimg})
# contentimg_feat_map_np = sess.run(content_layer_pretrain, feed_dict={styleimg_ph:contentimg}) # just for debug propose. It's not slow neural-style, so there is no target content img during training
styleimg_grams = [tf.constant(g, dtype=tf.float32) for g in styleimg_grams_np]


# In[8]:

# construct img transfrom network
img_pred = Lib.buildTransformNet(img_train, expected_shape=input_shape)


# In[9]:

# construct vgg19 to extract pred img's content & style
vgg19_pred = vgg19factory.build(img_pred)  # make sure pred img have VGG19's desired scale and range([0,1])
style_layers_pred = [getattr(vgg19_pred, name) for name in STYLE_LAYERS]
content_layer_pred = getattr(vgg19_pred, CONTENT_LAYER)

# construct vgg19 to extract train img's content as ground truth
vgg19_extractContent = vgg19factory.build(img_train)    # TODO ugly solution! 
   # So, in total I have to build 3 same vgg19 just because I have different input Tensor 
    # (two are placeholders of different shapes; the other one is the predicted image). Any way to avoid this?
content_layer_target = getattr(vgg19_extractContent, CONTENT_LAYER)


# In[10]:

style_losses = [Lib.compute_style_loss(styleimg_grams[i], style_layers_pred[i]) for i in xrange(len(styleimg_grams))]
content_loss = Lib.compute_content_loss(content_layer_target, content_layer_pred)
loss = STYLE_WEIGHT * reduce(tf.add, style_losses) + CONTENT_WEIGHT * content_loss # maybe add tv_loss as in slow neural-style ipynb later
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)



# ** ready to train **

saver=tf.train.Saver()
sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# some bookkeeping
MAX_ITER = options.epochs * NUM_EXAMPLES / BATCH_SIZE
duration = 0
chkpt_fname = 'chkpts/'+paramStr

printdebug('Training starts! NUM_EXAMPLES: %d BATCH_SIZE: %d' % (NUM_EXAMPLES,BATCH_SIZE), logfile)
for it in xrange(1, MAX_ITER+1):
    epoch = (it * BATCH_SIZE) / float(NUM_EXAMPLES)
    start_time = time.time()
    l = sess.run([train_op, loss]+ style_losses +[content_loss])
    duration += time.time() - start_time
    if it % 10 == 0:
        printdebug('epoch: {:.3f}, {:d}/{:d}, elapsed: {:.1f}s {:s}'.format(epoch, it, MAX_ITER, duration, str(l[1:])), logfile)
        duration = 0
    if it % 1000 == 0:
        saver.save(sess, '%s_%d.ckpt' % (chkpt_fname, it))

saver.save(sess, '%s.ckpt' % (chkpt_fname))


