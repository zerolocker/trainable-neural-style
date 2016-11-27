
import time,os, argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import custom_vgg19, Lib, InputPipeline, OtherNetDefs
from InputPipeline import printdebug

BATCH_SIZE = 10
input_shape = [BATCH_SIZE, 256, 256, 3]
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1')
CONTENT_LAYER = 'conv4_2' # I can get good result with relu3_2 with slow neural-style with same weight. maybe I can try here
CONTENT_WEIGHT = 7.5
STYLE_WEIGHT = 100
NEW_H, NEW_W = 256, 256
CHKPTS_DIR = 'chkpts/hasArtifact'


# Read all style images used in training
styleimg_fnames = os.listdir('styles/')
styleimgs = []
for fname in styleimg_fnames:
    styleimgs.append(Lib.load_image_as_batch_with_optional_resize('styles/'+fname, NEW_H, NEW_W))


parser = argparse.ArgumentParser()
parser.add_argument('--train-path', type=str, dest='train_path', help='path to training images folder', required=True)
parser.add_argument('--epochs', type=int, dest='epochs', help='num epochs', default=2)
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='dir to save checkpoint in', default='chkpts/')
parser.add_argument('--model-prefix', type=str, dest='model_prefix', help='filename prefix of saved checkpoint', default='')
parser.add_argument('--styconNet-type', type=str, dest='styconNet_type', help='either"concat" or "product" or "multiproduct"', default='product')
options = parser.parse_args()

paramStr = "%s_%s_s%d_c%d" % ('%dstyles' % len(styleimgs),options.model_prefix, int(STYLE_WEIGHT), int(CONTENT_WEIGHT))
logfile = open('out/'+paramStr+'.log','w+')
InputPipeline.logfile=logfile # let InputPipeline print some log, too
img_train, NUM_EXAMPLES = InputPipeline.create_input_pipeline(batch_size=BATCH_SIZE, img_dir_path=options.train_path, NEW_H=NEW_H, NEW_W=NEW_W)

""" This training method is not yet implemented.
def load_and_build_transNet_models(sess, chkpt_file_list):
    return_pred = []
    for fname in chkpt_file_list:
        scopename = os.path.basename(fname)
        with tf.variable_scope(scopename):
            pred = Lib.buildTransformNet(content_ph,expected_shape=input_shape)
            return_pred.append(pred)
            transNetVars = tf.get_collection(tf.GraphKeys.VARIABLES, scope = scopename)
            chkvarname2var = {v.name.replace(scopename+'/','').replace(':0',''): v for v in transNetVars}
            # if it fails to load, it's probably because the names don't match. then use this file to inspect the checkpoint: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
            saver = tf.train.Saver(chkvarname2var)
            saver.restore(sess, fname)
    return return_pred

def extract_styimgname_from_chkpt_name(chkpt_name):
    return re.match('.*\.jpg|.*\.jpeg|.*\.png', chkpt_name).group(0)

chkpt_file_list = [CHKPTS_DIR+'/'+f  for f in os.listdir(CHKPTS_DIR)]
"""

# In[9]:

# construct img transfrom network
sess=tf.Session()
styleimg_ph = tf.placeholder(tf.float32, shape=[1,]+input_shape[1:])
if options.styconNet_type == 'product':
    printdebug("StyconNet type: product", logfile)
    img_pred = Lib.buildStyconNet(img_train, styleimg_ph)
elif options.styconNet_type == 'concat':
    printdebug("StyconNet type: concat", logfile)
    img_pred = OtherNetDefs.buildStyconNetConcat(img_train, styleimg_ph)
elif options.styconNet_type == 'multiproduct':
    printdebug("StyconNet type: multiproduct", logfile)
    img_pred = OtherNetDefs.buildStyconNetMultiProduct(img_train, styleimg_ph)


# construct vgg19 to extract pred img's content & style
vgg19factory = custom_vgg19.Vgg19Factory()
vgg19_pred = vgg19factory.build(img_pred)  # make sure pred img have VGG19's desired scale and range([0,1])
# sanity check: make sure the layer names are correct
try:
    style_layers_pred = [getattr(vgg19_pred, name) for name in STYLE_LAYERS]
    content_layer_pred = getattr(vgg19_pred, CONTENT_LAYER)
except Exception as ex:
    print ex,  "incorrect layer name. Note: all layer named 'conv' is relu. e.g. 'conv1_1' is actually 'relu1_1'"
    sys.exit(1)

# construct vgg19 to extract train style img's style as ground truth
vgg19_extractStyle = vgg19factory.build(styleimg_ph)
style_layers_target = [getattr(vgg19_extractStyle, name) for name in STYLE_LAYERS]
styleimg_grams = [Lib.gram_matrix(l) for l in style_layers_target]

# construct vgg19 to extract train content img's content as ground truth
vgg19_extractContent = vgg19factory.build(img_train)    # TODO ugly solution! 
   # So, in total I have to build 3 same vgg19 just because I have different input Tensor 
    # (two are placeholders of different shapes; the other one is the predicted image). Any way to avoid this?
content_layer_target = getattr(vgg19_extractContent, CONTENT_LAYER)


# In[10]:

style_losses = [Lib.compute_style_loss(styleimg_grams[i], style_layers_pred[i]) for i in xrange(len(styleimg_grams))]
content_loss = Lib.compute_content_loss(content_layer_target, content_layer_pred)
# tv_loss = Lib.compute_tv_loss(img_pred) # seems not helping(may need exp to confirm) 
loss = STYLE_WEIGHT * reduce(tf.add, style_losses) + CONTENT_WEIGHT * content_loss 
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
chkpt_fname = options.checkpoint_dir+'/'+paramStr

printdebug('Training starts! NUM_EXAMPLES: %d BATCH_SIZE: %d' % (NUM_EXAMPLES,BATCH_SIZE), logfile)
for it in xrange(1, MAX_ITER+1):
    epoch = (it * BATCH_SIZE) / float(NUM_EXAMPLES)

    styleimg = styleimgs[np.random.choice(len(styleimgs))]

    start_time = time.time()
    l = sess.run([train_op, loss]+ style_losses +[content_loss], feed_dict={styleimg_ph: styleimg})
    duration += time.time() - start_time
    if it % 10 == 0:
        printdebug('epoch: {:.3f}, {:d}/{:d}, elapsed: {:.1f}s {:s}'.format(epoch, it, MAX_ITER, duration, str(l[1:])), logfile)
        duration = 0
    if it % 1000 == 0:
        saver.save(sess, '%s_%d.ckpt' % (chkpt_fname, it))

saver.save(sess, '%s.ckpt' % (chkpt_fname))


