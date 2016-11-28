
import time,os, argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import custom_vgg19, Lib, OstagramTrainInputPipeline, OtherNetDefs
from OstagramTrainInputPipeline import printdebug

BATCH_SIZE = 10
input_shape = [BATCH_SIZE, 256, 256, 3]
NEW_H, NEW_W = 256, 256

parser = argparse.ArgumentParser()
parser.add_argument('--train-path', type=str, dest='train_path', help='path to training images folder', required=True)
parser.add_argument('--epochs', type=int, dest='epochs', help='num epochs', default=2)
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='dir to save checkpoint in', default='chkpts/')
parser.add_argument('--model-prefix', type=str, dest='model_prefix', help='filename prefix of saved checkpoint', default='')
parser.add_argument('--styconNet-type', type=str, dest='styconNet_type', help='either"concat" or "product" or "multiproduct"', default='product')
parser.add_argument('--restore-chkpt', type=str, dest='restore_chkpt', help='path to checkpoint to restore training"', default='')

options = parser.parse_args()

paramStr = "%s_%s" % ('CSGTuples', options.model_prefix)
logfile = open('out/'+paramStr+'.log','w+')
OstagramTrainInputPipeline.logfile=logfile # let InputPipeline print some log, too
C_batch, S_batch, G_batch, NUM_EXAMPLES = OstagramTrainInputPipeline.create_input_pipeline(options.train_path, BATCH_SIZE, NEW_H, NEW_W)

# construct img transfrom network
sess=tf.Session()
if options.styconNet_type == 'product':
    printdebug("StyconNet type: product", logfile)
    img_pred = Lib.buildStyconNet(C_batch, S_batch)
elif options.styconNet_type == 'concat':
    printdebug("StyconNet type: concat", logfile)
    img_pred = OtherNetDefs.buildStyconNetConcat(C_batch, S_batch)
elif options.styconNet_type == 'multiproduct':
    printdebug("StyconNet type: multiproduct", logfile)
    img_pred = OtherNetDefs.buildStyconNetMultiProduct(C_batch, S_batch)

loss = tf.nn.l2_loss(img_pred - G_batch) / BATCH_SIZE
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# ** ready to train **

saver=tf.train.Saver()
sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
if options.restore_chkpt != '':
    printdebug('Resuming checkpoint: ' + options.restore_chkpt)
    saver.restore(sess, options.restore_chkpt)

# some bookkeeping
MAX_ITER = options.epochs * NUM_EXAMPLES / BATCH_SIZE
duration = 0
chkpt_fname = options.checkpoint_dir+'/'+paramStr

printdebug('Training starts! NUM_EXAMPLES: %d BATCH_SIZE: %d' % (NUM_EXAMPLES,BATCH_SIZE), logfile)
for it in xrange(1, MAX_ITER+1):
    epoch = (it * BATCH_SIZE) / float(NUM_EXAMPLES)

    start_time = time.time()
    l = sess.run([train_op, loss])
    duration += time.time() - start_time
    if it % 10 == 0:
        printdebug('epoch: {:.3f}, {:d}/{:d}, elapsed: {:.1f}s {:s}'.format(epoch, it, MAX_ITER, duration, str(l[1:])), logfile)
        duration = 0
    if it % 1000 == 0:
        saver.save(sess, '%s_%d.ckpt' % (chkpt_fname, it))

saver.save(sess, '%s.ckpt' % (chkpt_fname))


