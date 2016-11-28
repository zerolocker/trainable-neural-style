# ref: http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
# ref: https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#batching
# ref: https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

import numpy as np
import tensorflow as tf
from IPython import embed
import os, re

logfile = None

def create_input_pipeline(img_dir_path, batch_size, NEW_H, NEW_W): 
    tmp = os.listdir(img_dir_path)
    img_filenames = filter(lambda s: s.endswith('jpg'), tmp)
    img_filenames = [img_dir_path+'/'+name for name in img_filenames]
    assert len(tmp) == len(img_filenames), "Sanity check: should be a directory only containing .jpg files, " + \
        "because the image file reader assumes this"

    C_S_G_tuple = {} # content img, style img, generated img
    for f in img_filenames:
        matchobj = re.match('(\w+)_(\w)', os.path.basename(f))
        if matchobj is None:
            printdebug('Invalid filename: '+f, logfile)
        else:
            imgid, imgtype = matchobj.group(1), matchobj.group(2)
            if imgid not in C_S_G_tuple:
                C_S_G_tuple[imgid] = {}
            C_S_G_tuple[imgid][imgtype] = f

    C_list, S_list, G_list = [],[],[]
    for imgtuple_dict in C_S_G_tuple.values():
        if len(imgtuple_dict) == 3:  # all C,S,G images are present
            C_list.append(imgtuple_dict['C'])
            S_list.append(imgtuple_dict['S'])
            G_list.append(imgtuple_dict['G'])

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([C_list,S_list,G_list], num_epochs=None, shuffle=True)

    # read images from disk
    C_fname, S_fname, G_fname = input_queue[0], input_queue[1], input_queue[2]
    C_rgb = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(C_fname), channels=3), tf.float32)
    S_rgb = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(S_fname), channels=3), tf.float32)
    G_rgb = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(G_fname), channels=3), tf.float32)

    # processiong:
    assert NEW_H == NEW_W, 'this processing step assumes NEW_H == NEW_W to work correctly (not distort img)'
    # resize C and G to the same size with shorter edge == NEW_H
    shC = tf.shape(C_rgb)
    h,w = shC[0], shC[1]
    newh = tf.cond(h<w, lambda:tf.constant(NEW_H), lambda:NEW_H*h/w)
    neww = tf.cond(h<w, lambda:NEW_H*w/h, lambda:tf.constant(NEW_H))
    C_rgb = tf.image.resize_images(C_rgb, newh, neww)
    G_rgb = tf.image.resize_images(G_rgb, newh, neww)
    # crop a NEW_H*NEW_W area from both C and G to make sure output has NEW_H*NEW_W
    C_rgb = C_rgb[0:NEW_H, 0:NEW_W, :]
    G_rgb = G_rgb[0:NEW_H, 0:NEW_W, :]
    # resize S (style img) using the same method to make sure output has NEW_H*NEW_W
    shS = tf.shape(S_rgb)
    h,w = shS[0], shS[1]
    newh = tf.cond(h<w, lambda:tf.constant(NEW_H), lambda:NEW_H*h/w)
    neww = tf.cond(h<w, lambda:NEW_H*w/h, lambda:tf.constant(NEW_H))
    S_rgb = tf.image.resize_images(S_rgb, newh, neww)
    S_rgb = S_rgb[0:NEW_H, 0:NEW_W, :]

    printdebug('image scale of InputPipeline will be [0,1], could try [0,255] and substract mean later.',logfile)

    # Optional Preprocessing or Data Augmentation
    # if you have time, look at https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#preprocessing

    # Batching (input tensors backed by a queue; and then combine inputs into a batch)
    C_batch, S_batch, G_batch = tf.train.batch([C_rgb, S_rgb, G_rgb], batch_size=batch_size, capacity=3*batch_size)
    return C_batch, S_batch, G_batch, len(C_list)

def printdebug(str, logfile=None):
    print('  ----   DEBUG: '+str)
    if logfile is not None:
        logfile.write('  ----   DEBUG: '+str+'\n')
        logfile.flush()

if __name__ == "__main__":
    sess = tf.Session()
    
    batch_size=100
    
    # create input pipelines for training set and validation set
    train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, batch_size, num_epochs=None, produceVGGInput=False)
    val_image_batch, val_label_batch, VAL_SIZE = create_input_pipeline(LABELS_FILE_VAL, batch_size, num_epochs=None, produceVGGInput=False)
    printdebug("TRAIN_SIZE: %d VAL_SIZE: %d BATCH_SIZE: %d " % (TRAIN_SIZE, VAL_SIZE, batch_size))

    # Required. 
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    print ("  ----   Running unit tests for training set -----")
    imgs, labels = sess.run([train_image_batch, train_label_batch])
    assert(imgs.shape[0] == batch_size)
    assert(imgs[0].reshape(-1)[0] != imgs[1].reshape(-1)[0])
    print(imgs[0])
    print(labels[0])
    imgs2, labels2 = sess.run([train_image_batch, train_label_batch])
    assert(imgs2[0].reshape(-1)[0] != imgs[0].reshape(-1)[0]) # test if the batches change between two calls to sess.run()

    print ("  ----   Running unit tests for validation set -----")
    for i in range(VAL_SIZE / batch_size): 
        imgs, labels = sess.run([val_image_batch, val_label_batch])
    print ("  ----   All tests passed -----")


