# ref: http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
# ref: https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#batching
# ref: https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

import numpy as np
import tensorflow as tf
from IPython import embed
import os

logfile = None

def create_input_pipeline(img_dir_path, batch_size, NEW_H, NEW_W): 
    tmp = os.listdir(img_dir_path)
    img_filenames = filter(lambda s: s.endswith('jpg'), tmp)
    img_filenames = [img_dir_path+'/'+name for name in img_filenames]
    assert len(tmp) == len(img_filenames), "Sanity check: should be a directory only containing .jpg files, " + \
        "because the image file reader assumes this"


    # Makes an input queue
    input_queue = tf.train.string_input_producer(img_filenames, num_epochs=None, shuffle=True)

    # read images from disk
    file_contents = tf.read_file(input_queue.dequeue())
    one_image = tf.image.decode_jpeg(file_contents, channels=3)
    one_image = tf.image.convert_image_dtype(one_image, tf.float32)
    printdebug('image scale of InputPipeline will be [0,1], could try [0,255] and substract mean later.',logfile)
    one_image = tf.image.resize_images(one_image, NEW_H, NEW_W)

    # Optional Preprocessing or Data Augmentation
    # if you have time, look at https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#preprocessing

    # Batching (input tensors backed by a queue; and then combine inputs into a batch)
    image_batch = tf.train.batch([one_image], batch_size=batch_size, capacity=3*batch_size)
    printdebug("Inferred image_batch shape:(check if it is fully specified!)" + str(image_batch.get_shape().as_list()),logfile)
    return image_batch, len(img_filenames)

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


