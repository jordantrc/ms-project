# train_network.py

import tensorflow as tf
import numpy as np

NUM_EPOCHS = 1
TRAIN_FILE = "/home/jordanc/datasets/UCF-101/tfrecords/train.tfrecords"

with tf.Sessions() as sess:

    # setup video features
    feature = {}
    feature['label'] = tf.FixedLenFeature([], tf.string)
    feature['img_raw'] = tf.FixedLenFeature([], tf.string)
    feature['height'] = tf.FixedLenFeature([], tf.int64)
    feature['width'] = tf.FixedLenFeature([], tf.int64)

    filename_queue = tf.train.string_input_producer([TRAIN_FILE], num_epochs=NUM_EPOCHS)

    # define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # decode the record
    features = tf.parse_single_example(serialized_example, features=feature)
    
