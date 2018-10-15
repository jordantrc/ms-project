# train_network.py

import tensorflow as tf
import numpy as np

NUM_EPOCHS = 1
NUM_CLASSES = 101
TRAIN_FILE = "/home/jordanc/datasets/UCF-101/tfrecords/train.tfrecords"


with tf.Session() as sess:

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
    print("features = %s" % features)
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.string)
    width = tf.cast(features['width'], tf.uint8)
    height = tf.cast(features['height'], tf.uint8)
    num_features = 3 * width * height

    # init variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    x = tf.placeholder(tf.uint8, shape=[None, num_features], name='x')
    y_pred = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_pred')
    y_pred_class = tf.argmax(y_pred axis=1)
