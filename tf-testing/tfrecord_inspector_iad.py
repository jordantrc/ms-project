# tfrecord_inspector.py

# inspects a tfrecord file, attempts to pull the data from it

import cv2
import os
import sys
import tensorflow as tf

file_name = sys.argv[1]
assert os.path.isfile(file_name)

sess = tf.Session()

for example in tf.python_io.tf_record_iterator(file_name):
    features = dict()
    features['example_id'] = tf.FixedLenFeature((), tf.string)
    features['label'] = tf.FixedLenFeature((), tf.int64)
    features['num_channels'] = tf.FixedLenFeature((), tf.int64)
    features['num_frames'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    with sess.as_default():
        example_id = parsed_features['example_id']
        label = parsed_features['label']
        num_channels = parsed_features['num_channels']
        num_frames = parsed_features['num_frames']

        print("PARSED FEATURES:")
        print("example_id = %s" % example_id)
        print("label = %s" % label)
        print("num_channels = %s" % num_channels)
        print("num_frames = %s" % num_frames)

        # decode the image data
        for i in range(1, 5):
            # decode the image, get label
            img = tf.decode_raw(parsed_features['img/{:02d}'.format(i)], tf.float32)
            print("Image = %s, shape = %s" % (str(img.eval()) img.shape))
