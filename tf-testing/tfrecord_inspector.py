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
    features['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)
    for i in range(16):
        features['frames/{:04d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    # parse the features
    parsed_features = tf.parse_single_example(example, features)
    # print("parsed_features = %s" % parsed_features)

    with sess.as_default():
        # decode the encoded jpegs
        for i in range(16):
            # frame = tf.image.decode_jpeg(parsed_features['frames/{:04d}'.format(i)])
            frame = tf.decode_raw(parsed_features['frames/{:04d}'.format(i)], tf.uint8)
            frame = frame.eval()
            frame = frame.reshape((112, 112, 3))
            print("frame = %s" % (type(frame)))
            # save the frame
            cv2.imwrite("frame%d.jpg" % (i), frame)
