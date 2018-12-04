# tfrecord_inspector.py

# inspects a tfrecord file, attempts to pull the data from it

import cv2
import os
import sys
import tensorflow as tf

file_name = sys.argv[1]
assert os.path.isfile(file_name)

sess = tf.Session()

LAYER_PAD = {'1': [[0, 0], [0, 0], [24, 24], [0, 0]]
             '2': [[0, 0], [0, 0], [56, 56], [0, 0]],
             '3': [[0, 0], [0, 0], [124, 124], [0, 0]],
             '4': [[0, 0], [0, 0], [254, 254], [0, 0]],
             '5': [[0, 0], [0, 0], [255, 255], [0, 0]]
             }

for example in tf.python_io.tf_record_iterator(file_name):
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    with sess.as_default():
        # decode the image data
        for i in range(1, 5):
            # decode the image, get label
            img = tf.decode_raw(parsed_features['img/{:02d}'.format(i)], tf.float32)
            img = tf.reshape(img, img_geom, "parse_reshape")

            # pad the image to make it square and then resize
            padding = tf.constant(LAYER_PAD[str(i)])
            img = tf.pad(img, padding, 'CONSTANT')
            print("img shape = %s" % img.get_shape())
            img = tf.image.resize_bilinear(img, (64, 64))
            print("img shape = %s" % img.get_shape())
            img = tf.squeeze(img, 0)
            cv2.imwrite("img%d.jpg" % (i), img)

