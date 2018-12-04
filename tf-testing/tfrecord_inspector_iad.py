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
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    with sess.as_default():
        # decode the image data
        for i in range(16):
            # decode the image, get label
        img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
        img = tf.reshape(img, img_geom, "parse_reshape")

        # pad the image to make it square and then resize
        padding = tf.constant(LAYER_PAD)
        img = tf.pad(img, padding, 'CONSTANT')
        print("img shape = %s" % img.get_shape())
        img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        print("img shape = %s" % img.get_shape())
        img = tf.squeeze(img, 0)
        cv2.imwrite("img%d.jpg" % (i), img)
        
