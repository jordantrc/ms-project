# tfrecord_inspector.py

# inspects a tfrecord file, attempts to pull the data from it

import cv2
import os
import sys
import tensorflow as tf

file_name = sys.argv[1]
assert os.path.isfile(file_name)

sess = tf.Session()

LAYER_PAD = {'1': [[0, 0], [0, 0], [24, 24], [0, 0]],
             '2': [[0, 0], [0, 0], [56, 56], [0, 0]],
             '3': [[0, 0], [0, 0], [124, 124], [0, 0]],
             '4': [[0, 0], [0, 0], [254, 254], [0, 0]],
             '5': [[0, 0], [0, 0], [255, 255], [0, 0]]
             }
LAYER_GEOMETRY = {'1': (64, 16, 1),
                  '2': (128, 16, 1),
                  '3': (256, 8, 1),
                  '4': (512, 4, 1),
                  '5': (512, 2, 1)
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
            img_geom = tuple([1]) + LAYER_GEOMETRY[str(i)]
            print("img_geom = %s" % img_geom)
            # decode the image, get label
            img = tf.decode_raw(parsed_features['img/{:02d}'.format(i)], tf.float32)
            img = tf.reshape(img, img_geom, "parse_reshape")

            # pad the image to make it square and then resize
            padding = tf.constant(LAYER_PAD[str(i)])
            img = tf.pad(img, padding, 'CONSTANT')
            img = tf.image.resize_bilinear(img, (64, 64))
            img = tf.squeeze(img, 0)
            img = img.eval()
            cv2.imwrite("img%d.jpg" % (i), img)

