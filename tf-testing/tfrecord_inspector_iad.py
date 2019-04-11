# tfrecord_inspector.py

# inspects a tfrecord file, attempts to pull the data from it

import cv2
import os
import sys
import tensorflow as tf
import numpy as np

file_name = sys.argv[1]
assert os.path.isfile(file_name), "%s is not a file" % file_name

sess = tf.Session()

for example in tf.python_io.tf_record_iterator(file_name):
    features = dict()
    features = {        
        "example_id": tf.FixedLenFeature([], dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.int64),
        "network_depth": tf.FixedLenFeature([], dtype=tf.int64)
    }

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)
        features['num_rows/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['num_columns/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)

    parsed_features = tf.parse_single_example(example, features)

    with sess.as_default():
        example_id = parsed_features['example_id']
        label = parsed_features['label']
        network_depth = parsed_features['network_depth']

        print("PARSED FEATURES:")
        print("example_id = %s" % example_id.eval())
        print("label = %s" % label.eval())
        print("network_depth = %s" % network_depth.eval())

        # decode the image data
        for i in range(1, 6):
            # decode the image, get label
            img = tf.decode_raw(parsed_features['img/{:02d}'.format(i)], tf.float32)
            img_shape = tf.shape(img)
            nans_orig = tf.is_nan(img)
            img = tf.where(tf.is_nan(img), tf.zeros_like(img), img)
            img = tf.clip_by_value(img, 0.0, 1.0)
            nans_fixed = tf.is_nan(img)
            img_np = img.eval()
            #img_np = np.nan_to_num(img_np)
            print("Layer %s, img shape = %s\n\tmax = %s\n\tmin = %s\n\tmean = %s\n\tnans_count = [%s/%s]\n\timg = %s" % 
                (
                    i,
                    str(img_shape.eval()), 
                    img_np.max(), 
                    img_np.min(), 
                    img_np.mean(),
                    np.sum(nans_orig.eval()),
                    np.sum(nans_fixed.eval()),
                    img_np[0:10]
                ))
            num_rows = parsed_features['num_rows/{:02d}'.format(i)]
            num_columns = parsed_features['num_columns/{:02d}'.format(i)]
            print("num_columns = %s, num_rows = %s" % (num_columns.eval(), num_rows.eval()))
