# iad_input.py

# Provides IAD input data and labels for IAD training.

import numpy as np
import os
import random
import tensorflow as tf


class IADInput():

    def __init__(self, train_file, test_file, iad_dimensions, num_classes, layer, train_epochs):
        self.train_file = train_file
        self.test_file = test_file
        self.iad_dimensions = iad_dimensions
        self.num_classes = num_classes
        self.layer = layer
        self.train_epochs = train_epochs

        # build the sample list
        self.train_samples = self.create_samples_list(self.train_file)
        self.test_samples = self.create_samples_list(self.test_file)

        self.train_counter = 0

        # training samples are randomized
        random.shuffle(self.train_samples)

    def create_samples_list(self, file_name):
        '''obtaines the sample names and classes for each sample file'''
        assert os.path.isfile(file_name), "file %s does not exist or is not a file" % file_name

        # sample data structure:
        # samples = [
        #   ['file', label]
        #   ]
        samples = []

        with open(file_name, 'r') as fd:
            raw_text = fd.read()
            for l in raw_text.splitlines():
                if len(l) > 0:
                    sample_file, label = l.split()
                    samples.append([sample_file, int(label)])

        return samples

    def get_train_sample(self, batch_size):
        '''gets a training sample'''
        if self.train_counter > self.train_epochs * len(self.train_samples):
            # exit early if we've exceeded the total number of training samples
            return None, None

        random_offset = random.randint(0, len(self.train_samples) - batch_size)
        samples = self.train_samples[random_offset:random_offset + batch_size]

        # get the features and labels from the samples
        x, y, _ = self.get_features_labels(samples, offset=-1)

        self.train_counter += batch_size

        return x, y

    def get_test_sample(self, batch_size, offset, iad_offset):
        '''gets a testing sample'''
        samples = self.test_samples[offset:offset + batch_size]

        x, y, new_offset = self.get_features_labels(samples, offset=iad_offset)

        return x, y, new_offset

    def get_test_sample_random(self, batch_size):
        '''gets a testing sample'''
        offset = random.randint(0, self.test_samples)
        samples = self.test_samples[offset:offset + batch_size]

        x, y, _ = self.get_features_labels(samples, offset=-2)

        return x, y

    def get_features_labels(self, sample_list, offset):
        '''get features and labels from the samples'''
        
        all_images = []
        all_labels = []

        for sample in sample_list:
            file_name = sample[0]
            for example in tf.python_io.tf_record_iterator(file_name):
                img_geom = self.iad_dimensions[str(self.layer)]
                features = dict()
                features['label'] = tf.FixedLenFeature((), tf.int64)

                features['img/{:02d}'.format(self.layer)] = tf.FixedLenFeature((), tf.string)
                features['num_rows/{:02d}'.format(self.layer)] = tf.FixedLenFeature((), tf.int64)
                features['num_columns/{:02d}'.format(self.layer)] = tf.FixedLenFeature((), tf.int64)

                parsed_features = tf.parse_single_example(example, features)
                num_rows = parsed_features['num_rows/{:02d}'.format(self.layer)]
                num_columns = parsed_features['num_columns/{:02d}'.format(self.layer)]

                # decode the image, get label
                img = tf.decode_raw(parsed_features['img/{:02d}'.format(self.layer)], tf.float32)
                img = tf.where(tf.is_nan(img), tf.zeros_like(img), img)
                img = tf.clip_by_value(img, 0.0, 1.0)
                #img = tf.subtract(img, 0.5)

                img = tf.reshape(img, (num_rows, num_columns, 1), "parse_reshape_test")
                print("img shape = %s" % img.get_shape())
                
                # random slice of the image
                #img = tf.random_crop(img, [img_geom[0], img_geom[1], 1])
                #column_offsets = list(range(num_columns))[::img_geom[1]]
                column_offsets = tf.range(0, num_columns - img_geom[1], delta=img_geom[1])

                # determine the offset for the IAD slice
                if offset == -1:
                    # select a random IAD slice
                    start_column = tf.cast(tf.random_shuffle(column_offsets)[0], dtype=tf.int32)
                    new_offset = -1
                elif offset == -2:
                    start_column = 0
                    new_offset = -2
                else:
                    start_column = offset
                    new_offset = offset + img_geom[1]
                    if new_offset > img_geom[1]:
                        new_offset = 0

                # slice the image
                img = tf.slice(img, [0, start_column, 0], [img_geom[0], img_geom[1], img_geom[2]])
                print("slice shape = %s" % img.get_shape())

                # get a random slice of the image, use column offsets
                #column_offsets = list(range(num_columns))[::img_geom[1]]
                #start_column = random.choice(column_offsets)
                #img = tf.slice(img, [0, start_column, 0], [img_geom[0], img_geom[1], img_geom[2]])

                #if NORMALIZE_IMAGE:
                #    img = tf.image.per_image_standardization(img)

                label = tf.cast(parsed_features['label'], tf.int64)
                label = tf.one_hot(label, depth=self.num_classes, dtype=tf.int32)

                all_images.append(img)
                all_labels.append(label)

        # convert list to ndarray
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)

        return all_images, all_labels, new_offset
