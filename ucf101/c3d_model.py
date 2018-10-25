# c3d_model.py
#
# builds the C3D network
#
# adapted from:
# https://github.com/hx173149/C3D-tensorflow/blob/master/c3d_model.py

import tensorflow as tf
import random


NUM_CLASSES = 101
TRAIN_DIR = "/home/jordanc/datasets/UCF-101/tfrecords/train"
TEST_DIR = "/home/jordanc/datasets/UCF-101/tfrecords/test"
MODEL_DIR = "/home/jordanc/datasets/UCF-101/model_ckpts"
DROPOUT = 0.5
FRAMES_PER_VIDEO = 250
FRAMES_PER_CLIP = 16
BATCH_SIZE = 30
LEARNING_RATE = 1e-3


def _clip_image_batch(image_batch, num_frames, randomly=True):
    '''generates a clip for each video in the batch'''

    batch_dimensions = image_batch.shape()
    # print("dimensions = %s" % dimensions)
    batch_size = batch_dimensions[0]
    num_frames_in_video = FRAMES_PER_VIDEO

    # sample frames for each video
    clip_batch = []
    for i in range(batch_size):
        clips = []
        video = image_batch[i]
        # print("video = %s, shape = %s" % (video, video.get_shape().as_list()))
        # randomly sample frames
        sample_indexes = random.sample(list(range(num_frames_in_video)), num_frames)
        sample_indexes.sort()

        # print("sample indexes = %s" % sample_indexes)

        for j in sample_indexes:
            clips.append(video[j])

        # turn clips list into a tensor
        clips = tf.stack(clips)

        clip_batch.append(clips)

    # turn clip_batch into tensor
    clip_batch = tf.stack(clip_batch)
    # print("clip_batch = %s, shape = %s" % (clip_batch, clip_batch.get_shape().as_list()))

    return clip_batch


def _parse_function(example_proto):
    """parse map function for video data"""
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)

    for i in range(FRAMES_PER_VIDEO):
        features['frames/{:04d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    # parse the features
    parsed_features = tf.parse_single_example(example_proto, features)

    # decode the encoded jpegs
    images = []
    for i in range(FRAMES_PER_VIDEO):
        # frame = tf.image.decode_jpeg(parsed_features['frames/{:04d}'.format(i)])
        frame = tf.decode_raw(parsed_features['frames/{:04d}'.format(i)], tf.uint8)
        frame = tf.reshape(frame, tf.stack([112, 112, 3]))
        frame = tf.reshape(frame, [1, 112, 112, 3])
        # normalization
        frame = tf.cast(frame, tf.float32) * (1. / 255.) - 0.5
        images.append(frame)

    # pack the individual frames into a tensor
    images = tf.stack(images)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return images, label


def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
        tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
        b)


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME')


def inference_3d(_X, _dropout, batch_size, _weights, _biases):

    print("_X = %s" % _X)

    # Convolution layer
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    # Reshape conv3 output to fit dense layer input
    print("pool5 = %s, shape = %s" % (pool5, pool5.get_shape().as_list()))
    dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

    dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout)

    # Output: class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']

    return out
