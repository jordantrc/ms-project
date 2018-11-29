# c3d_model.py
#
# builds the C3D network
#
# adapted from:
# https://github.com/hx173149/C3D-tensorflow/blob/master/c3d_model.py

import tensorflow as tf
import random
import c3d

NUM_CLASSES = 101
FRAMES_PER_VIDEO = 250
NUM_FRAMES_PER_CLIP = 16
CROP_SIZE = 112
CHANNELS = 3

class C3DModel():

    def __init__(self,
                 num_classes=NUM_CLASSES,
                 class_map=None,
                 model_dir="/home/jordanc/datasets/UCF-101/model_ckpts",
                 tfrecord_dir="/home/jordanc/datasets/UCF-101/tfrecords",
                 dropout=0.5,
                 frames_per_video=FRAMES_PER_VIDEO,
                 frames_per_clip=NUM_FRAMES_PER_CLIP,
                 learning_rate=1e-3):
        '''initializes the object'''
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.tfrecord_dir = tfrecord_dir
        self.dropout = dropout
        self.frames_per_video = frames_per_video
        self.frames_per_clip = frames_per_clip
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        if class_map is not None:
            new_map = None
            for i in range(NUM_CLASSES):
                if i == 0 and i not in class_map:
                    new_map = tf.constant([-1])
                elif i == 0 and i in class_map:
                    new_map = tf.constant([0])
                elif i in class_map:
                    new_map = tf.concat([new_map, tf.constant([class_map.index(i)])], 0)
                else:
                    new_map = tf.concat([new_map, tf.constant([-1])], 0)
            self.class_map = new_map
        else:
            self.class_map = None

    def _clip_image_batch(self, image_batch, num_frames, randomly=True):
        '''generates a clip for each video in the batch'''

        dimensions = image_batch.get_shape().as_list()
        # print("dimensions = %s" % dimensions)
        batch_size = dimensions[0]
        num_frames_in_video = dimensions[1]

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

    def _parse_function(self, example_proto):
        """parse map function for video data"""
        features = dict()
        features['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)
        features['num_frames'] = tf.FixedLenFeature((), tf.int64, default_value=0)

        for i in frame_indexes:
            features['frames/{:04d}'.format(i)] = tf.FixedLenFeature((), tf.string)        

        # choose a starting index for the clip
        s_index = random.randint(0, self.frames_per_video - 1 - self.frames_per_clip)
        frame_indexes = range(s_index, s_index + self.frames_per_clip)

        for i in frame_indexes:
            features['frames/{:04d}'.format(i)] = tf.FixedLenFeature((), tf.string)

        # parse the features
        parsed_features = tf.parse_single_example(example_proto, features)

        # decode the encoded jpegs
        images = []
        for i in frame_indexes:
            # frame = tf.image.decode_jpeg(parsed_features['frames/{:04d}'.format(i)])
            frame = tf.decode_raw(parsed_features['frames/{:04d}'.format(i)], tf.uint8)
            # normalize the frame
            frame = tf.reshape(frame, tf.stack([112, 112, 3]))
            frame = tf.image.per_image_standardization(frame)
            frame = tf.reshape(frame, [1, 112, 112, 3])
            # normalization
            # frame = tf.cast(frame, tf.float32) * (1.0 / 255.0) - 0.5
            images.append(frame)

        # pack the individual frames into a tensor
        images = tf.stack(images)

        label = tf.cast(parsed_features['label'], tf.int64)
        #if self.class_map is not None:
        #    label = self.class_map[label]
        label = tf.one_hot(label, depth=self.num_classes)

        return images, label

    def conv3d(self, name, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
            b)

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME')

    def inference_3d(self, _X, _weights, _biases, batch_size, train):

        if not train:
            dropout = 0.6
        else:
            dropout = self.dropout

        # Convolution layer
        conv1 = self.conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = self.max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = self.conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = self.max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = self.conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = self.conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = self.max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = self.conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = self.conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = self.max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = self.conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = self.conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = self.max_pool('pool5', conv5, k=2)

        # Fully connected layer
        # this next line is only necessary if loading the sports1m model
        # pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        # Reshape conv3 output to fit dense layer input
        # print("pool5 = %s, shape = %s" % (pool5, pool5.get_shape().as_list()))
        dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        dense1 = tf.nn.dropout(dense1, dropout)

        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
        dense2 = tf.nn.dropout(dense2, dropout)

        # Output: class prediction
        out = tf.matmul(dense2, _weights['wdout']) + _biases['bdout']

        return out

    def c3d(self, _X, training):
        '''based on https://github.com/tqvinhcs/C3D-tensorflow/blob/master/m_c3d.py'''

        net = tf.layers.conv3d(inputs=_X, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  

        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  
        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
        net = tf.identity(net, name='fc1')
        net = tf.layers.dropout(inputs=net, rate=self.dropout, training=training)

        net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
        net = tf.identity(net, name='fc2')
        net = tf.layers.dropout(inputs=net, rate=self.dropout, training=training)

        net = tf.layers.dense(inputs=net, units=self.num_classes, activation=None)
        net = tf.identity(net, name='logits')
        return net


def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def inference_c3d(_X, _dropout, batch_size, _weights, _biases):

  # Convolution Layer
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
  # pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
  dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
  dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

  dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)

  dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
  dense2 = tf.nn.dropout(dense2, _dropout)

  # Output: class prediction
  out = tf.matmul(dense2, _weights['out']) + _biases['out']

  return out