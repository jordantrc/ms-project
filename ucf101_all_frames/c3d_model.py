# c3d_model.py
#
# builds the C3D network
#
# adapted from:
# https://github.com/hx173149/C3D-tensorflow/blob/master/c3d_model.py

import tensorflow as tf
import random
import c3d
import zlib
import numpy as np
import time
import os
import cv2
import sys

from PIL import Image
from itertools import cycle

NUM_CLASSES = 101
FRAMES_PER_VIDEO = 250
NUM_FRAMES_PER_CLIP = 32
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

def inference_c3d(_X, _dropout, batch_size, _weights, _biases, depth=4):

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
  # the following line is only necessary if the sports1m pretrained model is in-use
  # pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
  dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
  dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

  dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)

  dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
  dense2 = tf.nn.dropout(dense2, _dropout)

  # Output: class prediction
  out = tf.matmul(dense2, _weights['out']) + _biases['out']

  # list of convolutional layers to select from
  layers = [conv1, conv2, conv3, conv4, conv5]

  return out, layers[depth]

def get_frame_data_tfrecord(filename, num_frames_per_clip=16):
  '''opens the tfrecord and returns the number of frames required'''
  
  ret_arr = []
  s_index = 0
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([filename], num_epochs=1)

  # open the tfrecord file for reading
  feature_dict = dict()
  feature_dict['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['num_frames'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['height'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['width'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['channels'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['frames'] = tf.FixedLenFeature([], tf.string)

  # read the tfrecord file
  _, serialized_example = reader.read(filename_queue)
  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature_dict)

  # reshape the images into an ndarray, first decompress the image data
  #frame_stack = zlib.decompress(features['frames'])
  frame_stack = tf.decode_raw(features['frames'], tf.uint8)
  print("frame_stack type = %s, shape = %s" % (type(frame_stack), str(tf.shape(frame_stack))))
  num_frames = features['num_frames']
  height = features['height']
  width = features['width']
  channels = features['channels']
  frame_shape = tf.parallel_stack([num_frames, height, width, channels])
  frames = tf.reshape(frame_stack, frame_shape)

  # pad frame data until larger than num_frames_per_clip
  # this is kind of ugly

  # sample num_frames_per_clip frames from the frame stack
  if num_frames == num_frames_per_clip:
    ret_arr = frames
  elif num_frames < num_frames_per_clip:
    # oversample
    frames = cycle(frames)
    i = 0
    while len(ret_arr) < num_frames_per_clip:
      ret_arr.append(frames[i])
      i += 1
  elif num_frames > num_frames_per_clip:
    # pick a random starting index
    s_index = random.randint(0, num_frames - num_frames_per_clip)
    ret_arr = frames[s_index:s_index + num_frames_per_clip]

  assert len(ret_arr) == num_frames_per_clip

  return ret_arr, s_index


def read_clip_and_label_tfrecord(directory, filename, classes, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  '''this function modified to work with tfrecord files'''
  lines = open(filename,'r')
  read_filenames = []
  data = []
  label = []
  sample_names = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index >= batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    filename = line[0]
    sample_name = os.path.basename(filename)
    filename = os.path.join(directory, filename + ".tfrecord")
    tmp_label = sample_name.split('_')[1]
    tmp_label = classes.index(tmp_label)
    if not shuffle:
      print("Loading a video clip from {}...".format(filename))
    tmp_data, _ = get_frame_data(filename, num_frames_per_clip)
    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      sample_names.append(sample_name)
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len, sample_names


def get_frames_data_32(filename, num_frames_per_clip=32):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  pad_size = 0
  for parent, dirnames, filenames in os.walk(filename):
    filenames = sorted(filenames)
    if(len(filenames) < num_frames_per_clip):
        #return ret_arr, s_index
        s_index = 0
        sample = Image.open(str(filename) + "/" + str(filenames[0]))
        height, width = sample.size
        blank_image = np.zeros([width, height, 3], dtype=int)
        e_index = len(filenames)
        pad_size = num_frames_per_clip - len(filenames)
    elif len(filenames) == num_frames_per_clip:
        s_index = 0
        e_index = len(filenames)
    else:
        # s_index calc also changed to have 50% overlap on clips
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        # s_index = random.randrange(0, len(filenames) - num_frames_per_clip, int(num_frames_per_clip / 8))
        e_index = s_index + num_frames_per_clip

    for i in range(s_index, e_index):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)

    for i in range(pad_size):
        ret_arr.append(blank_image)

    assert len(ret_arr) == num_frames_per_clip, "ret_arr length (%s) != num_frames_per_clip" % len(ret_arr)

  return ret_arr, s_index


def get_frames_data(filename):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    filenames = sorted(filenames)
    print("length filenames = %s" % len(filenames))
    
    for i in range(len(filenames)):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)

  return ret_arr

def read_clip_and_label(directory, filename, batch_size, start_pos=-1, crop_size=112, shuffle=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  sample_names = []
  sample_lengths = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  #np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    # print("line = %s" % line)
    dirname = line[0]
    sample_name = os.path.basename(dirname)
    dirname = os.path.join(directory, dirname)
    tmp_label = int(line[1])
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    tmp_data = get_frames_data(dirname)
    num_frames = len(tmp_data)
    print("num_frames = %s" % num_frames)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        #img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
        img_datas.append(img)
      data.append(img_datas)
      sample_names.append(sample_name)
      sample_lengths.append(num_frames)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  try:
    # print("len data = %s" % (len(data)))
    np_arr_data = np.array(data).astype(np.float32)
  except ValueError as e:
    print("type = %s, len = %s, shape = %s" % (type(data), len(data), str(data[0].shape)))
    print(e)
    sys.exit(1)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len, sample_names, sample_lengths
