# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# modified by Jordan Chadwick
# jordanc@wildcats.unh.edu

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np

from itertools import cycle

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test():
  model_name = "./sports1m_finetuning_ucf101.model"
  test_list_file = 'list/test.list'
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  for gpu_index in range(0, gpu_num):
    with tf.device('/gpu:%d' % gpu_index):
      logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
      logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  bufsize = 0
  write_file = open("predict_ret.txt", "w+", bufsize)
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  for step in xrange(all_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = read_clip_and_label(
                                                                                test_list_file,
                                                                                FLAGS.batch_size * gpu_num,
                                                                                start_pos=next_start_pos
                                                                                )
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
    for i in range(0, valid_len):
      true_label = test_labels[i],
      top1_predicted_label = np.argmax(predict_score[i])
      # Write results: true label, class prob for true label, predicted label, class prob for predicted label
      write_file.write('{}, {}, {}, {}\n'.format(
              true_label[0],
              predict_score[i][true_label],
              top1_predicted_label,
              predict_score[i][top1_predicted_label]))
  write_file.close()
  print("done")


def get_frame_data(filename, num_frames_per_clip=16):
  '''opens the tfrecord and returns the number of frames required'''
  
  ret_arr = []
  s_index = 0
  reader = tf.TFRecordReader()

  # open the tfrecord file for reading
  feature_dict = dict()
  feature_dict['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['num_frames'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['height'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['width'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['channels'] = tf.FixedLenFeature((), tf.int64, default_value=0)
  feature_dict['frames'] = tf.FixedLenFeature((), tf.string)

  # read the tfrecord file
  _, serialized_example = reader.read(filename)
  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature_dict)

  # reshape the images into an ndarray
  frame_stack = tf.decode_raw(features['frames'], tf.uint8)
  num_frames = features['num_frames']
  height = features['height']
  width = features['width']
  channels = features['channels']
  frames = tf.reshape(frame_stack, [num_frames, height, width, channels])

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


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  '''this function modified to work with tfrecord files'''
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
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
    tmp_label = line[1]
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

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len


def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
