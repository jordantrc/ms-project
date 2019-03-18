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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import sys

#def get_frames_data(filename, num_frames_per_clip=16):
def get_frames_data(filename, num_frames_per_clip=16, flip=False, pad_short_clips=False):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  pad_size = 0
  for parent, dirnames, filenames in os.walk(filename):
    filenames = sorted(filenames)
    if(len(filenames) < num_frames_per_clip) and pad_short_clips:
        #return ret_arr, s_index
        s_index = 0
        sample = Image.open(str(filename) + "/" + str(filenames[0]))
        height, width = sample.size
        blank_image = np.zeros([width, height, 3], dtype=int) 
        e_index = len(filenames)
        pad_size = num_frames_per_clip - len(filenames)
    elif len(filenames) < num_frames_per_clip and not pad_short_clips:
        return ret_arr, s_index
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
        if flip:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_data = np.array(img)
        ret_arr.append(img_data)

    for i in range(pad_size):
        ret_arr.append(blank_image)

    assert len(ret_arr) == num_frames_per_clip, "ret_arr length (%s) != num_frames_per_clip" % len(ret_arr)

  return ret_arr, s_index

#def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
def read_clip_and_label(filename, 
                        batch_size, 
                        start_pos=-1, 
                        num_frames_per_clip=16, 
                        crop_size=112, 
                        shuffle=False, 
                        flip_with_probability=0.0,
                        pad_short_clips=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  sample_names = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  if flip_with_probability != 0.0:
    flip = random.random()
    if flip < flip_with_probability:
      flip = True
  else:
    flip = False
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
    dirname = line[0]
    tmp_label = line[1]
    sample_name = os.path.basename(dirname)
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip, flip, pad_short_clips)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        #print("img height = %s, width = %s" % (img.height, img.width))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        #print("img shape = %s, crop_x = %s, crop_y = %s" % (str(img.shape), crop_x, crop_y))
        #img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
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

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, sample_names, valid_len
