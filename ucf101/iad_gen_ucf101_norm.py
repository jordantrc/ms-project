# iad_gen_ucf101_norm.py
# 
# contains the code for generating IAD images from the
# trained model, normalizes all data based on observed max
# and min values.

import fnmatch
import time
import PIL.Image as Image
import numpy as np
import os
import random
import tensorflow as tf

import c3d_model
from thresholding_3d import thresholding

MODEL = '/home/jordanc/C3D-tensorflow-master/models/c3d_ucf_model-4999'
IMAGE_DIRECTORY = '/home/jordanc/datasets/UCF-101/UCF-101/'
TRAIN_LIST = 'train-test-splits/train.list'
TEST_LIST = 'train-test-splits/test.list'
IAD_DIRECTORY = '/home/jordanc/datasets/UCF-101/iad_global_norm_32/'
NPY_DIRECTORY = '/home/jordanc/datasets/UCF-101/iad_global_norm_32/npy/'
TRAIN_EPOCHS = 1
NUM_CLASSES = 101
# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3
# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 32
COMPRESSION = {"type": "max", "value": 1, "num_channels": 1}
THRESHOLDING = "norm"

# 16 frame layer dimensions
LAYER_DIMENSIONS_16 = [
                    (64, 16, 1),
                    (128, 16, 1),
                    (256, 8, 1),
                    (512, 4, 1),
                    (512, 2, 1)
                    ]
# 32 frame layer dimensions
LAYER_DIMENSIONS_32 = [
                    (64, 32, 1),
                    (128, 32, 1),
                    (256, 16, 1),
                    (512, 8, 1),
                    (512, 4, 1)
                    ]
LAYER_DIMENSIONS = LAYER_DIMENSIONS_32

# tensorflow flags
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
FLAGS = flags.FLAGS

"-----------------------------------------------------------------------------------------------------------------------"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_sequence_example(img_raw, label, example_id, num_channels):
    """creates the tfrecord example"""
    assert len(img_raw) == 5
    features = dict()
    features['example_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[example_id]))
    features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    features['num_channels'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[num_channels]))

    for i, img in enumerate(img_raw):
        layer = i + 1
        #print("img shape = %s" % str(img.shape))
        assert img.shape == LAYER_DIMENSIONS[i], "invalid dimensions for img, shape is %s" % str(img.shape)
        img = img.tostring()
        features['img/{:02d}'.format(layer)] = _bytes_feature(img)

    example = tf.train.Example(features=tf.train.Features(feature=features))


    #print("ENTER make_sequence_example:")
    # print(len(img_raw))
    #print(label, example_id)

    # The object we return
    #ex = tf.train.SequenceExample()

    # ---- descriptive data ----
    #ex.context.feature["example_id"].bytes_list.value.append(example_id)
    #ex.context.feature["label"].int64_list.value.append(label)
    # ex.context.feature["c3d_depth"].int64_list.value.append(c3d_depth)
    #ex.context.feature["num_channels"].int64_list.value.append(num_channels)

    #for i, img in enumerate(img_raw):
    #    layer = i + 1
    #    ex.context.feature["length/{:02d}".format(layer)].int64_list.value.append(img.shape[1])
        
        # ---- data sequences ----

    #    def load_array(example, name, data, dtype):
    #        fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
    #        print("newShape:", np.asarray(data).astype(dtype).shape)
    #        fl_data.append(np.asarray(data).astype(dtype).tostring())

    #    load_array(ex, "img/{:02d}".format(layer), img, np.float32)

    return example


def get_file_sequence(directory, sample, extension):
  '''find the next sequence number for oversampling'''
  # get a list of files matching this name from the directory
  file_list = os.listdir(directory)
  file_list = [x for x in file_list if extension in x]

  matching_samples = []
  pattern = sample + "*" + extension
  for f in file_list:
    if fnmatch.fnmatch(f, pattern):
      matching_samples.append(f)
  
  # if there are already samples, just pick the next index
  if len(matching_samples) > 0:
    last_index = int(sorted(matching_samples)[-1].split('_')[4].split('.')[0])
  else:
    last_index = -1
  new_index = last_index + 1

  return new_index


def get_min_maxes(directory, layers, sample_names, labels, mins, maxes, compression_method, thresholding_approach):
  '''returns new minimums and maximum values determined from the activation layers'''
  num_layers = 5
  assert len(layers) % num_layers == 0

  for i, s in enumerate(sample_names):
    # get a list of files matching this name from the directory
    new_index = get_file_sequence(NPY_DIRECTORY, s, '.npy')

    s_index = i * num_layers
    sample_layers = layers[s_index:s_index + num_layers]  # layers ordered from 1 to 5
    assert len(sample_layers) == num_layers, "sample_layers has invalid length - %s" % len(sample_layers)

    thresholded_data = []
    for i, l in enumerate(sample_layers):
        layer_data = np.squeeze(l, axis=0)
        data_ratio = float(layer_data.shape[0] / 32.0)  # num columns / 16
        #print("layer_data shape = %s, ratio = %s" % (str(layer_data.shape), data_ratio))
        data = thresholding(layer_data, data_ratio, compression_method, thresholding_approach)
        #if i == 0:
        #  print("layer 0 thresholded data = %s, min = %s" % (data, np.min(data)))
        thresholded_data.append(data)

    # for each layer, determine the min, max values for each row
    for j, l in enumerate(thresholded_data):
      #print("%s l.shape = %s, mins[j].shape = %s" % (j, l.shape, mins[j].shape))
      assert l.shape[0] == mins[j].shape[0], "l.shape[0] %s != mins[i].shape[0] %s" % (l.shape[0], mins[j].shape[0])
      for k, row in enumerate(l):
        row_max = np.max(row)
        row_min = np.min(row)
        #print("row = %s, max = %s, min = %s" % (k, row_max, row_min))

        if row_max > maxes[j][k]:
          #print("new max for layer %s, row %s - %s > %s" % (j, k, row_max, maxes[j][k]))
          maxes[j][k] = row_max

        if row_min < mins[j][k]:
          #print("new min for layer %s, row %s - %s < %s" % (j, k, row_min, mins[j][k]))
          mins[j][k] = row_min

      # save the layer data
      # sample_sequence_layer.npy
      npy_filename = "%s_%02d_%s.npy" % (s, new_index, j + 1)
      npy_path = os.path.join(NPY_DIRECTORY, npy_filename)
      np.save(npy_path, l)
      print("write npy to %s" % (npy_path))

  return mins, maxes


def rethreshold_iad(iad, mins, maxes):
  '''re-threshold given iad with new max and mins'''
  assert iad.shape[0] == mins.shape[0], "iad and mins/maxes different shapes %s/%s" % (iad.shape[0], mins.shape[0])

  for index in range(iad.shape[0]):
    data_row = iad[index]

    max_val_divider = maxes[index] - mins[index]
    data_row -= mins[index]
    if max_val_divider != 0.0:
      data_row /= max_val_divider
    else:
      data_row = list(np.zeros_like(data_row))

    # clip values between [0.0, 1.0]
    floor_values = data_row > 1.0
    data_row[floor_values] = 1.0

    ceil_values = data_row < 0.0
    data_row[ceil_values] = 0.0

    iad[index] = data_row

  return iad


def threshold_data(list_file, training=False, mins=None, maxes=None):
    '''
    Provides the training input for the ITR network by generating an IAD from the
    activation map of the C3D network. Outputs two dictionaries. The first contains
    the placeholders that will be used when evaluating the full ITR model. The second 
    contains information about the observation being read (ie. true labels, number of
    prompts, file name, etc). 
    -placeholders: the list of placeholders used by the network
    -tf_records: the TFRecord data source to read from
    -sess: the tensorflow Session
    -c3d_model: the c3d network model
    '''
    print("thresholding data for %s" % list_file)
    with open(list_file, 'r') as fd:
      file_list = fd.read().split('\n')
    while '' in file_list:
      file_list.remove('')
    count = 0

    # get the list of files from the npy directory
    npy_files = sorted(os.listdir(NPY_DIRECTORY))

    # filter to only the files included in the list_file
    for f in file_list:
      sample_files = []
      filepath, label = f.split()
      label = int(label)
      sample_name = os.path.basename(filepath)
      
      # obtain full list of files for each sample
      for n in npy_files:
        if sample_name in n:
          sample_files.append(n)

      # for each sample file, threshold the activations and create the tfrecord output
      # split sample files into lists by sequence number
      num_samples = len(sample_files) / 5
      for i in range(num_samples):
        s_index = i * 5
        sequence = int(sample_files[s_index].split("_")[4])
        sample_layer_files = sample_files[s_index:s_index + 5]
        assert len(sample_layer_files) == 5
        
        # threshold the layer data
        thresholded_data = []
        for l, s in enumerate(sample_layer_files):
          layer_data = np.load(os.path.join(NPY_DIRECTORY, s))
          # threshold using the min and maxes for the layer
          layer_mins = mins[l]
          layer_maxes = maxes[l]
          rethreshold_data = rethreshold_iad(layer_data, layer_mins, layer_maxes)
          rethreshold_data = np.expand_dims(rethreshold_data, 2)
          thresholded_data.append(rethreshold_data)

        # create tfrecord and write to file
        assert len(thresholded_data) == 5
        ex = make_sequence_example(thresholded_data, label, sample_name + "_" + str(sequence), 1)
        tfrecord_path = os.path.join(IAD_DIRECTORY, "%s_%02d.tfrecord" % (sample_name, sequence))
        print("write tfrecord to: %s" % tfrecord_path)
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        writer.write(ex.SerializeToString())
        count += 1

        # generate images with 5% chance
        if random.random() < 0.05:
          for i, d in enumerate(thresholded_data):
              img_name = os.path.join(IAD_DIRECTORY, "%s_%02d_%s.jpg" % (sample_name, sequence, i))
              print("write test image to: %s" % img_name)
              #print("single layer type = %s, shape = %s" % (type(d), str(thresholded_data[layer_to_test].shape)))
              pixels = np.squeeze(d, axis=2)
              rescaled = (255.0 / pixels.max() * (pixels - pixels.min())).astype(np.uint8)
              img = Image.fromarray(rescaled)
              img = img.convert("L")
              # img = img.resize((img.width * 10, img.height))
              img.save(img_name, quality=95)

      print("generated %s tfrecord files for %s" % (count, list_file))

    '''num_layers = 5
    assert len(layers) % num_layers == 0
    # assert (len(layers) / num_layers) == len(sample_names), "layers list and sample_names list have different lengths (%s/%s)" % (len(layers), len(sample_names))
    # print("sample_names = %s" % (sample_names))

    for i, s in enumerate(sample_names):
        # get a list of files matching this name from the directory
        new_index = get_file_sequence(directory, s, '.tfrecord')
        video_name = os.path.join(directory, "%s_%02d.tfrecord" % (s, new_index))

        s_index = i * num_layers
        sample_layers = layers[s_index:s_index + num_layers]
        assert len(sample_layers) == num_layers, "sample_layers has invalid length - %s" % len(sample_layers)

        thresholded_data = []
        for l in sample_layers:
            layer_data = np.squeeze(l, axis=0)
            thresholded_data.append(thresholding(layer_data, compression_method, thresholding_approach))
            # print("thresholded_data shape = %s" % str(thresholded_data.shape))

        # generate the tfrecord
        ex = make_sequence_example(thresholded_data, labels[i], s, compression_method["value"])
        print("write tfrecord to: %s" % video_name)
        writer = tf.python_io.TFRecordWriter(video_name)
        writer.write(ex.SerializeToString())

        # generate images with 5% chance
        if random.random() < 0.05:
          for i, d in enumerate(thresholded_data):
              img_name = os.path.join(directory, "%s_%02d_%s.jpg" % (s, new_index, i))
              print("write test image to: %s" % img_name)
              #print("single layer type = %s, shape = %s" % (type(d), str(thresholded_data[layer_to_test].shape)))
              pixels = np.squeeze(d, axis=2)
              rescaled = (255.0 / pixels.max() * (pixels - pixels.min())).astype(np.uint8)
              img = Image.fromarray(rescaled)
              img = img.convert("L")
              # img = img.resize((img.width * 10, img.height))
              img.save(img_name, quality=95)'''


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

  conv_layers = [conv1, conv2, conv3, conv4, conv5]

  return out, conv_layers


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
                                                         NUM_FRAMES_PER_CLIP,
                                                         CROP_SIZE,
                                                         CROP_SIZE,
                                                         CHANNELS))
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


def layer_array(layer, value):
  a = np.empty((LAYER_DIMENSIONS[layer][0]))
  a[:] = value
  return a


def generate_iads(list_file, training=False):
    max_vals = [
                layer_array(0, np.NINF),
                layer_array(1, np.NINF),
                layer_array(2, np.NINF),
                layer_array(3, np.NINF),
                layer_array(4, np.NINF)
               ]
    min_vals = [
                layer_array(0, np.Inf),
                layer_array(1, np.Inf),
                layer_array(2, np.Inf),
                layer_array(3, np.Inf),
                layer_array(4, np.Inf)
                ]

    num_videos = len(list(open(list_file, 'r')))

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
                #'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd1': _variable_with_weight_decay('wd1', [16384, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, NUM_CLASSES], 0.04, 0.005)
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
                'out': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
                }
    logits = []
    layers = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit, layer = inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
            logits.append(logit)
            layers.extend(layer)
    #print("layers type = %s, length = %s" % (type(layers), len(layers)))
    #print("layers[0] type = %s, length = %s" % (type(layers[0]), len(layers[0])))
    #print("layers[0][0] type = %s, shape = %s" % (type(layers[0][0]), layers[0][0].shape))
    # layers is a list of length 10 (5 * gpu_num)
    # layers[0] is a tensor with shape (1, 16, 112, 112, 64)

    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, MODEL)

    # And then after everything is built, start the training loop.
    #for list_file in [TEST_LIST, TRAIN_LIST]:
    print("Generting IADs for %s" % list_file)
    # only write predictions if it's a test list
    if training:
        predict_write_file = None
        epochs = TRAIN_EPOCHS
    else:
        predict_write_file = "predict_ret.txt"
        write_file = open(predict_write_file, "w", 0)
        epochs = 1
    
    # determine if oversampling should be used
    for e in range(epochs):
      next_start_pos = 0
      valid_files_processed = 0

      while valid_files_processed < num_videos:
          # Fill a feed dictionary with the actual set of images and labels
          # for this particular training step.
          start_time = time.time()
          test_images, test_labels, next_start_pos, _, valid_len, sample_names = \
                  c3d_model.read_clip_and_label(
                          IMAGE_DIRECTORY,
                          list_file,
                          FLAGS.batch_size * gpu_num,
                          start_pos=next_start_pos
                          )
          predict_score, layers_out = sess.run([norm_score, layers],
                  feed_dict={images_placeholder: test_images}
                  )

          if predict_write_file is not None:
              for i in range(0, valid_len):
                true_label = test_labels[i],
                top1_predicted_label = np.argmax(predict_score[i])
                # Write results: true label, class prob for true label, predicted label, class prob for predicted label
                write_file.write('{}, {}, {}, {}\n'.format(
                        true_label[0],
                        predict_score[i][true_label],
                        top1_predicted_label,
                        predict_score[i][top1_predicted_label]))

          #for i, l in enumerate(layers_out):
          #  print("layer %s = type = %s, shape %s" % (i, type(l), l.shape))
          valid_files_processed += valid_len
          # add to min/max values, store the temporary activation result
          min_vals, max_vals = get_min_maxes(IAD_DIRECTORY, layers_out, sample_names, test_labels, min_vals, max_vals, COMPRESSION, "none")
          #convert_to_IAD_input(IAD_DIRECTORY, layers_out, sample_names, test_labels, COMPRESSION, THRESHOLDING)
          end_time = time.time()
          print("[%s:%s:%s/%s - %.3fs]" % (list_file, e, valid_files_processed, num_videos, end_time - start_time))

      if predict_write_file is not None:
          write_file.close()
    print("done generating IADs for %s" % list_file)

    return min_vals, max_vals


def main():
  # generate training data, obtain max values first
  mins, maxes = generate_iads(TRAIN_LIST, training=True)
  threshold_data(TRAIN_LIST, training=True, mins=mins, maxes=maxes)

  tf.reset_default_graph()
  generate_iads(TEST_LIST)
  threshold_data(TEST_LIST, mins=mins, maxes=maxes)

  # save the mins and maxes
  for i, m in enumerate(mins):
    file_path = os.path.join(NPY_DIRECTORY, "min_%s.npy" % (i))
    np.save(file_path, m)
    file_path = os.path.join(NPY_DIRECTORY, "max_%s.npy" % (i))
    np.save(file_path, maxes[i])


if __name__ == '__main__':
  main()
