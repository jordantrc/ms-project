# iad.py
# 
# contains the code for generating IAD images from the
# trained model.

import time
import tensorflow as tf

MODEL = '/home/jordanc/C3D-tensorflow-master/models/c3d_ucf_model-9999'
TRAIN_LIST = 'train-test-splits/trainlist01.txt'
TEST_LIST = 'train-test-splits/testlist01.txt'
IAD_DIRECTORY = '/home/jordanc/datasets/UCF-101/iad'
NUM_CLASSES = 101
# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3
# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

# tensorflow flags
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
FLAGS = flags.FLAGS

"-----------------------------------------------------------------------------------------------------------------------"

def make_sequence_example(
	img_raw, 
	first_action,
	example_id,
	c3d_depth,
	num_channels):

	print(first_action, example_id)

	# The object we return
	ex = tf.train.SequenceExample()

	# ---- descriptive data ----

	ex.context.feature["length"].int64_list.value.append(img_raw.shape[1])
	ex.context.feature["example_id"].bytes_list.value.append(example_id)

	# ---- label data ----

	ex.context.feature["total_lab"].int64_list.value.append(first_action)
	ex.context.feature["c3d_depth"].int64_list.value.append(c3d_depth)
	ex.context.feature["num_channels"].int64_list.value.append(num_channels)
	
	# ---- data sequences ----

	def load_array(example, name, data, dtype):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		print("newShape:", np.asarray(data).astype(dtype).shape)
		fl_data.append(np.asarray(data).astype(dtype).tostring())

	load_array(ex, "img_raw", img_raw, np.float32)

	return ex


def convert_to_IAD_input(layers):
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
  print("convert_to_IAD_input: layers.shape = %s" % str(layers.shape))

  #thresholded_data = thresholding(c3d_activation_map[0], info_values["data_ratio"], compression_method, thresholding_approach)

  #print(thresholded_data)

  #ex = make_sequence_example(thresholded_data, info_values["label"][0], info_values["example_id"][0], c3d_depth, compression_method["value"])
  #print("write to: ", video_name)
  #writer = tf.python_io.TFRecordWriter(video_name)
  #writer.write(ex.SerializeToString())


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

def run_test():
  num_train_videos = len(list(open(TRAIN_LIST, 'r')))
  num_test_videos = len(list(open(TEST_LIST,'r')))
  print("Number of train videos = {}, test videos = {}".format(num_train_videos, num_test_videos))

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
      layers.append(layer)
  #print("layers type = %s, length = %s" % (type(layers), len(layers)))
  #print("layers[0] type = %s, length = %s" % (type(layers[0]), len(layers[0])))
  #print("layers[0][0] type = %s, shape = %s" % (type(layers[0][0]), layers[0][0].shape))
  # layers is a list of length gpu_num
  # layers[0] is a list of length 5
  # layers[0][0] is a tensor with shape (1, 16, 112, 112, 64)

  logits = tf.concat(logits, 0)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, MODEL)
  # And then after everything is built, start the training loop.
  bufsize = 0
  write_file = open("predict_ret.txt", "w+", bufsize)
  next_start_pos = 0
  test_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  for step in xrange(test_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = \
            c3d_model.read_clip_and_label(
                    TEST_LIST,
                    FLAGS.batch_size * gpu_num,
                    start_pos=next_start_pos
                    )
    predict_score, layers_out = sess.run([norm_score, layers],
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

    # generate IAD output
    convert_to_IAD_input(layers_out)

  write_file.close()
  print("done")

def main():
  run_test()

if __name__ == '__main__':
  main()
