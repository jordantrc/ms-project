# iad_cnn.py
#
# Convolutional Neural Network for IAD data for the UCF101 dataset
#

import fnmatch
import math
import os
import random
import sys
import tensorflow as tf

import analysis
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list


TEST_FILE_LIST = 'train-test-splits/iad_madison_test'
TRAIN_FILE_LIST = 'train-test-splits/iad_madison_train'
MODEL_SAVE_DIR = 'iad_models/'

NUM_CLASSES = 101
#CLASSES_TO_INCLUDE = ['ApplyEyeMakeup', 'Knitting', 'Lunges', 'HandStandPushups', 'Archery', 'MilitaryParade',
#                      'YoYo', 'BabyCrawling', 'BaseballPitch', 'BenchPress', 'Bowling', 'Drumming',
#                      'BalanceBeam', 'BandMarching', 'Fencing', 'FloorGymnastics', 'Haircut', 'Hammering',
#                      'HeadMassage', 'HighJump', 'HulaHoop', 'JavelinThrow', 'JumpingJack', 'Kayaking']
CLASSES_TO_INCLUDE = 'all'
TRAINING_DATA_SAMPLE = 1.0

# neural network variables
# softmax, autoencode, 
CLASSIFIER = 'lenet'
WEIGHT_STDDEV = 0.1
BIAS = 0.0
LEAKY_RELU_ALPHA = 0.04
DROPOUT = 0.5
LEARNING_RATE = 1e-5
BETA = 0.01  # used for the L2 regularization loss function
NORMALIZE_IMAGE = False
SOFTMAX_HIDDEN_SIZE = 1024

# autoencoder hyper parameters
#AUTOENCODER_LAYERS = [500, 200, 50, 10]
#AUTOENCODER_LAYERS = [512, 256, 128, 64, 32, 16, 8]
#AUTOENCODER_LAYERS = [500, 200, 50, 10]
#AUTOENCODER_LAYERS = [500, 200, 50, 10]
#AUTOENCODER_LAYERS = [500, 200, 50, 10]


# the layer from which to load the activation map
# layer geometries - shallowest to deepest
# layer 1 - 64 features x 16 time slices
# layer 2 - 128 features x 16 time slices
# layer 3 - 256 features x 8 time slices
# layer 4 - 512 features x 4 time slices
# layer 5 - 512 features x 2 time slices
#FIRST_CNN_WIDTH = 16  # mctnet
FIRST_CNN_WIDTH = 32  # lenet
#FIRST_CNN_WIDTH = -1  # softmax
# 16 frame layer dimensions
LAYER_GEOMETRY_16 = {
                    '1': (64, 16, 1),
                    '2': (128, 16, 1),
                    '3': (256, 8, 1),
                    '4': (512, 4, 1),
                    '5': (512, 2, 1)
                    }
# 32 frame layer dimensions
LAYER_GEOMETRY_32 = {
                    '1': (64, 32, 1),
                    '2': (128, 32, 1),
                    '3': (256, 16, 1),
                    '4': (512, 8, 1),
                    '5': (512, 4, 1)
                    }
LAYER_GEOMETRY_FULL_FRAME = {
                    '1': (64, 1000, 1),
                    '2': (128, 1000, 1),
                    '3': (256, 500, 1),
                    '4': (256, 250, 1),
                    '5': (256, 125, 1)
                    }
LAYER_GEOMETRY = LAYER_GEOMETRY_FULL_FRAME

#-------------General helper functions----------------#
def save_settings(run_name):
    '''saves the parameters to a file'''
    with open('runs/%s_parameters.txt' % run_name, 'w') as fd:
        fd.write("CLASSIFIER = %s\n" % CLASSIFIER)
        fd.write("BATCH_SIZE = %s\n" % BATCH_SIZE)
        fd.write("TRAIN_FILE_LIST = %s\n" % TRAIN_FILE_LIST)
        fd.write("TEST_FILE_LIST = %s\n" % TEST_FILE_LIST)
        fd.write("MODEL_SAVE_DIR = %s\n" % MODEL_SAVE_DIR)
        fd.write("LOAD_MODEL = %s\n" % LOAD_MODEL)
        fd.write("EPOCHS = %s\n" % EPOCHS)
        fd.write("CLASSES_TO_INCLUDE = %s\n" % CLASSES_TO_INCLUDE)
        fd.write("TRAINING_DATA_SAMPLE = %s\n" % TRAINING_DATA_SAMPLE)
        fd.write("WEIGHT_STDDEV = %s\n" % WEIGHT_STDDEV)
        fd.write("BIAS = %s\n" % BIAS)
        fd.write("LEAKY_RELU_ALPHA = %s\n" % LEAKY_RELU_ALPHA)
        fd.write("DROPOUT = %s\n" % DROPOUT)
        fd.write("LEARNING_RATE = %s\n" % LEARNING_RATE)
        fd.write("LAYER = %s\n" % LAYER)
        fd.write("NORMALIZE_IMAGE = %s\n" % NORMALIZE_IMAGE)
        fd.write("SOFTMAX_HIDDEN_SIZE = %s\n" % SOFTMAX_HIDDEN_SIZE)


def list_to_filenames(list_file, balance_classes=False):
    '''converts a list file to a list of filenames'''
    filenames = []
    class_counts = {}
    class_files = {}

    with open(list_file, 'r') as list_fd:
        text = list_fd.read()
        lines = text.split('\n')

    if '' in lines:
        lines.remove('')

    for l in lines:
        found_files = 0
        sample, label = l.split()
        sample_basename = os.path.basename(sample)

        class_name = sample_basename.split('_')[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
            class_files[class_name].append(sample)
        else:
            class_counts[class_name] = 1
            class_files[class_name] = [sample]


    # balance classes if we're training
    if balance_classes:
        print("balancing files across classes")
        max_class_count = -1
        for k in class_counts.keys():
            if class_counts[k] > max_class_count:
                max_class_count = class_counts[k]

        # add files to filenames, add as many as possible and then
        # sample the remainder less than max_class_count
        print("oversample size = %s" % max_class_count)
        if CLASSES_TO_INCLUDE == 'all':
            keys = class_counts.keys()
        else:
            keys = [x for x in class_counts.keys() if x in CLASSES_TO_INCLUDE]

        for k in keys:
            oversample = max_class_count - class_counts[k]
            filenames.extend(class_files[k])
            if oversample <= len(class_files[k]):
                filenames.extend(random.sample(class_files[k], oversample))
            else:
                class_filenames = class_files[k]
                while len(class_filenames) < max_class_count:
                    class_filenames.append(random.sample(class_files[k], 1)[0])
                filenames.extend(class_filenames)

    else:
        if CLASSES_TO_INCLUDE == 'all':
            keys = class_counts.keys()
        else:
            keys = [x for x in class_counts.keys() if x in CLASSES_TO_INCLUDE]

        for k in sorted(keys):
            filenames.extend(class_files[k])

    return filenames


def save_model(sess, saver, step):
    save_path = os.path.join(MODEL_SAVE_DIR, "iad_model_layer_%s_step_%s.ckpt" % (LAYER, step))
    saver.save(sess, save_path)

#-------------CNN Functions---------------------------#

def _drop_connect(W, p):
    # from https://nickcdryan.com/2017/06/13/dropconnect-implementation-in-python-and-tensorflow/
    return tf.nn.dropout(W, keep_prob=p) * p

def _weight_variable(name, shape):
    initial = tf.truncated_normal_initializer(stddev=WEIGHT_STDDEV)
    return tf.get_variable(name, shape, initializer=initial)

def _weight_variable_scaling(name, shape):
    initializer = tf.variance_scaling_initializer()
    return tf.Variable(initializer(shape), dtype=tf.float32)

def _bias_variable(name, shape):
    initial = tf.constant(BIAS, name=name, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, W, b, activation_function='leaky_relu'):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    if activation_function == 'relu':
        result = tf.nn.relu(conv)
    elif activation_function == 'leaky_relu':
        result = tf.nn.leaky_relu(conv, alpha=LEAKY_RELU_ALPHA)
    return result

def _max_pool_kxk(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def _parse_function(example):
    
    layer_padding = {
                    '1': 128000,
                    '2': 256000,
                    '3': 256000,
                    '4': 128000,
                    '5': 64000
                    }
    img_geom = tuple([1]) + LAYER_GEOMETRY[str(LAYER)]
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)
        features['num_rows/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['num_columns/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)

    parsed_features = tf.parse_single_example(example, features)
    num_rows = parsed_features['num_rows/{:02d}'.format(LAYER)]
    num_columns = parsed_features['num_columns/{:02d}'.format(LAYER)]

    # decode the image, get label
    img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
    padding = [[0, layer_padding[str(LAYER)] - tf.shape(img)[0]]]
    img = tf.pad(img, padding, 'CONSTANT', constant_values=0.0)
    #print("rows = %s columns = %s" % (
    #    parsed_features['num_rows/{:02d}'.format(LAYER)].eval(),
    #    parsed_features['num_columns/{:02d}'.format(LAYER)].eval()))
    img = tf.reshape(img, img_geom, "parse_reshape")
    print("img shape = %s" % img.get_shape())
    #img = tf.expand_dims(img, 0)
    #img = tf.expand_dims(img, 0)
    #print("3 img shape = %s" % img.get_shape())

    #if FIRST_CNN_WIDTH != -1:
    #    layer_dim3_pad = (FIRST_CNN_WIDTH - img_geom[2])

        # pad the image
    #    pad_shape = [[0, 0], [0, 0], [0, layer_dim3_pad], [0, 0]]
    #    padding = tf.constant(pad_shape)
    #    img = tf.pad(img, padding, 'CONSTANT', constant_values=-1.0)
    #    print("img shape = %s" % img.get_shape())
    #img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #print("img shape = %s" % img.get_shape())
    #img = tf.squeeze(img, 0)
    if NORMALIZE_IMAGE:
        img = tf.image.per_image_standardization(img)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.int32)

    return img, label


def get_variables_jtcnet(model_name, num_channels=1):
    geom = LAYER_GEOMETRY[str(LAYER)]
    num_features = geom[0] * geom[1] * num_channels
    with tf.variable_scope(model_name) as var_scope:
        weights = {
            'W_0': _weight_variable('W_0', [num_features, num_features * 2]),
            'W_1': _weight_variable('W_1', [num_features * 2, num_features]),
            'W_2': _weight_variable('W_2', [num_features, num_features * 0.5]),
            'W_3': _weight_variable('W_3', [num_features * 0.5, NUM_CLASSES])
        }
        biases = {
            'b_0': _bias_variable('b_0', [num_features * 2]),
            'b_1': _bias_variable('b_1', [num_features]),
            'b_2': _bias_variable('b_2', [num_features * 0.5]),
            'b_3': _bias_variable('b_3', [NUM_CLASSES])
        }
    return weights, biases


def get_variables_lenet(model_name, num_channels=1):
    # w_fc1 = _weight_variable('W_fc1', [pool2_flat_size, 1024])
    # b_fc1 = _bias_variable('b_fc1', [1024])
    # w_fc2 = _weight_variable('W_fc2', [1024, NUM_CLASSES])
    # b_fc2 = _bias_variable('b_fc2', [NUM_CLASSES])
    geom = LAYER_GEOMETRY[str(LAYER)]
    num_features = geom[0] * geom[1] * num_channels
    pool2_flat_size = num_features * 8

    with tf.variable_scope(model_name) as var_scope:
        weights = {
            'W_0': _weight_variable('W_0', [3, 3, num_channels, 32]),
            'W_1': _weight_variable('W_1', [3, 3, 32, 64]),
            'W_fc1': _weight_variable('W_fc1', [pool2_flat_size, 1024]),
            'W_fc2': _weight_variable('W_fc2', [1024, 101])
            }
        biases = {
            'b_0': _bias_variable('b_0', [32]),
            'b_1': _bias_variable('b_1', [64]),
            'b_fc1': _bias_variable('b_fc1', [1024]),
            'b_fc2': _bias_variable('b_fc2', [101])
            }
    return weights, biases


def get_variables_mctnet(model_name, num_channels=1):
    with tf.variable_scope(model_name) as var_scope:
        weights = {
                'W_0': _weight_variable('W_0', [3, 3, num_channels, 16]),
                'W_1': _weight_variable('W_1', [3, 3, 16, 16]),
                'W_2': _weight_variable('W_2', [3, 3, 16, 32]),
                'W_3': _weight_variable('W_3', [3, 3, 32, 32])
                }
        biases = {
                'b_0': _bias_variable('b_0', [16]),
                'b_1': _bias_variable('b_1', [16]),
                'b_2': _bias_variable('b_2', [32]),
                'b_3': _bias_variable('b_3', [32])
                }
    return weights, biases

def get_variables_softmax(model_name, num_channels=1):
    geom = LAYER_GEOMETRY[str(LAYER)]
    num_features = geom[0] * geom[1] * num_channels * 2

    with tf.variable_scope(model_name) as var_scope:
        weights = {
                #'W_0': _weight_variable_scaling('W_0', [num_features, NUM_CLASSES]),
                'W_0': _weight_variable('W_0', [num_features, NUM_CLASSES]),
                #'W_0': _weight_variable('W_0', [num_features, SOFTMAX_HIDDEN_SIZE]),
                #'out': _weight_variable('out', [SOFTMAX_HIDDEN_SIZE, NUM_CLASSES])
                }
        biases = {
                'b_0': _bias_variable('b_0', [NUM_CLASSES]),
                #'b_0': _bias_variable('b_0', [SOFTMAX_HIDDEN_SIZE]),
                #'out': _bias_variable('out', [NUM_CLASSES])
        }
    return weights, biases


def get_variables_autoencode(model_name, num_channels=1):
    geom = LAYER_GEOMETRY[str(LAYER)]
    num_features = geom[0] * geom[1] * num_channels

    layer_pairs = [-1] * 2 * len(AUTOENCODER_LAYERS)
    for i, l in enumerate(AUTOENCODER_LAYERS):
        if i == 0:
            layer_pairs[0] = [num_features, l]
            layer_pairs[-1] = [l, num_features]
        else:
            layer_pairs[i] = [AUTOENCODER_LAYERS[i - 1], l]
            layer_pairs[-1 - i] = [l, AUTOENCODER_LAYERS[i - 1]]

    # create weights and biases
    weights = {}
    biases = {}
    with tf.variable_scope(model_name) as var_scope:
        for i, l in enumerate(layer_pairs):
            weight_name = 'W_' + str(i)
            bias_name = 'b_' + str(i)
            weights[weight_name] = _weight_variable(weight_name, [l[0], l[1]])
            biases[bias_name] = _bias_variable(bias_name, [l[1]])

    # output weights and biases
    with tf.variable_scope(model_name) as var_scope:
        weights['W_out'] = _weight_variable('W_out', [num_features, NUM_CLASSES])
        biases['b_out'] = _bias_variable('b_out', [NUM_CLASSES])

    return weights, biases


def get_variables_temporal_softmax(model_name, num_channels=1):
    geom = LAYER_GEOMETRY[str(LAYER)]
    num_rows = geom[0]
    num_columns = geom[1]

    weights = {}
    biases = {}
    with tf.variable_scope(model_name) as var_scope:
        for i in range(num_columns):
            str_i = str(i)
            weights['W_0_' + str_i] = _weight_variable('W_0_' + str_i, [num_rows, TEMP_SOFTMAX_LAYER])
            biases['b_0_' + str_i] = _bias_variable('b_0_' + str_i, [TEMP_SOFTMAX_LAYER])
        
        weights['W_1'] = _weight_variable('W_1', [num_columns * TEMP_SOFTMAX_LAYER, NUM_CLASSES])
        biases['b_1'] = _bias_variable('b_1', [NUM_CLASSES])

    return weights, biases


def nn_jtcnet(x, batch_size, weights, biases, dropout):
    geom = LAYER_GEOMETRY[str(LAYER)]
    
    # layer 1
    x = tf.reshape(x, [batch_size, geom[0] * geom[1]])

    # first layer
    net = tf.matmul(x, weights['W_0']) + biases['b_0']
    net = tf.nn.leaky_relu(net, alpha=LEAKY_RELU_ALPHA)

    # second layer
    net = tf.matmul(net, weights['W_1']) + biases['b_1']
    net = tf.nn.leaky_relu(net, alpha=LEAKY_RELU_ALPHA)

    # third layer
    net = tf.matmul(net, weights['W_2']) + biases['b_2']
    net = tf.nn.leaky_relu(net, alpha=LEAKY_RELU_ALPHA)

    # third layer
    net = tf.nn.dropout(net, dropout)
    net = tf.matmul(net, weights['W_3']) + biases['b_3']

    return net, []


def cnn_mctnet(x, batch_size, weights, biases, dropout):
     # first layer
    conv1 = _conv2d(x, weights['W_0'], biases['b_0'])
    #pool1 = _max_pool_kxk(conv1, 2)

    # second layer
    conv2 = _conv2d(conv1, weights['W_1'], biases['b_1'])
    #pool2 = _max_pool_kxk(conv2, 2)

    # third layer
    conv3 = _conv2d(conv2, weights['W_2'], biases['b_2'])
    #pool3 = _max_pool_kxk(conv3, 2)

    # fourth layer
    conv4 = _conv2d(conv3, weights['W_3'], biases['b_3'])
    #pool4 = _max_pool_kxk(conv4, 2)

    # one fully connected layer
    #conv4_shape = conv4.get_shape().as_list()
    #print("conv4_shape = %s" % conv4.get_shape())
    #flatten_size = conv4_shape[1] * conv4_shape[2] * conv4_shape[3]
    #conv4_flat = tf.reshape(conv4, (-1, flatten_size))
    conv4_flat = tf.layers.flatten(conv4)
    flatten_size = conv4_flat.get_shape().as_list()[1]

    w_fc1 = _weight_variable('W_fc1', [flatten_size, NUM_CLASSES])
    b_fc1 = _bias_variable('b_fc1', [NUM_CLASSES])

    logits = tf.add(tf.matmul(conv4_flat, w_fc1), b_fc1)

    conv_layers = [conv1, conv4]

    return logits, conv_layers


def cnn_lenet(x, batch_size, weights, biases, dropout):

    # first layer
    conv1 = _conv2d(x, weights['W_0'], biases['b_0'], activation_function='leaky_relu')
    pool1 = _max_pool_kxk(conv1, 2)

    # second layer
    #conv2 = _conv2d(conv1, weights['W_1'], biases['b_1'], activation_function='leaky_relu')
    #pool2 = _max_pool_kxk(conv2, 2)
    pool2 = pool1

    # third layer
    #conv3 = _conv2d(pool2, weights['W_2'], biases['b_2'])
    #pool3 = _max_pool_kxk(conv3, 2)

    # fourth layer
    #conv4 = _conv2d(pool3, weights['W_3'], biases['b_3'])
    #pool4 = _max_pool_kxk(conv4, 2)
    pool2_shape = pool2.get_shape().as_list()
    print("pool2 shape = %s" % pool2_shape)

    # fully connected layer
    #pool2_flat_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

    # flatten pool2
    #pool2_flat = tf.reshape(pool2, [-1, pool2_flat_size])
    pool2_flat = tf.layers.flatten(pool2)
    print("pool2_flat shape = %s" % pool2_flat.get_shape().as_list())
    fc1 = tf.matmul(pool2_flat, weights['W_fc1']) + biases['b_fc1']
    fc1 = tf.nn.leaky_relu(fc1, alpha=LEAKY_RELU_ALPHA)

    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # readout with dropconnect
    #w_fc2 = _drop_connect(w_fc2, dropout)

    logits = tf.add(tf.matmul(fc1, weights['W_fc2']), biases['b_fc2'])

    conv_layers = [pool1]

    return logits, conv_layers


def autoencode(x, batch_size, weights, biases):
    model = tf.matmul(x, weights['W_0']) + biases['b_0']
    model = tf.nn.relu(model)
    for i in range(1, len(AUTOENCODER_LAYERS) * 2):
        w_id = 'W_' + str(i)
        b_id = 'b_' + str(i)
        
        if i == len(AUTOENCODER_LAYERS) - 1:
            model = tf.matmul(model, weights[w_id]) + biases[b_id]
            model = tf.nn.sigmoid(model)
        else:
            model = tf.matmul(model, weights[w_id]) + biases[b_id]
            model = tf.nn.relu(model)

    return model, []


def softmax_regression(x, batch_size, weights, biases, dropout):
    model = tf.matmul(x, weights['W_0']) + biases['b_0']
    #model = tf.nn.leaky_relu(model, alpha=LEAKY_RELU_ALPHA)
    #model = _drop_connect(model, dropout)
    
    # layer 2
    #model = tf.matmul(model, weights['out']) + biases['out']

    return model, []


def temporal_softmax_regression(x, batch_size, weights, biases, dropout):
    geom = LAYER_GEOMETRY[str(LAYER)]
    #x = tf.reshape(x, [batch_size, geom[0] * geom[1]])
    num_rows = geom[0]
    num_cols = geom[1]

    # a 512 row x 4 column tensor will be reshaped row by row, i.e.
    # row 0 row 1 row 2 row 3 row 4, cobe below extracts the columns
    # into slices. Index of row 1, column 0 = row * num_columns + column

    x_slices = []
    for i in range(num_cols):
        x_slice = tf.slice(x, [0, 0, i, 0], [-1, -1, 1, -1])
        x_slice = tf.squeeze(x_slice, [2, 3])  # only squeeze the final two dimensions
        x_slices.append(x_slice)
        #print("x_slice shape = %s" % (x_slice.get_shape().as_list()))

    models = []

    # first layer
    for i in range(num_cols):
        str_i = str(i)
        models.append(tf.matmul(x_slices[i], weights['W_0_' + str_i]) + biases['b_0_' + str_i])

    # second layer
    model = tf.concat(models, 1)
    print("model shape = %s" % (model.get_shape().as_list()))
    #model = tf.nn.dropout(model, dropout)
    # drop connect
    model = _drop_connect(model, dropout)
    model = tf.matmul(model, weights['W_1']) + biases['b_1']

    return model, []


def iad_nn(run_string):
    '''main function'''
    if LOAD_MODEL is None:
        training = True
    else:
        training = False

    # get the list of classes
    class_list = get_class_list(CLASS_INDEX_FILE)

    # get the list of filenames
    print("loading train file list from %s" % TRAIN_FILE_LIST)
    filenames_train = list_to_filenames(TRAIN_FILE_LIST, balance_classes=False)
    filenames_test = list_to_filenames(TEST_FILE_LIST)
    print("%s training files, %s testing files" % (len(filenames_train), len(filenames_test)))

    random.shuffle(filenames_train)
    if TRAINING_DATA_SAMPLE != 1.0:
        filenames_train = random.sample(filenames_train, int(TRAINING_DATA_SAMPLE * len(filenames_train)))

    # ensure filenames list is evenly divisable by batch size
    pad_filenames = len(filenames_train) % BATCH_SIZE
    filenames_train.extend(filenames_train[0:pad_filenames])

    # create the TensorFlow sessions
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # setup the CNN
    if CLASSIFIER == 'softmax':
        weights, biases = get_variables_softmax('ucf101_iad')
    elif CLASSIFIER == 'lenet':
        weights, biases = get_variables_lenet('ucf101_iad')
    elif CLASSIFIER == 'mctnet':
        weights, biases = get_variables_mctnet('ucf101_iad')
    elif CLASSIFIER == 'autoencode':
        weights, biases = get_variables_autoencode('ucf101_iad')

    # placeholders
    input_filenames = tf.placeholder(tf.string, shape=[None])
    input_filenames_test = tf.placeholder(tf.string, shape=[None])

    # testing dataset
    dataset_test = tf.data.TFRecordDataset(input_filenames_test)
    dataset_test = dataset_test.map(_parse_function)
    dataset_test = dataset_test.batch(1)
    dataset_test = dataset_test.shuffle(500)
    dataset_test = dataset_test.repeat(1000000)
    dataset_test_iterator = dataset_test.make_initializable_iterator()
    x_test, y_test_true = dataset_test_iterator.get_next()
    y_test_true_class = tf.argmax(y_test_true, axis=1)

    # training or evaluation dataset
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    if training:
        dataset = dataset.shuffle(200)
        dataset = dataset.repeat(EPOCHS)
    else:
        dataset = dataset.repeat(1)
    dataset_iterator = dataset.make_initializable_iterator()
    x, y_true = dataset_iterator.get_next()
    y_true_class = tf.argmax(y_true, axis=1)
    print("x shape = %s" % x.get_shape().as_list())
    print("y_true shape = %s" % y_true.get_shape().as_list())

    # get neural network response
    if CLASSIFIER in ['softmax', 'autoencode', 'lenet', 'mctnet']:
        if CLASSIFIER == 'softmax':
            geom = LAYER_GEOMETRY[str(LAYER)]
            # layer 1
            print("LAYER = %s, geom = %s" % (LAYER, geom))
            x = tf.reshape(x, [BATCH_SIZE, geom[0] * geom[1] * 2])
            x_test = tf.reshape(x_test, [1, geom[0] * geom[1] * 2])

            logits, conv_layers = softmax_regression(x, BATCH_SIZE, weights, biases, DROPOUT)
            logits_test, _ = softmax_regression(x_test, 1, weights, biases, DROPOUT)

        elif CLASSIFIER == 'lenet':
            logits, conv_layers = cnn_lenet(x, BATCH_SIZE, weights, biases, DROPOUT)
            logits_test, conv_layers = cnn_lenet(x_test, 1, weights, biases, DROPOUT)

        elif CLASSIFIER == 'mctnet':
            logits, conv_layers = cnn_mctnet(x, BATCH_SIZE, weights, biases, DROPOUT)
            logits_test, conv_layers = cnn_mctnet(x_test, 1, weights, biases, DROPOUT)

        elif CLASSIFIER == 'autoencode':
            geom = LAYER_GEOMETRY[str(LAYER)]
            # layer 1

            x = tf.reshape(x, [BATCH_SIZE, geom[0] * geom[1]])
            x_test = tf.reshape(x_test, [1, geom[0] * geom[1]])

            x_autoencode, conv_layers = autoencode(x, BATCH_SIZE, weights, biases)
            x_test_autoencode, _ = autoencode(x_test, 1, weights, biases)

            # softmax regression with output
            # final 
            logits = tf.matmul(x_autoencode, weights['W_out']) + biases['b_out']
            logits_test = tf.matmul(x_test_autoencode, weights['W_out']) + biases['b_out']

        print("logits shape = %s" % logits.get_shape().as_list())
        y_pred = tf.nn.softmax(logits)
        y_pred_class = tf.argmax(y_pred, axis=1)

        # testing/validation
        y_pred_test = tf.nn.softmax(logits_test)
        y_pred_test_class = tf.argmax(y_pred_test, axis=1)
        correct_pred_test = tf.equal(y_pred_test_class, y_test_true_class)
        accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

        # loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        # evaluation
        correct_pred = tf.equal(y_pred_class, y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    elif CLASSIFIER == 'logistic_regression':
        geom = LAYER_GEOMETRY[str(LAYER)]
        p5 = tf.constant(0.5)  # logistic regression threshold
        W = tf.Variable(tf.zeros([geom[0] * geom[1], NUM_CLASSES]))
        b = tf.Variable(tf.zeros([NUM_CLASSES]))

        x = tf.reshape(x, [BATCH_SIZE, geom[0] * geom[1]])
        x_test = tf.reshape(x_test, [1, geom[0] * geom[1]])

        y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
        y_pred_class = tf.argmax(y_pred, axis=1)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # validation/testing
        y_pred_test = tf.nn.softmax(tf.matmul(x, W) + b)
        y_pred_test_class = tf.argmax(y_pred_test, axis=1)
        correct_pred_test = tf.equal(tf.argmax(y_pred_test, 1), y_test_true_class)
        accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

    # initializer
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init_op)

    # start the training/testing steps
    if training:
        print("begin training")
        step = 0
        sess.run([
            dataset_iterator.initializer, 
            dataset_test_iterator.initializer], 
            feed_dict={
            input_filenames: filenames_train,
            input_filenames_test: filenames_test
            })

        # loop until out of data
        while True:
            try:
                train_result = sess.run([train_op, accuracy, x, y_true, y_pred, logits, weights])
                if step == 0 or step == 1:
                    print("\tweights (shape = %s) = %s" % (train_result[6]['W_0'].shape, train_result[6]['W_0']))

                if step != 0 and step % 100 == 0:
                    print("step %s, accuracy = %.03f" % (step, train_result[1]))
                    print("\tx (shape = %s) = %s" % (train_result[2].shape, train_result[2]))
                    print("\ty_true (shape = %s) = %s" % (train_result[3].shape, train_result[3]))
                    print("\ty_pred (shape = %s) = %s" % (train_result[4].shape, train_result[4]))
                    print("\tlogits (shape = %s) = %s" % (train_result[5].shape, train_result[5]))
                    print("\tweights (shape = %s) = %s" % (train_result[6]['W_0'].shape, train_result[6]['W_0']))
                    # save the current model every 1000 steps
                    if step % 1000 == 0:
                        save_model(sess, saver, step)

                        # mini batch validation
                        if step % 10000 == 0:
                            mini_batch_size = 1000
                        else:
                            mini_batch_size = 50
                        test_accuracy = 0.0
                        for i in range(mini_batch_size):
                            test_result = sess.run([accuracy_test])
                            test_accuracy += test_result[0]
                        print("mini-batch validation accuracy = %.03f" % (test_accuracy / mini_batch_size))


                # print("x = %s, logits = %s" % (train_result[2], train_result[3]))
                step += 1
            except tf.errors.OutOfRangeError:
                print("data exhausted, saving final model")
                save_model(sess, saver, 'final')
                break
        final_accuracy = -1.0
    else:
        print("begin testing")
        step = 0
        saver.restore(sess, LOAD_MODEL)
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames_test})

        cumulative_accuracy = 0.0
        predictions = []
        true_classes = []
        # loop until out of data
        while True:
            try:
                test_result = sess.run([accuracy, y_pred_class, y_true_class])
                cumulative_accuracy += test_result[0]
                predictions.append(test_result[1])
                true_classes.append(test_result[2])
                if step % 100 == 0:
                    print("step %s, accuracy = %s, cumulative accuracy = %s" %
                          (step, test_result[0], cumulative_accuracy / step / BATCH_SIZE))
                step += 1
            except tf.errors.OutOfRangeError:
                break

        # wrap up, provide test results
        final_accuracy = cumulative_accuracy / step / BATCH_SIZE
        results_fd = open("runs/" + run_string + ".txt", 'w')
        print("data exhausted, test results:")
        print("steps = %s, cumulative accuracy = %.04f" % (step, final_accuracy))
        results_fd.write("steps = %s, cumulative accuracy = %.04f\n" % (step, final_accuracy))
        #for i, p in enumerate(predictions):
        #    print("[%s] true class = %s, predicted class = %s" % (i, true_classes[i], p))

        cm = analysis.confusion_matrix(predictions, true_classes, class_list)
        print("confusion matrix = %s" % cm)
        analysis.plot_confusion_matrix(cm, class_list, "runs/" + run_string + ".pdf")
        print("per-class accuracy:")
        analysis.per_class_table(predictions, true_classes, class_list, "runs/" + run_string + '.csv')

    sess.close()
    return final_accuracy

def random_layer_list(min_layers, max_layers, min_hidden_size, max_hidden_size):
    layers = []
    num_layers = random.randint(min_layers, max_layers)
    layers.append(max_hidden_size)
    
    for n in range(1, num_layers - 1):
        size = random.randint(int(layers[n - 1] / 2), layers[n - 1])
        layers.append(size)

    if int(layers[-1] / 2) <= min_hidden_size:
        layers.append(min_hidden_size)
    else:
        layers.append(random.randint(min_hidden_size, int(layers[-1] / 2)))
    return layers


if __name__ == "__main__":

    # get the run name
    run_name = sys.argv[1]

    # various hyperparameter settings
    hyper_softmax_hidden_powers = list(range(2, 17))
    hyper_dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 100 autoencoder settings to test
    layer_list = []
    for n in range(0, 100):
        layer_list.append(random_layer_list(2, 20, 2, 512))

    # settings for softmax hidden layer size
    #hyper_settings = hyper_softmax_hidden_powers
    # settings for dropout
    hyper_settings = layer_list
    hyper_settings = hyper_dropout
    hyper_settings = [""]
    hyper_parameter = "none"

    accuracies = []

    for h in hyper_settings:
        #SOFTMAX_HIDDEN_SIZE = int(math.pow(2, h))
        #hyper_var = "SOFTMAX_HIDDEN_SIZE"
        #hyper_value = SOFTMAX_HIDDEN_SIZE
        if hyper_parameter == "dropout":
            DROPOUT = h
            hyper_name = "Dropout"
            hyper_value = DROPOUT
        elif hyper_parameter == "none":
            hyper_name = ""
            hyper_value = ""
        elif hyper_parameter == "autoencoder_layers":
            AUTOENCODER_LAYERS = h
            hyper_name = "autoencoder_layers"
            hyper_value = AUTOENCODER_LAYERS

        trial_count = 1
        for layer in [1, 2, 3, 4, 5]:
            LAYER = layer

            print("##############################")
            print("%s %s - LAYER %s" % (hyper_name, h, LAYER))
            print("##############################")

            # training run
            BATCH_SIZE = 10
            LOAD_MODEL = None
            EPOCHS = 20
            run_string = run_name + "_" + str(hyper_value) + "_" + str(LAYER) + "_train"
            save_settings(run_string)
            iad_nn(run_string)

            # reset the graph
            tf.reset_default_graph()

            # testing run
            BATCH_SIZE = 1
            LOAD_MODEL = 'iad_models/iad_model_layer_%s_step_final.ckpt' % LAYER
            EPOCHS = 1
            run_string = run_name + "_" + str(hyper_value) + "_" + str(LAYER) + "_test"
            save_settings(run_string)
            layer_accuracy = iad_nn(run_string)
            accuracies.append([hyper_value, layer, layer_accuracy])
            print("Accuracies after %s trials:" % (trial_count))
            for a in accuracies:
                print(a)
            trial_count += 1

            # reset the graph before moving to the next layer
            tf.reset_default_graph()

    # print accuracy information from all layers and hidden layer sizes
    print("Final accuracy data:")
    for a in accuracies:
        print(a)

