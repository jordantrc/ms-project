# iad_cnn.py
#
# Convolutional Neural Network for IAD data for the UCF101 dataset
#
import os
import random
import tensorflow as tf

BATCH_SIZE = 10
FILE_LIST = 'train-test-splits/trainlist01.txt'
LOAD_MODEL = None
EPOCHS = 5
NUM_CLASSES = 101

# neural network variables
WEIGHT_STDDEV = 0.1
BIAS = 0.1
LEAKY_RELU_ALPHA = 0.01
DROPOUT = 0.5
LEARNING_RATE = 1e-3
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# the layer from which to load the activation map
# layer geometries - shallowest to deepest
# layer 1 - 64 features x 16 time slices
# layer 2 - 128 features x 16 time slices
# layer 3 - 256 features x 8 time slices
# layer 4 - 512 features x 4 time slices
# layer 5 - 512 features x 2 time slices
LAYER = 5
LAYER_GEOMETRY = (512, 2, 1)
LAYER_PAD = [[0, 0], [0, 0], [255, 255], [0, 0]]

#-------------General helper functions----------------#

def list_to_filenames(list_file):
    '''converts a list file to a list of filenames'''
    filenames = []
    iad_directory = '/home/jordanc/datasets/UCF-101/iad'

    with open(list_file, 'r') as list_fd:
        text = list_fd.read()
        lines = text.split('\n')

    if '' in lines:
        lines.remove('')

    for l in lines:
        sample, label = l.split()
        sample_basename = os.path.basename(sample)
        iad_file = sample_basename + ".tfrecord"
        iad_file_path = os.path.join(iad_directory, iad_file)
        filenames.append(iad_file_path)

    return filenames


#-------------CNN Functions---------------------------#

def _weight_variable(name, shape):
    initial = tf.truncated_normal_initializer(stddev=WEIGHT_STDDEV)
    return tf.get_variable(name, shape, initializer=initial)

def _bias_variable(name, shape):
    initial = tf.constant(BIAS, name=name, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    return tf.nn.leaky_relu(conv, alpha=LEAKY_RELU_ALPHA)

def _max_pool_kxk(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def _parse_function(example):
    img_geom = tuple([1]) + LAYER_GEOMETRY
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    # decode the image, get label
    img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
    img = tf.reshape(img, img_geom)
    print("img shape = %s" % img.get_shape())

    # pad the image to make it square and then resize
    padding = tf.constant(LAYER_PAD)
    img = tf.pad(img, padding, 'CONSTANT')
    print("img shape = %s" % img.get_shape())
    img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    print("img shape = %s" % img.get_shape())
    img = tf.squeeze(img, 0)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return img, label


def get_variables(model_name, num_channels=1):
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


def cnn_inference(x, weights, biases, dropout):

    # first layer
    conv1 = _conv2d(x, weights['W_0'], biases['b_0'])
    pool1 = _max_pool_kxk(conv1, 2)

    # second layer
    conv2 = _conv2d(pool1, weights['W_1'], biases['b_1'])
    pool2 = _max_pool_kxk(conv2, 2)

    # third layer
    conv3 = _conv2d(pool2, weights['W_2'], biases['b_2'])
    pool3 = _max_pool_kxk(conv3, 2)

    # fourth layer
    conv4 = _conv2d(pool3, weights['W_3'], biases['b_3'])
    pool4 = _max_pool_kxk(conv4, 2)
    pool4_shape = pool4.get_shape().as_list()
    print("pool4 shape = %s" % pool4_shape)

    # fully connected layer
    w_fc1 = _weight_variable('W_fc1', [pool4_shape[1] * pool4_shape[2] * pool4_shape[3], 1024])
    b_fc1 = _bias_variable('b_fc1', [1024])

    # flatten pool4
    pool4_flat = tf.reshape(pool4, [-1, pool4_shape[1] * pool4_shape[2] * pool4_shape[3]])
    fc1 = tf.nn.leaky_relu(tf.matmul(pool4_flat, w_fc1) + b_fc1)

    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # readout
    w_fc2 = _weight_variable('W_fc2', [1024, NUM_CLASSES])
    b_fc2 = _bias_variable('b_fc2', [NUM_CLASSES])

    logits = tf.add(tf.matmul(fc1, w_fc2), b_fc2)

    return logits

def main():
    '''main function'''
    if LOAD_MODEL is None:
        training = True
    else:
        training = False

    # get the list of filenames
    filenames = list_to_filenames(FILE_LIST)
    if training:
        random.shuffle(filenames)

    # ensure filenames list is evenly divisable by batch size
    pad_filenames = len(filenames) % BATCH_SIZE
    filenames.extend(filenames[0:pad_filenames])

    # create the TensorFlow sessions
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # setup the CNN
    weights, biases = get_variables('ucf101_iad')

    # placeholders
    input_filenames = tf.placeholder(tf.string, shape=[None])
    dropout = tf.placeholder(tf.float32)

    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    if training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat(EPOCHS)
    else:
        dataset = dataset.repeat(1)
    dataset_iterator = dataset.make_initializable_iterator()
    x, y_true = dataset_iterator.get_next()

    y_true_class = tf.argmax(y_true, axis=1)

    # get neural network response
    logits = cnn_inference(x, weights, biases, dropout)
    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, axis=1)

    # loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    # evaluation
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initializer
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init_op)

    # load model
    if LOAD_MODEL is not None:
        saver.restore(sess, LOAD_MODEL)

    # start the training/testing steps
    step = 0
    sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

    # loop until out of data
    while True:
        try:
            feed_dict = {'dropout': DROPOUT}
            train_result = sess.run([train_op, accuracy], feed_dict={dropout: DROPOUT})
            print("step %s, accuracy = %s" % accuracy)
        except tf.errors.OutOfRangeError:
            print("data exhausted")
            break


if __name__ == "__main__":
    main()