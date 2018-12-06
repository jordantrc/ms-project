# iad_cnn.py
#
# Convolutional Neural Network for IAD data for the UCF101 dataset
#
import os
import random
import tensorflow as tf

BATCH_SIZE = 1
FILE_LIST = 'train-test-splits/testlist01.txt'
MODEL_SAVE_DIR = 'iad_models/'
LOAD_MODEL = 'iad_models/iad_model_layer_1_step_final.ckpt'
#LOAD_MODEL = None
EPOCHS = 1
NUM_CLASSES = 101

# neural network variables
WEIGHT_STDDEV = 0.04
BIAS = 0.04
LEAKY_RELU_ALPHA = 0.01
DROPOUT = 0.5
LEARNING_RATE = 1e-3

# the layer from which to load the activation map
# layer geometries - shallowest to deepest
# layer 1 - 64 features x 16 time slices
# layer 2 - 128 features x 16 time slices
# layer 3 - 256 features x 8 time slices
# layer 4 - 512 features x 4 time slices
# layer 5 - 512 features x 2 time slices
FIRST_CNN_WIDTH = 32
LAYER = 1
LAYER_GEOMETRY = {'1': (64, 16, 1),
                  '2': (128, 16, 1),
                  '3': (256, 8, 1),
                  '4': (512, 4, 1),
                  '5': (512, 2, 1)
                  }

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


def save_model(sess, saver, step):
    save_path = os.path.join(MODEL_SAVE_DIR, "iad_model_layer_%s_step_%s.ckpt" % (LAYER, step))
    saver.save(sess, save_path)

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
    img_geom = tuple([1]) + LAYER_GEOMETRY[str(LAYER)]
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    # decode the image, get label
    img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
    img = tf.reshape(img, img_geom, "parse_reshape")

    # determine padding
    layer_dim3_pad = (FIRST_CNN_WIDTH - img_geom[2]) / 2

    # pad the image to make it square and then resize
    pad_shape = [[0, 0], [0, 0], [layer_dim3_pad, layer_dim3_pad], [0, 0]]
    padding = tf.constant(pad_shape)
    img = tf.pad(img, padding, 'CONSTANT')
    print("img shape = %s" % img.get_shape())
    #img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #print("img shape = %s" % img.get_shape())
    img = tf.squeeze(img, 0)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return img, label


def get_variables_lenet(model_name, num_channels=1):
    with tf.variable_scope(model_name) as var_scope:
        weights = {
            'W_0': _weight_variable('W_0', [5, 5, num_channels, 32]),
            'W_1': _weight_variable('W_1', [5, 5, 32, 64])
            }
        biases = {
            'b_0': _bias_variable('b_0', [32]),
            'b_1': _bias_variable('b_1', [64])
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
    conv4_shape = conv4.get_shape().as_list()
    flatten_size = conv4_shape[1] * conv4_shape[2] * conv4_shape[3]
    conv4_flat = tf.reshape(conv4, (-1, flatten_size))

    w_fc1 = _weight_variable('W_fc1', [flatten_size, NUM_CLASSES])
    b_fc1 = _bias_variable('b_fc1', [NUM_CLASSES])

    logits = tf.add(tf.matmul(conv4_flat, w_fc1), b_fc1)

    return logits


def cnn_lenet(x, batch_size, weights, biases, dropout):

    # first layer
    conv1 = _conv2d(x, weights['W_0'], biases['b_0'])
    pool1 = _max_pool_kxk(conv1, 2)

    # second layer
    conv2 = _conv2d(pool1, weights['W_1'], biases['b_1'])
    pool2 = _max_pool_kxk(conv2, 2)

    # third layer
    #conv3 = _conv2d(pool2, weights['W_2'], biases['b_2'])
    #pool3 = _max_pool_kxk(conv3, 2)

    # fourth layer
    #conv4 = _conv2d(pool3, weights['W_3'], biases['b_3'])
    #pool4 = _max_pool_kxk(conv4, 2)
    pool2_shape = pool2.get_shape().as_list()
    print("pool2 shape = %s" % pool2_shape)

    # fully connected layer
    pool2_flat_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    w_fc1 = _weight_variable('W_fc1', [pool2_flat_size, 1024])
    b_fc1 = _bias_variable('b_fc1', [1024])

    # flatten pool2
    #pool2_flat = tf.reshape(pool2, [-1, pool2_flat_size])
    pool2_flat = tf.layers.flatten(pool2)
    print("pool2_flat shape = %s" % pool2_flat.get_shape().as_list())
    fc1 = tf.matmul(pool2_flat, w_fc1) + b_fc1
    fc1 = tf.nn.leaky_relu(fc1, alpha=LEAKY_RELU_ALPHA)

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
    print("loading file list from %s" % FILE_LIST)
    filenames = list_to_filenames(FILE_LIST)
    if training:
        random.shuffle(filenames)

    # ensure filenames list is evenly divisable by batch size
    pad_filenames = len(filenames) % BATCH_SIZE
    filenames.extend(filenames[0:pad_filenames])
    print("filenames = %s..." % filenames[0:5])

    # create the TensorFlow sessions
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # setup the CNN
    weights, biases = get_variables_mctnet('ucf101_iad')

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
    print("x shape = %s" % x.get_shape().as_list())
    print("y_true shape = %s" % y_true.get_shape().as_list())

    # get neural network response
    logits = cnn_lenet(x, BATCH_SIZE, weights, biases, dropout)
    print("logits shape = %s" % logits.get_shape().as_list())
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

    # start the training/testing steps
    if training:
        print("begin training")
        step = 0
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

        # loop until out of data
        while True:
            try:
                train_result = sess.run([train_op, accuracy, x, logits], feed_dict={dropout: DROPOUT})
                if step != 0 and step % 100 == 0:
                    print("step %s, accuracy = %s" % (step, train_result[1]))
                    # save the current model every 1000 steps
                    if step % 1000 == 0:
                        save_model(sess, saver, step)
                # print("x = %s, logits = %s" % (train_result[2], train_result[3]))
                step += 1
            except tf.errors.OutOfRangeError:
                print("data exhausted, saving final model")
                save_model(sess, saver, 'final')
                break
    else:
        print("begin testing")
        step = 0
        saver.restore(sess, LOAD_MODEL)
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

        cumulative_accuracy = 0.0
        # loop until out of data
        while True:
            try:
                test_result = sess.run([accuracy, x, logits], feed_dict={dropout: 1.0})
                cumulative_accuracy += test_result[0]
                if step % 100 == 0:
                    print("step %s, accuracy = %s, cumulative accuracy = %s" %
                          (step, test_result[0], cumulative_accuracy / step / BATCH_SIZE))
                step += 1
            except tf.errors.OutOfRangeError:
                print("data exhausted, test results:")
                print("steps = %s, cumulative accuracy = %.04f" % (step, cumulative_accuracy / step / BATCH_SIZE))
                break


if __name__ == "__main__":
    main()