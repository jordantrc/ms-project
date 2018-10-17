# train_network.py
#
# TODO:
# - incorporate tfrecord data input queue
# - fix storage of labels as integers (indexes into the class index file)

import tensorflow as tf
import numpy as np
import random
import c3d_model
import c3d

NUM_CLASSES = 101
TRAIN_FILE = "/home/jordanc/datasets/UCF-101/tfrecords/train.tfrecord"
TEST_FILE = "/home/jordanc/datasets/UCF-101/tfrecords/test.tfrecord"
DROPOUT = 0.5
FRAMES_PER_VIDEO = 250
FRAMES_PER_CLIP = 16
BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3


def _clip_image(image, num_frames, randomly=True):
    '''clips an image'''

    first_d = image.get_shape().as_list()[0]
    indexes = range(first_d)

    if randomly:
        sampled_indexes = random.sample(indexes, num_frames)
        sampled_indexes.sort()
    else:
        sampled_indexes = indexes

    clip = None
    for i in sampled_indexes:
        if clip is None:
            clip = image[i]
        else:
            clip = tf.stack([clip, image[i]])
    print("clip = %s" % clip)

    return clip


def _parse_function(example_proto):
    """parse map function for video data"""
    features = {"label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                # "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                # "width": tf.FixedLenFeature((), tf.int64, default_value=0)
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    image = parsed_features['img_raw']
    print("image = %s, shape = %s" % (image, image.get_shape().as_list()))
    image = tf.cast(image, tf.float32)
    print("image = %s, shape = %s" % (image, image.get_shape().as_list()))
    image = tf.reshape(parsed_features['img_raw'],
                       [
                       c3d.INPUT_DATA_SIZE['t'],  # frames per sample
                       c3d.INPUT_DATA_SIZE['h'],
                       c3d.INPUT_DATA_SIZE['w'],
                       c3d.INPUT_DATA_SIZE['c']
                       ])
    # sample 16 random frames from the stack of frames, maintain temporal
    # order
    # image = _clip_image(image, FRAMES_PER_CLIP, True)

    return image, parsed_features["label"]


with tf.Session() as sess:

    # repeatable randomness during development
    tf.set_random_seed(1234)

    # placeholders
    y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')

    # using tf.data.TFRecordDataset iterator
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    x, y_true = iterator.get_next()

    # convert x to float, reshape to 5d
    #x = tf.cast(x, tf.float32)
    #x_5d = tf.reshape(x, [BATCH_SIZE,
    #                      c3d.INPUT_DATA_SIZE['t'],  # frames per sample
    #                      c3d.INPUT_DATA_SIZE['h'],
    #                      c3d.INPUT_DATA_SIZE['w'],
    #                      c3d.INPUT_DATA_SIZE['c']
    #                      ])
    print("x = %s" % x)

    # init variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # placeholders
    # x = tf.placeholder(tf.uint8, shape=[None, num_features], name='x')
    y_true_class = tf.argmax(y_true, axis=1)

    weights, biases = c3d.get_variables(NUM_CLASSES)
    logits = c3d_model.inference_3d(x, DROPOUT, BATCH_SIZE, weights, biases)

    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, axis=1)

    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entroy_with_logits(logits=logits, labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    train_op = optimizer.minimize(loss_op)

    # evaluate the model
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess.run(init_op)

    print("Beginning training epochs")

    for i in range(NUM_EPOCHS):
        sess.run(iterator.initializer)
        sess.eval(x)
        sess.run(train_op)

    print("end training epochs")
