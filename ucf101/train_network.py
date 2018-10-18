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


def _clip_image_batch(image_batch, num_frames, randomly=True):
    '''generates a clip for each video in the batch'''

    dimensions = image_batch.get_shape().as_list()
    batch_size = dimensions[0]
    num_frames_in_video = dimensions[1]
    height = dimensions[2]
    width = dimensions[3]
    channels = dimensions[4]

    # sample frames for each video
    clip = None
    clip_batch = None
    for i in range(batch_size):
        video = image_batch[i]
        print("video = %s, shape = %s" % (video, video.get_shape().as_list()))
        # randomly sample frames
        sample_indexes = random.sample(list(range(num_frames_in_video)), num_frames)
        sample_indexes.sort()

        print("sample indexes = %s" % sample_indexes)

        for j in sample_indexes:
            if clip is None:
                clip = video[j]
                clip = tf.reshape(clip, [1, height, width, channels])
                print("clip (init) = %s, shape = %s" % (clip, clip.get_shape().as_list()))
            else:
                # video_4d = tf.reshape(video[j], [1, height, width, channels])
                clip = tf.stack([clip, tf.reshape(video[j], [1, height, width, channels])])
                print("clip (stack) = %s, shape = %s" % (clip, clip.get_shape().as_list()))

        # clip is finished, add to the clip_batch
        if clip_batch is None:
            clip_batch = clip
        else:
            clip_batch = tf.stack([clip_batch, clip])
        print("clip_batch = %s, shape = %s" % (clip_batch, clip_batch.get_shape().as_list()))

    return clip_batch


def _parse_function(example_proto):
    """parse map function for video data"""
    features = {"label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                # "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                # "width": tf.FixedLenFeature((), tf.int64, default_value=0)
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features['img_raw'], parsed_features['label']


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

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    x = tf.cast(x, tf.float32)
    print("reshaping x")
    x = tf.reshape(x,
                   [BATCH_SIZE,
                    c3d.INPUT_DATA_SIZE['t'],  # frames per sample
                    c3d.INPUT_DATA_SIZE['h'],
                    c3d.INPUT_DATA_SIZE['w'],
                    c3d.INPUT_DATA_SIZE['c']
                    ])
    print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

    # generate clips for each video in the batch
    print("generating clips from x")
    x = _clip_image_batch(x, FRAMES_PER_CLIP, True)

    print("x post-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

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
