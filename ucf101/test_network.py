# test_network.py
#
# Tests the model checkpoints with the
# test data from test tfrecord files
#

import os
import tensorflow as tf
import c3d
import c3d_model
import sys

DROPOUT = 1.0
NUM_CLASSES = 101
TEST_DIR = "/home/jordanc/datasets/UCF-101/tfrecords/test"

# specify a model directory or use the default
if len(sys.argv) == 2:
    MODEL_DIR = sys.argv[1]
    assert os.path.isdir(MODEL_DIR), "%s is not a directory" % MODEL_DIR
else:
    MODEL_DIR = "/home/jordanc/datasets/UCF-101/model_ckpts"

test_files = os.listdir(TEST_DIR)
test_files = [os.path.join(TEST_DIR, x) for x in test_files]

# generate list of model checkpoints, get the latest
models = os.listdir(MODEL_DIR)
models.sort()
latest_model = models[-1]
latest_model = latest_model[:latest_model.index('.ckpt') + len('.ckpt')]
latest_model = os.path.join(MODEL_DIR, latest_model)
print("latest_model = %s" % latest_model)

with tf.Session() as sess:
    
    # init variables
    tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    weights, biases = c3d.get_variables(c3d_model.NUM_CLASSES)

    # placeholders
    # y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
    train_filenames = tf.placeholder(tf.string, shape=[None])
    test_filenames = tf.placeholder(tf.string, shape=[None])

    # using tf.data.TFRecordDataset iterator
    dataset = tf.data.TFRecordDataset(test_filenames)
    dataset = dataset.map(c3d_model._parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(c3d_model.BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    x, y_true = iterator.get_next()

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    # x = tf.cast(x, tf.float32)
    # print("reshaping x")
    # print("x pre-reshape = %s, shape = %s" % (x, x.get_shape().as_list()))
    # print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))
    x = tf.reshape(x, [c3d_model.BATCH_SIZE, c3d_model.FRAMES_PER_VIDEO, 112, 112, 3])

    # generate clips for each video in the batch
    x = c3d_model._clip_image_batch(x, c3d_model.FRAMES_PER_CLIP, True)

    print("x post-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

    # placeholders
    # x = tf.placeholder(tf.uint8, shape=[None, num_features], name='x')
    y_true_class = tf.argmax(y_true, axis=1)

    logits = c3d_model.inference_3d(x, c3d_model.DROPOUT, c3d_model.BATCH_SIZE, weights, biases)

    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, axis=1)

    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=1.0)

    train_op = optimizer.minimize(loss_op)

    # evaluate the model
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # restore the model
    saver = tf.train.Saver()
    saver.restore(sess, latest_model)
    print("Restored model %s" % latest_model)

    # test a single run through of the test data
    sess.run(init_op)
    sess.run(iterator.initializer, feed_dict={test_filenames: test_files})

    i = 0
    cumulative_accuracy = 0.0
    while True:
        try:
            test_results = sess.run([accuracy, y_pred_class, y_true_class, correct_pred])
            acc = test_results[0]
            y_pred_class_actual = test_results[1]
            y_true_class_actual = test_results[2]
            correct_pred_actual = test_results[3]
            cumulative_accuracy += float(acc)
            print("[%s] correct = %s, pred/true = [%s/%s], accuracy = %s" % (i, correct_pred_actual, 
                                                                             y_pred_class_actual,
                                                                             y_true_class_actual,
                                                                             acc))
            i += 1
        except tf.errors.OutOfRangeError:
            break

    print("Cumulative accuracy = %s" % (cumulative_accuracy / i))
    print("Exhausted test data")
