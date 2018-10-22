# test_network.py
#
# Tests the model checkpoints with the
# test data from test tfrecord files
#

import os
import tensorflow as tf
import c3d
import c3d_model

DROPOUT = 0.0
NUM_CLASSES = 101
TEST_DIR = "/home/jordanc/datasets/UCF-101/tfrecords/test"
MODEL_DIR = "/home/jordanc/datasets/UCF-101/model_ckpts"

tf.reset_default_graph()

test_files = os.listdir(TEST_DIR)

# generate list of model checkpoints, get the latest
models = os.listdir(MODEL_DIR)
models.sort()
latest_model = models[-1]
latest_model = latest_model[:latest_model.index('.ckpt') + len('.ckpt')]

with tf.Session() as sess:
    # variables
    weights, biases = c3d.get_variables(NUM_CLASSES)

    test_filenames = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(test_filenames)
    dataset = dataset.map(c3d_model._parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(c3d_model.BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    x, y_true = iterator.get_next()

    y_true_class = tf.argmax(y_true, axis=1)

    logits = c3d_model.inference_3d(x, DROPOUT, c3d_model.BATCH_SIZE, weights, biases)

    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, axis=1)

    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=c3d_model.LEARNING_RATE)

    train_op = optimizer.minimize(loss_op)

    # evaluate the model
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # load the last model checkpoint
    saver = tf.train.Saver()
    saver.restore(latest_model)
    print("Restored model %s" % latest_model)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # test a single run through of the test data
    sess.run(iterator.initializer, feed_dict={test_filenames: test_files})

    acc = sess.run(accuracy)
    print("test accuracy = %g" % acc)
