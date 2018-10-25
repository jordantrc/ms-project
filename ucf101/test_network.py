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
    saver = tf.train.import_meta_graph(latest_model + ".meta")
    saver.restore(sess, latest_model)

    print("Restored model %s" % latest_model)

    test_filenames = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(test_filenames)
    dataset = dataset.map(c3d_model._parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(c3d_model.BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    x, y_true = iterator.get_next()

    # initialize
    # sess.run(init_op)

    # test a single run through of the test data
    sess.run(iterator.initializer, feed_dict={test_filenames: test_files})

    while True:
        try:
            acc = sess.run(accuracy)
            print("test accuracy = %g" % acc)
        except tf.errors.OutOfRangeError:
            break
