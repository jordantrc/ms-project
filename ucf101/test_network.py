# test_network.py
#
# Tests the model checkpoints with the
# test data from test tfrecord files
#

import itertools
import os
import tensorflow as tf
import c3d
import c3d_model
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import metrics


CLASS_LIST = "/home/jordanc/datasets/UCF-101/classInd.txt"
DROPOUT = 1.0
NUM_CLASSES = 101
TEST_DIR = "/home/jordanc/datasets/UCF-101/tfrecords/test"

def tf_confusion_matrix(predictions, labels, classes):
    """
    produces and returns a confusion matrix given the predictions generated by
    tensorflow (in one-hot format), and string labels.
    """
    # print("pred = %s, type = %s, labels = %s, type = %s, classes = %s, type = %s" % (predictions, type(predictions), labels, type(labels), classes, type(classes)))

    y_true = []
    y_pred = []

    for p in predictions:
        pred = p[0]
        y_true.append(classes[pred])

    for l in labels:
        label = l[0]
        y_pred.append(classes[label])

    cm = metrics.confusion_matrix(y_true, y_pred, classes)

    return cm

def plot_confusion_matrix(cm, classes, filename,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() * 0.73
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(filename)
    plt.gcf().clear()
    plt.cla()
    plt.clf()
    plt.close()


# specify a model to load or use the default
if len(sys.argv) == 2:
    model_to_load = sys.argv[1]
else:
    model_dir = "/home/jordanc/datasets/UCF-101/model_ckpts"
    # generate list of model checkpoints, get the latest
    models = os.listdir(model_dir)
    models.sort()
    latest_model = models[-1]
    latest_model = latest_model[:latest_model.index('.ckpt') + len('.ckpt')]
    model_to_load = os.path.join(model_dir, latest_model)

# get the list of class names
class_names = []
with open(CLASS_LIST) as class_fd:
    text = class_fd.read()
    lines = text.split("\n")
    for l in lines:
        if len(l) > 0:
            i, c = l.split(" ")
            class_names.append(c)

assert len(class_names) == NUM_CLASSES

# get the list of test files
test_files = os.listdir(TEST_DIR)
test_files = [os.path.join(TEST_DIR, x) for x in test_files]

with tf.Session() as sess:
    # init variables
    # tf.set_random_seed(1234)
    # weights, biases = c3d.get_variables(c3d_model.NUM_CLASSES)

    # placeholders
    # y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
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

    # y_true_class = tf.argmax(y_true, axis=1)

    # logits = c3d_model.inference_3d(x, c3d_model.DROPOUT, c3d_model.BATCH_SIZE, weights, biases)

    # y_pred = tf.nn.softmax(logits)
    # y_pred_class = tf.argmax(y_pred, axis=1)

    # loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    # optimizer = tf.train.AdamOptimizer(learning_rate=1.0)

    # train_op = optimizer.minimize(loss_op)

    # evaluate the model
    # correct_pred = tf.equal(y_pred_class, y_true_class)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # restore the model
    saver = tf.train.Saver()
    saver.restore(sess, model_to_load)
    print("Restored model %s" % model_to_load)

    # test a single run through of the test data
    sess.run(init_op)
    sess.run(iterator.initializer, feed_dict={test_filenames: test_files})

    i = 0
    cumulative_accuracy = 0.0
    predictions = []
    labels = []
    while True:
        try:
            test_results = sess.run([accuracy, y_pred_class, y_true_class, correct_pred])
            acc = test_results[0]
            y_pred_class_actual = test_results[1]
            y_true_class_actual = test_results[2]
            correct_pred_actual = test_results[3]
            print("[%s] correct = %s, pred/true = [%s/%s], accuracy = %s" % (i, correct_pred_actual, 
                                                                             y_pred_class_actual,
                                                                             y_true_class_actual,
                                                                             acc))

            # add to accumulations
            cumulative_accuracy += float(acc)
            predictions.append(y_pred_class_actual)
            labels.append(y_true_class_actual)

            i += 1
        except tf.errors.OutOfRangeError:
            break
    
    print("Exhausted test data")
    print("Cumulative accuracy = %s" % (cumulative_accuracy / i))
    print("Confusion matrix =")
    cm = tf_confusion_matrix(predictions, labels, class_names)
    plot_confusion_matrix(cm, class_names, "confusion_matrix.jpg")
