# train_network.py
#
# TODO:
# - incorporate tfrecord data input queue
# - fix storage of labels as integers (indexes into the class index file)

import os
import sys
import tensorflow as tf
import c3d_model
import c3d
import csv
import time
import datetime
import random
import itertools
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import metrics

NUM_EPOCHS = 16
MINI_BATCH_SIZE = 50
CLASS_LIST = "/home/jordanc/datasets/UCF-101/classInd.txt"

def print_help():
    '''prints a help message'''
    print("Usage:\ntrain_network.py <run name> <sample size as decimal percentage>")


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
    plt.figure(figsize=(50, 50))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm / cm.astype(np.float).sum(axis=1)
        # replace NaN with 0
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() * 0.73
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, "{0:.4f}".format(cm[i, j]),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(filename)
    plt.gcf().clear()
    plt.cla()
    plt.clf()
    plt.close()

if len(sys.argv) != 3:
    print("Must provide a run name and sample size")
    print_help()
    sys.exit(1)
else:
    run_name = sys.argv[1]
    sample = float(sys.argv[2])
    print("Beginning run %s using %s sample size" % (run_name, sample))


# open the csv data file and write the header to it
run_csv_file = 'runs/%s.csv' % run_name
run_csv_fd = open(run_csv_file, 'wb')
run_csv_writer = csv.writer(run_csv_fd, dialect='excel')
run_csv_writer.writerow(['epoch', 'iteration', 'loss', 'train_accuracy', 'mini_batch_accuracy'])

# get the list of files for train and test
train_files = [os.path.join(c3d_model.TRAIN_DIR, x) for x in os.listdir(c3d_model.TRAIN_DIR)]
random.shuffle(train_files)
test_files = [os.path.join(c3d_model.TEST_DIR, x) for x in os.listdir(c3d_model.TEST_DIR)]
random.shuffle(test_files)

if sample < 1.0:
    sample_size = int(len(train_files) * sample)
    train_files = random.sample(train_files, sample_size)
    print("Sampled %s training samples" % sample_size)

    sample_size_test = int(len(test_files) * sample)
    test_files = random.sample(test_files, sample_size_test)
    print("Sampled %s testing samples" % sample_size_test)

assert len(test_files) > 0 and len(train_files) > 0

# get the list of classes
class_names = []
with open(CLASS_LIST) as class_fd:
    text = class_fd.read()
    lines = text.split("\n")
    for l in lines:
        if len(l) > 0:
            i, c = l.split(" ")
            class_names.append(c)

assert len(class_names) == c3d_model.NUM_CLASSES

current_learning_rate = c3d_model.LEARNING_RATE

with tf.Session() as sess:

    # init variables
    # tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    weights, biases = c3d.get_variables(c3d_model.NUM_CLASSES)

    # placeholders
    # y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
    train_filenames = tf.placeholder(tf.string, shape=[None])
    test_filenames = tf.placeholder(tf.string, shape=[None])

    # using tf.data.TFRecordDataset iterator
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(c3d_model._parse_function)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(c3d_model.BATCH_SIZE)
    test_iterator = test_dataset.make_initializable_iterator()
    x_test, y_true_test = test_iterator.get_next()

    # using tf.data.TFRecordDataset iterator
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(c3d_model._parse_function)
    train_dataset = train_dataset.repeat(NUM_EPOCHS)
    train_dataset = train_dataset.batch(c3d_model.BATCH_SIZE)
    train_iterator = train_dataset.make_initializable_iterator()
    x, y_true = train_iterator.get_next()

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    # x = tf.cast(x, tf.float32)
    # print("reshaping x")
    # print("x pre-reshape = %s, shape = %s" % (x, x.get_shape().as_list()))
    # print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))
    x = tf.reshape(x, [c3d_model.BATCH_SIZE, c3d_model.FRAMES_PER_VIDEO, 112, 112, 3])
    x_test = tf.reshape(x_test, [c3d_model.BATCH_SIZE, c3d_model.FRAMES_PER_VIDEO, 112, 112, 3])

    # generate clips for each video in the batch
    x = c3d_model._clip_image_batch(x, c3d_model.FRAMES_PER_CLIP, True)
    x_test = c3d_model._clip_image_batch(x_test, c3d_model.FRAMES_PER_CLIP, True)

    print("x post-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

    # placeholders
    # x = tf.placeholder(tf.uint8, shape=[None, num_features], name='x')
    y_true = tf.cast(y_true, tf.int64)
    y_true_class = tf.argmax(y_true, axis=1)
    y_true_test_class = tf.argmax(y_true_test, axis=1)

    logits = c3d_model.inference_3d(x, c3d_model.DROPOUT, c3d_model.BATCH_SIZE, weights, biases)

    y_pred = tf.nn.softmax(logits)
    y_pred = tf.cast(y_pred, tf.int64)
    y_pred_class = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=current_learning_rate)

    train_op = optimizer.minimize(loss_op)

    # model evaluation
    logits_test = c3d_model.inference_3d(x_test, 0.5, c3d_model.BATCH_SIZE, weights, biases)
    y_pred_test = tf.nn.softmax(logits_test)
    y_pred_test = tf.cast(y_pred_test, tf.int64)
    y_pred_test_class = tf.argmax(y_pred_test, axis=1)

    eval_correct_pred = tf.equal(y_pred_test_class, y_true_test_class)
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
    sess.run(init_op)

    print("Beginning training epochs")

    for i in range(NUM_EPOCHS):
        print("START EPOCH %s" % i)
        start = time.time()
        sess.run(train_iterator.initializer, feed_dict={train_filenames: train_files})
        sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})

        j = 0
        report_step = 200
        train_acc_accum = 0.0
        while True:
            try:
                # _, y_true, y_pred = sess.run([train_op, y_true, y_pred])
                # print("y_true = %s" % y_true)
                # print("y_pred = %s" % y_pred)
                _, loss_op_out, train_acc, logits_out = sess.run([train_op, loss_op, accuracy, logits])
                print("logits = %s" % logits_out)
                train_acc_accum += train_acc

                # report out results and run a test mini-batch every now and then
                if j != 0 and j % report_step == 0:
                    run_time = time.time()
                    run_time_str = str(datetime.timedelta(seconds=run_time - start))
                    train_step_acc = train_acc_accum / report_step

                    # mini batch accuracy - every 1000 iterations
                    if j % 1000 == 0:
                        mini_batch_acc = 0.0
                        for k in range(MINI_BATCH_SIZE):
                            acc = sess.run(eval_accuracy)
                            mini_batch_acc += acc
                        mini_batch_acc = mini_batch_acc / MINI_BATCH_SIZE
                        
                        print("\titeration %s - epoch %s run time = %s, loss = %s, train accuracy= %s,  mini-batch accuracy = %s" % (j, i, run_time_str, loss_op_out, train_step_acc, mini_batch_acc))
                        csv_row = [i, j, loss_op_out, train_step_acc, mini_batch_acc]
                    
                    else:
                        print("\titeration %s - epoch %s run time = %s, loss = %s, train accuracy = %s" % (j, i, run_time_str, loss_op_out, train_step_acc))
                        csv_row = [i, j, loss_op_out, train_step_acc, ""]

                    # write the csv data to 
                    run_csv_writer.writerow(csv_row)
                    train_acc_accum = 0.0

                j += 1
            except tf.errors.OutOfRangeError:
                break

        # save a model checkpoint and report end of epoch information
        save_path = os.path.join(c3d_model.MODEL_DIR, "model_epoch_%s.ckpt" % i)
        save_path = saver.save(sess, save_path)
        end = time.time()
        train_time = str(datetime.timedelta(seconds=end - start))
        print("END EPOCH %s, epoch training time: %s" % (i, train_time))
        print("model checkpoint saved to %s\n\n" % save_path)

        # test accuracy, save a confusion matrix
        sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})
        k = 0
        cumulative_accuracy = 0.0
        predictions = []
        labels = []
        while True:
            try:
                test_results = sess.run([eval_accuracy, y_pred_test_class, y_true_test_class, eval_correct_pred])
                acc = test_results[0]
                y_pred_class_actual = test_results[1]
                y_true_class_actual = test_results[2]
                correct_pred_actual = test_results[3]
                print("test [%s] correct = %s, pred/true = [%s/%s], accuracy = %s" % (k, correct_pred_actual, 
                                                                                 y_pred_class_actual,
                                                                                 y_true_class_actual,
                                                                                 acc))

                # add to accumulations
                cumulative_accuracy += float(acc)
                predictions.append(y_pred_class_actual)
                labels.append(y_true_class_actual)

                k += 1
            except tf.errors.OutOfRangeError:
                print("OutOfRangeError - k = %s" % k)
                break
    
        print("Exhausted test data")
        print("Cumulative accuracy at end of epoch %s = %s" % (i, cumulative_accuracy / k))
        print("Confusion matrix =")
        cm = tf_confusion_matrix(predictions, labels, class_names)
        plot_confusion_matrix(cm, class_names, "runs/%s_confusion_matrix_%s.jpg" % (run_name, i))

        if i != 0 and i % 4 == 0:
            current_learning_rate = current_learning_rate / 10
            print("learning rate adjusted to %g" % current_learning_rate)

    print("end training epochs")

    coord.request_stop()
    coord.join(threads)
    sess.close()

run_csv_fd.close()
