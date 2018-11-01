# train_network.py
#
# TODO:
# - incorporate tfrecord data input queue
# - fix storage of labels as integers (indexes into the class index file)

import os
import sys
import tensorflow as tf
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
from c3d_model import C3DModel

NUM_EPOCHS = 16
MINI_BATCH_SIZE = 50
CLASS_LIST = "/home/jordanc/datasets/UCF-101/classInd.txt"

def print_help():
    '''prints a help message'''
    print("Usage:\ntrain_network.py <run name> <sample size as decimal percentage> <classes to include>")


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
    if cm.shape[0] < 10 and cm.shape[1] < 10:
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


all_classes = True

if len(sys.argv) != 4:
    print("Must provide a run name, sample size, and a list of classes to include (or 'all' for all classes)")
    print_help()
    sys.exit(1)
else:
    run_name = sys.argv[1]
    sample = float(sys.argv[2])
    included_classes = sys.argv[3]

    if ',' in included_classes:
        all_classes = False
        included_classes = [x.strip() for x in sys.argv[3].split(',')]
        num_classes_actual = len(included_classes)
    elif included_classes != 'all':
        print("Invalid value for class inclusion [%s]" % included_classes)
        sys.exit(1)

    print("Beginning run %s using %s sample size and %s classes" % (run_name, sample, num_classes_actual))

print("Using classes definition %s" % included_classes)

# get the list of classes
class_names = []
num_classes = 0
num_classes_actual = 0
with open(CLASS_LIST) as class_fd:
    text = class_fd.read()
    lines = text.split("\n")
    for l in lines:
        if len(l) > 0:
            i, c = l.split(" ")
            num_classes += 1
            if all_classes or c in included_classes:
                num_classes_actual += 1
                class_names.append(c)   
assert len(class_names) == num_classes_actual and num_classes_actual > 0

# create the model object
model = C3DModel(num_classes=num_classes_actual)

# open the csv data file and write the header to it
run_csv_file = 'runs/%s.csv' % run_name
run_csv_fd = open(run_csv_file, 'wb')
run_csv_writer = csv.writer(run_csv_fd, dialect='excel')
run_csv_writer.writerow(['epoch', 'iteration', 'loss', 'train_accuracy', 'mini_batch_accuracy'])

# open the log file
run_log_file = 'runs/%s.log' % run_name
run_log_fd = open(run_log_file, 'w')

# get the list of files for train and test
train_files = [os.path.join(model.train_dir, x) for x in os.listdir(model.train_dir)]
test_files = [os.path.join(model.test_dir, x) for x in os.listdir(model.test_dir)]

print("train_files = %s" % train_files)
if not all_classes:
    train_files_filtered = []
    test_files_filtered = []
    for c in included_classes:
        print("c = %s" % c)
        for t in train_files:
            print("t = %s" % t)
            print("c in t = %s" % (c in t))
            if c in t:
                train_files_filtered.append(t)

        for t in test_files:
            if c in t:
                test_files_filtered.append(t)

    train_files = train_files_filtered
    test_files = test_files_filtered
assert len(test_files) > 0 and len(train_files) > 0

random.shuffle(train_files)

# sample from the test and train files if necessary
if sample < 1.0:
    sample_size = int(len(train_files) * sample)
    train_files = random.sample(train_files, sample_size)
    print("Sampled %s training samples" % sample_size)

    sample_size_test = int(len(test_files) * sample)
    test_files = random.sample(test_files, sample_size_test)
    print("Sampled %s testing samples" % sample_size_test)

assert len(test_files) > 0 and len(train_files) > 0

print("Training samples = %s, testing samples = %s" % (len(train_files), len(test_files)))

with tf.Session() as sess:

    # init variables
    # tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    weights, biases = c3d.get_variables(model.num_classes)

    # placeholders
    # y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
    train_filenames = tf.placeholder(tf.string, shape=[None])
    test_filenames = tf.placeholder(tf.string, shape=[None])

    # using tf.data.TFRecordDataset iterator
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(model._parse_function)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(model.batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    x_test, y_true_test = test_iterator.get_next()

    # using tf.data.TFRecordDataset iterator
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(model._parse_function)
    train_dataset = train_dataset.repeat(NUM_EPOCHS)
    train_dataset = train_dataset.batch(model.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    x, y_true = train_iterator.get_next()

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    # x = tf.cast(x, tf.float32)
    # print("reshaping x")
    # print("x pre-reshape = %s, shape = %s" % (x, x.get_shape().as_list()))
    # print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))
    x = tf.reshape(x, [model.batch_size, model.frames_per_video, 112, 112, 3])
    x_test = tf.reshape(x_test, [model.batch_size, model.frames_per_video, 112, 112, 3])

    # generate clips for each video in the batch
    x = model._clip_image_batch(x, model.frames_per_clip, True)
    x_test = model._clip_image_batch(x_test, model.frames_per_clip, True)

    print("x post-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

    # placeholders
    # x = tf.placeholder(tf.uint8, shape=[None, num_features], name='x')
    y_true = tf.cast(y_true, tf.int64)
    y_true_class = tf.argmax(y_true, axis=1)
    y_true_test_class = tf.argmax(y_true_test, axis=1)

    logits = model.inference_3d(x, model.dropout, model.batch_size, weights, biases)

    y_pred = tf.nn.softmax(logits)
    y_pred = tf.cast(y_pred, tf.int64)
    y_pred_class = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true_class))
    optimizer = tf.train.AdamOptimizer(learning_rate=model.current_learning_rate)

    train_op = optimizer.minimize(loss_op)

    # model evaluation
    logits_test = model.inference_3d(x_test, 0.5, model.batch_size, weights, biases)
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
                train_result = sess.run([train_op, loss_op, accuracy, x, y_true_class, y_pred_class, logits])
                loss_op_out = train_result[1]
                train_acc = train_result[2]
                x_actual = train_result[3]
                y_true_actual = train_result[4]
                y_pred_actual = train_result[5]
                logits_out = train_results[6]

                train_acc_accum += train_acc

                # report out results and run a test mini-batch every now and then
                if j != 0 and j % report_step == 0:
                    print("logits = %s" % logits_out)
                    print("x = %s" % x_actual)
                    print("y_true = %s" % y_true_actual)
                    print("y_pred = %s" % y_pred_actual)
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
                        
                        print("\titeration %s - epoch %s run time = %s, loss = %s, train accuracy = %s,  mini-batch accuracy = %s" % (j, i, run_time_str, loss_op_out, train_step_acc, mini_batch_acc))
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
        save_path = os.path.join(model.model_dir, "model_epoch_%s.ckpt" % i)
        save_path = saver.save(sess, save_path)
        end = time.time()
        train_time = str(datetime.timedelta(seconds=end - start))
        print("END EPOCH %s, iterations = %s, epoch training time: %s" % (i, j, train_time))
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
