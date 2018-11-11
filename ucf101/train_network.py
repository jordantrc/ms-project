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
from tensorflow.python import debug as tf_debug

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import metrics
from c3d_model import C3DModel
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

NUM_EPOCHS = 16
MINI_BATCH_SIZE = 50
BATCH_SIZE = 30
TRAIN_SPLIT = 'train-test-splits/trainlist01.txt'
TEST_SPLIT = 'train-test-splits/testlist01.txt'
VALIDATE_WITH_TRAIN = True
BALANCE_CLASSES = True
SHUFFLE_SIZE = 1000
VARIABLE_TYPE = 'default'

def print_help():
    '''prints a help message'''
    print("Usage:\ntrain_network.py <run name> <sample size as decimal percentage> <classes to include> <model save dir> <tfrecord directory>")

def print_class_counts(file_list):
    '''prints the class counts in the file list'''
    class_counts = {}
    for f in file_list:
        class_name = os.path.basename(f).split('_')[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    keys = list(class_counts.keys())
    keys.sort()
    for k in keys:
        print("%s = %s" % (k, class_counts[k]))


def file_split(list_file, directory):
    '''returns the absolute path to the samples given train-test split file and root directory'''
    file_names = []
    with open(list_file, 'r') as list_file_fd:
        text = list_file_fd.read()
        lines = text.split('\n')
        for l in lines:
            if len(l) > 0:
                _, sample = l.split('/')
                sample = sample.strip()
                if ' ' in sample:
                    sample, _ = sample.split(' ')
                    sample = sample.strip()
                file_names.append(sample)
    # print('file_names = %s...' % file_names[0:5])

    file_paths = []
    file_list = os.listdir(directory)
    # print("file_list = %s..." % file_list[0:5])
    for n in file_names:
        for f in file_list:
            if n in f:
                file_paths.append(os.path.join(directory, f))
    # print("file_paths = %s..." % file_paths[0:5])

    return file_paths


def balance_classes(files):
    '''returns a new list of files with balanced classes'''

    # count each class in the file set
    class_counts = {}
    for f in files:
        class_name = os.path.basename(f).split('_')[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    print("class_counts = %s" % class_counts)

    smallest_class_count = None
    smallest_class = None
    for k, v in class_counts.items():
        if smallest_class is None:
            smallest_class_count = v
            smallest_class = k
        elif v < smallest_class_count:
            smallest_class_count = v
            smallest_class = k

    print("smallest_class = %s, count = %s" % (smallest_class, smallest_class_count))

    balanced_files = []
    for k in class_counts:
        class_files = [x for x in files if k in x]
        # print("class %s has %s class_files = %s" % (k, len(class_files), class_files))
        # sample the files
        class_files = random.sample(class_files, smallest_class_count)
        balanced_files.extend(class_files)

    return balanced_files


def tf_confusion_matrix(predictions, labels, classes):
    """
    returns a confusion matrix given the predictions generated by
    tensorflow (in one-hot format), and string labels.
    """
    # print("pred = %s, type = %s, labels = %s, type = %s, classes = %s, type = %s" % 
    #     (predictions, type(predictions), labels, type(labels), classes, type(classes)))

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

    # np.set_printoptions(threshold='nan')
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


def test_network(sess, test_files, run_name, epoch):
    '''tests the neural network'''
    sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})
    k = 0
    cumulative_accuracy = 0.0
    cumulative_hit_at_5 = 0.0
    predictions = []
    labels = []
    more_data = True
    while more_data:
        try:
            test_results = sess.run([eval_accuracy, y_pred_test_class, y_true_test_class, eval_correct_pred, eval_hit_5, eval_top_5, logits_test])
            acc = test_results[0]
            y_pred_class_actual = test_results[1]
            y_true_class_actual = test_results[2]
            correct_pred_actual = test_results[3]
            hit_5_actual = test_results[4]
            top_5_actual = test_results[5]
            logits_test_out = test_results[6]
            print("test [%s] correct = %s, pred/true = [%s/%s], accuracy = %s, hit@5 = %s, top 5 = %s" %
                                                                            (k,
                                                                             correct_pred_actual, 
                                                                             y_pred_class_actual,
                                                                             y_true_class_actual,
                                                                             acc,
                                                                             hit_5_actual,
                                                                             top_5_actual))

            # add to accumulations
            if hit_5_actual[0]:
                cumulative_hit_at_5 += 1.0
            cumulative_accuracy += float(acc)
            predictions.append(y_pred_class_actual)
            labels.append(y_true_class_actual)

            k += 1
        except tf.errors.OutOfRangeError:
            # print("OutOfRangeError - k = %s" % k)
            more_data = False
            break

    print("Exhausted test data")
    print("Cumulative test accuracy at end of epoch %s = %s" % (epoch, cumulative_accuracy / k))
    print("Cumulative test hit@5 accuracy at end of epoch %s = %s" % (epoch, cumulative_hit_at_5 / k))
    print("Confusion matrix =")
    cm = tf_confusion_matrix(predictions, labels, class_names)
    plot_confusion_matrix(cm, class_names, "runs/%s_confusion_matrix_%s.pdf" % (run_name, epoch))

    return ['test', epoch, "", "", cumulative_accuracy / k, cumulative_hit_at_5 / k]


# get the list of classes
class_names = get_class_list(CLASS_INDEX_FILE)
all_classes = True

# argument collection
run_name = sys.argv[1]
sample = float(sys.argv[2])
included_classes = sys.argv[3]
model_dir = sys.argv[4]
tfrecord_dir = sys.argv[5]

if ',' in included_classes:
    all_classes = False
    included_classes = [x.strip() for x in sys.argv[3].split(',')]
    for c in included_classes:
        assert c in class_names, "%s is not a valid class" % c

elif included_classes.isdigit():
    all_classes = False
    included_classes = random.sample(class_names, int(included_classes))

elif included_classes == "all":
    all_classes = True

else:
    print("Invalid value for class inclusion [%s]" % included_classes)
    sys.exit(1)

print("Beginning run %s using %s sample size and %s classes" % (run_name, sample, included_classes))

# create the model object
model = C3DModel(model_dir=model_dir,tfrecord_dir=tfrecord_dir)

# open the csv data file and write the header to it
run_csv_file = 'runs/%s.csv' % run_name
if sys.version_info[0] == 3:
    run_csv_fd = open(run_csv_file, 'w', newline='', buffering=1)
else:
    run_csv_fd = open(run_csv_file, 'wb', buffering=0)
run_csv_writer = csv.writer(run_csv_fd, dialect='excel')
run_csv_writer.writerow(['step_type', 'epoch', 'iteration', 'loss', 'accuracy', 'hit_at_5'])

# get the list of files for train and test
train_files = file_split(TRAIN_SPLIT, model.tfrecord_dir)
test_files = file_split(TEST_SPLIT, model.tfrecord_dir)

if not all_classes:
    train_files_filtered = []
    test_files_filtered = []
    for c in included_classes:
        # print("c = %s" % c)
        for t in train_files:
            if c in t:
                train_files_filtered.append(t)

        for t in test_files:
            if c in t:
                test_files_filtered.append(t)

    train_files = train_files_filtered
    test_files = test_files_filtered
    # print("train files = %s" % train_files)
    # print("test files = %s" % test_files)
assert len(test_files) > 0 and len(train_files) > 0, 'test = %s, train = %s' % (len(test_files), len(train_files))

# sample from the test and train files if necessary
if sample < 1.0:
    sample_size = int(len(train_files) * sample)
    train_files = random.sample(train_files, sample_size)
    print("%s training samples" % sample_size)

    sample_size_test = int(len(test_files) * sample)
    test_files = random.sample(test_files, sample_size_test)
    print("%s testing samples" % sample_size_test)

assert len(test_files) > 0 and len(train_files) > 0

# reset size of mini-batch based on the number of test files
MINI_BATCH_SIZE = min(MINI_BATCH_SIZE, len(test_files))
SHUFFLE_SIZE = min(SHUFFLE_SIZE, int(len(train_files) * 0.05))

if BALANCE_CLASSES:
    # balance the classes for training
    train_files = balance_classes(train_files)

random.shuffle(train_files)
# throw out samples to fit the batch size
num_samples_batch_fit = len(train_files) - (len(train_files) % BATCH_SIZE)
train_files = random.sample(train_files, num_samples_batch_fit)
print("Training samples = %s, testing samples = %s" % (len(train_files), len(test_files)))
print("Training class counts:")
print_class_counts(train_files)
print("Test class counts:")
print_class_counts(test_files)

# open the log file
run_log_file = 'runs/%s.log' % run_name
run_log_fd = open(run_log_file, 'w', buffering=1)
run_log_fd.write("run name = %s\nsample = %s\nincluded_classes = %s\n" % (run_name, sample, included_classes))
run_log_fd.write("HYPER PARAMETERS:\n")
run_log_fd.write("NUM_EPOCHS = %s\nMINI_BATCH_SIZE = %s\nTRAIN_SPLIT = %s\nTEST_SPLIT = %s\nSHUFFLE_SIZE = %s\n" % 
                (NUM_EPOCHS, MINI_BATCH_SIZE, TRAIN_SPLIT, TEST_SPLIT, SHUFFLE_SIZE))
print("BATCH_SIZE = %s" % (BATCH_SIZE))
run_log_fd.write("VALIDATE_WITH_TRAIN = %s\nBALANCE_CLASSES = %s\n" % (VALIDATE_WITH_TRAIN, BALANCE_CLASSES))
run_log_fd.write("WEIGHT_STDDEV = %s\nBIAS = %s\n" % (c3d.WEIGHT_STDDEV, c3d.BIAS))
run_log_fd.write("WEIGHT_DECAY = %s\nBIAS_DECAY = %s\n" % (c3d.WEIGHT_DECAY, c3d.BIAS_DECAY))
run_log_fd.write("VARIABLE_TYPE = %s\n" % (VARIABLE_TYPE))
run_log_fd.write("Training samples = %s, testing samples = %s\n" % (len(train_files), len(test_files)))

# Tensorflow configuration
config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=config) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # init variables
    # tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if VARIABLE_TYPE == 'default':
        weights, biases = c3d.get_variables(model.num_classes)
    elif VARIABLE_TYPE == 'weight decay':
        weights, biases = c3d.get_variables(model.num_classes, var_type="weight decay")

    # placeholders and constants
    # y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
    train_filenames = tf.placeholder(tf.string, shape=[None])
    test_filenames = tf.placeholder(tf.string, shape=[None])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # constants
    global_step = tf.Variable(0, trainable=False)

    # using tf.data.TFRecordDataset iterator
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(model._parse_function)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(1)
    test_iterator = test_dataset.make_initializable_iterator()
    x_test, y_true_test = test_iterator.get_next()

    # using tf.data.TFRecordDataset iterator
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(model._parse_function)
    train_dataset = train_dataset.repeat(NUM_EPOCHS)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_iterator = train_dataset.make_initializable_iterator()
    x, y_true = train_iterator.get_next()

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    # x = tf.cast(x, tf.float32)
    # print("reshaping x")
    # print("x pre-reshape = %s, shape = %s" % (x, x.get_shape().as_list()))
    # print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))
    x = tf.reshape(x, [BATCH_SIZE, model.frames_per_clip, 112, 112, 3])
    x_test = tf.reshape(x_test, [1, model.frames_per_clip, 112, 112, 3])

    # generate clips for each video in the batch
    # x = model._clip_image_batch(x, model.frames_per_clip, True)
    # x_test = model._clip_image_batch(x_test, model.frames_per_clip, True)

    print("x post-clip = %s, shape = %s" % (x, x.get_shape().as_list()))

    y_true = tf.cast(y_true, tf.int64)
    y_true_class = tf.argmax(y_true, axis=1)
    y_true_test_class = tf.argmax(y_true_test, axis=1)

    logits = model.inference_3d(x, weights, biases, BATCH_SIZE, True)
    # logits = model.c3d(x, training=True)

    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    hit_5 = tf.nn.in_top_k(logits, y_true_class, 5)
    top_5 = tf.nn.top_k(logits, k=5)

    # loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true_class, name="softmax"), name="reduce_mean")
    #learning_rate = tf.train.exponential_decay(learning_rate=model.learning_rate, global_step=global_step, 
    #                                           decay_steps=(4 * len(train_files)), decay_rate=0.96,
    #                                           staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer")
    train_op = optimizer.minimize(loss_op, name="train")

    # model evaluation
    # logits_test = model.c3d(x_test, training=False)
    logits_test = model.inference_3d(x_test, weights, biases, 1, False)
    y_pred_test = tf.nn.softmax(logits_test)
    y_pred_test_class = tf.argmax(y_pred_test, axis=1)

    eval_hit_5 = tf.nn.in_top_k(logits_test, y_true_test_class, 5)
    eval_top_5 = tf.nn.top_k(logits_test, k=5)
    eval_correct_pred = tf.equal(y_pred_test_class, y_true_test_class)
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
    sess.run(init_op)

    # TRAINING
    report_step = min(200, int(len(train_files) * 0.05))
    print("Beginning training epochs, reporting every %s, mini-test batch every %s samples" % (report_step, report_step * 5))

    in_epoch = 1
    print("START EPOCH %s" % in_epoch)
    start = time.time()
    sess.run(train_iterator.initializer, feed_dict={train_filenames: train_files})
    if VALIDATE_WITH_TRAIN:
        sess.run(test_iterator.initializer, feed_dict={test_filenames: train_files})
    else:
        sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})

    j = 0
    train_acc_accum = 0.0
    train_hit5_accum = 0.0
    while True:
        if j != 0 and j % (len(train_files) / BATCH_SIZE) == 0 :
            # end of epoch
            # save a model checkpoint and report end of epoch information
            save_path = os.path.join(model.model_dir, "model_epoch_%s.ckpt" % in_epoch)
            save_path = saver.save(sess, save_path)
            end = time.time()
            train_time = str(datetime.timedelta(seconds=end - start))
            print("END EPOCH %s, iterations = %s, epoch training time: %s" % (in_epoch, j, train_time))
            print("model checkpoint saved to %s\n\n" % save_path)

            # test the network
            test_results = test_network(sess, test_files, run_name, in_epoch)
            run_csv_writer.writerow(test_results)

            in_epoch += 1

            if in_epoch % 4 == 0:
                model.current_learning_rate = model.current_learning_rate / 10
                print("learning rate adjusted to %g" % model.current_learning_rate)

            print("START EPOCH %s" % in_epoch)
            start = time.time()
            if VALIDATE_WITH_TRAIN:
                sess.run(test_iterator.initializer, feed_dict={test_filenames: train_files})
            else:
                sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})

        try:
            train_result = sess.run([train_op, loss_op, accuracy, x, y_true, y_true_class, y_pred, y_pred_class, logits, hit_5], 
                                    feed_dict={learning_rate: model.current_learning_rate})
            loss_op_out = train_result[1]
            train_acc = train_result[2]
            x_actual = train_result[3]
            y_true_actual = train_result[4]
            y_true_class_actual = train_result[5]
            y_pred_actual = train_result[6]
            y_pred_class_actual = train_result[7]
            logits_out = train_result[8]
            hit_5_out = train_result[9]

            train_acc_accum += train_acc
            if hit_5_out[0]:
                train_hit5_accum += 1.0

            # report out results and run a test mini-batch every now and then
            if j != 0 and j % (report_step * BATCH_SIZE) == 0:
                #print("logits = %s" % logits_out)
                #print("x = %s" % x_actual)
                #print("y_true = %s, y_true_class = %s, y_pred = %s, y_pred_class = %s" % (y_true_actual, y_true_class_actual, y_pred_actual, y_pred_class_actual))
                #print("train_acc = %s" % train_acc)
                #print("hit_5_out = %s, type = %s, hit_5_out[0] = %s, type = %s" % (hit_5_out, type(hit_5_out), hit_5_out[0], type(hit_5_out[0])))
                run_time = time.time()
                run_time_str = str(datetime.timedelta(seconds=run_time - start))
                train_step_acc = train_acc_accum / report_step

                # mini batch accuracy - every 5 report step iterations
                if j % (report_step * 5) == 0:
                    mini_batch_acc = 0.0
                    mini_batch_hit5 = 0.0
                    for k in range(MINI_BATCH_SIZE):
                        try:
                            acc, hit_5_out, top_5_out, x_out = sess.run([eval_accuracy, eval_hit_5, eval_top_5, x])
                            # print("type(x) = %s, x = %s" % (type(x_out), x_out))
                            mini_batch_acc += acc
                            if hit_5_out[0]:
                                mini_batch_hit5 += 1.0
                        except tf.errors.OutOfRangeError:
                            # if out of data, just reinitialize the iterator
                            if VALIDATE_WITH_TRAIN:
                                sess.run(test_iterator.initializer, feed_dict={test_filenames: train_files})
                            else:
                                sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})
                    mini_batch_acc = mini_batch_acc / MINI_BATCH_SIZE
                    mini_batch_hit5 = mini_batch_hit5 / MINI_BATCH_SIZE
                    
                    print("\titeration %s - epoch %s run time = %s, loss = %s, mini-batch accuracy = %s, hit@5 = %s, y_true = %s, top 5 = %s" %
                         (j, in_epoch, run_time_str, loss_op_out, mini_batch_acc, mini_batch_hit5, y_true_class_actual, top_5_out))
                    csv_row = ['mini-batch', in_epoch, j, loss_op_out, mini_batch_acc, mini_batch_hit5]
                
                else:
                    train_acc_accum = train_acc_accum / report_step
                    train_hit5_accum = train_hit5_accum / report_step
                    print("\titeration %s - epoch %s run time = %s, loss = %s, train accuracy = %s, hit@5 = %s" %
                         (j, in_epoch, run_time_str, loss_op_out, train_acc_accum, train_hit5_accum))
                    csv_row = ['train', in_epoch, j, loss_op_out, train_acc_accum, train_hit5_accum]

                # write the csv data to 
                run_csv_writer.writerow(csv_row)
                train_acc_accum = 0.0
                train_hit5_accum = 0.0

            j += BATCH_SIZE
        except tf.errors.OutOfRangeError:
            print("Out of range error")
            break

    print("end training epochs")
    # end of epoch
    # save a model checkpoint and report end of epoch information
    save_path = os.path.join(model.model_dir, "model_epoch_%s.ckpt" % in_epoch)
    save_path = saver.save(sess, save_path)
    end = time.time()
    train_time = str(datetime.timedelta(seconds=end - start))
    print("END EPOCH %s, iterations = %s, epoch training time: %s" % (in_epoch, j, train_time))
    print("model checkpoint saved to %s\n\n" % save_path)

    # final test
    test_results = test_network(sess, test_files, run_name, in_epoch)
    run_csv_writer.writerow(test_results)

    coord.request_stop()
    coord.join(threads)
    sess.close()

run_csv_fd.close()
run_log_fd.close()
