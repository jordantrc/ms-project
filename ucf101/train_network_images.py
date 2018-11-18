# train_network.py
#
# TODO:
# - incorporate tfrecord data input queue
# - fix storage of labels as integers (indexes into the class index file)

import os
import sys
import tensorflow as tf
import collections
import c3d
import csv
import cv2
import time
import datetime
import random
import itertools
import numpy as np
import matplotlib
from tensorflow.python import debug as tf_debug
from itertools import cycle

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn import metrics
from c3d_model import C3DModel
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

NUM_EPOCHS = 16
MINI_BATCH_SIZE = 50
BATCH_SIZE = 15
TRAIN_SPLIT = 'train-test-splits/train.list'
TEST_SPLIT = 'train-test-splits/test.list'
VALIDATE_WITH_TRAIN = True
BALANCE_CLASSES = True
SHUFFLE_SIZE = 1000
VARIABLE_TYPE = 'default'
ONE_CLIP_PER_VIDEO = False
LEARNING_RATE_DECAY = 0.1
OPTIMIZER = 'Adam'

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


def file_split(list_file, directory, one_clip_per_vid):
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

    if one_clip_per_vid:
        print("file_list length pre clip sample = %s" % len(file_list))
        new_file_list = []
        # make a list of distinct videos
        videos = set()
        for f in file_list:
            video, _ = f.split(".avi")
            videos.add(video)
        # now sample one clip per video
        for v in videos:
            video_list = [x for x in file_list if v in x]
            clip_sample = random.sample(video_list, 1)
            new_file_list.append(clip_sample[0])

        file_list = new_file_list
        print("file_list length post clip sample = %s" % len(file_list))

    print("file_list = %s..." % file_list[0:5])
    print("file_names = %s..." % file_names[0:5])
    for n in file_names:
        for f in file_list:
            if n in f:
                file_paths.append(os.path.join(directory, f))
    print("file_paths = %s..." % file_paths[0:5])

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


def get_frames(directory, frames_per_clip):
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(directory):
        if len(filenames) < frames_per_clip:
            return [], s_index
        filenames = sorted(filenames)

        s_index = random.randint(0, len(filenames) - frames_per_clip)
        for i in range(s_index, s_index + frames_per_clip):
            image_name = os.path.join(directory, filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)

    ret_arr = np.stack(ret_arr)
    return ret_arr, s_index


def get_image_batch(filename, batch_size, frames_per_clip, num_classes, offset=-1, crop_size=112, shuffle=True):
    '''retrieves a batch of images'''
    # open the file containing the list of clips from which to sample
    data = []
    labels = []
    lines = []
    s_index = 0
    with open(filename, 'r') as file_list:
        text = file_list.read()
        lines = text.split('\n')

    video_indices = range(len(lines))
    if shuffle:
        random.shuffle(video_indices)

    if offset >= 0:
        video_indices = cycle(video_indices)
        s_index = offset
    
    batch_index = 0
    for index in video_indices[s_index:]:
        if batch_index >= batch_size:
            next_batch_start = index
            break
        #print("[get_image_batch] line = %s" % lines[index])
        dirname, label = lines[index].split()
        label = int(label)
        frames, _ = get_frames(dirname, frames_per_clip)
        #print("[get_image_batch] len. frames = %s, shape = %s" % (len(frames), np.shape(frames)))

        # process the images
        if len(frames) > 0:
            images = []
            for j in xrange(len(frames)):
                img = Image.fromarray(frames[j].astype(np.uint8))
                img = np.array(cv2.resize(np.array(img), (crop_size, crop_size))).astype(np.float32)
                # if img.width > img.height:
                #     scale = float(crop_size) / float(img.height)
                #     img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
                # else:
                #     scale = float(crop_size) / float(img.width)
                #     img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
                # img = tf.reshape(img, [1, crop_size, crop_size, 3])
                # img = tf.image.per_image_standardization(img)
                images.append(img)
            # print("[get_image_batch] images shape = %s, type = %s, elem shape = %s, type = %s" %
            #        (np.shape(images), type(images), np.shape(images[0]), type(images[0])))
            data.append(images)
            labels.append(label)
            batch_index += 1

    #print("[get_image_batch] final data = %s, elem shape = %s" % (np.shape(data), np.shape(data[0])))
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        j = 0
        for i in range(pad_len):
            data.append(data[j])
            labels.append(labels[j])
            j += 1

    np_arr_label = np.array(labels).astype(np.int64)
    np_arr_label = np.reshape(np_arr_label, (batch_size))
    data = np.array(data).astype(np.float32)
    data = np.reshape(data, (batch_size, frames_per_clip, crop_size, crop_size, 3))
    #print("[get_image_batch] data = %s, np_arr_label = %s" % (np.shape(data), np.shape(np_arr_label)))
    #print("[get_image_batch] np_arr_label = %s" % (np_arr_label))
    return data, np_arr_label, next_batch_start, len(lines)


def tf_confusion_matrix(predictions, labels, classes):
    """
    returns a confusion matrix given the predictions generated by
    tensorflow (in one-hot format), and string labels.
    """
    print("pred = %s, type = %s\nlabels = %s, type = %s\nclasses = %s, type = %s" % 
         (predictions[0:5], type(predictions), labels[0:5], type(labels), classes[0:5], type(classes)))

    y_true = []
    y_pred = []

    for p in predictions:
        pred = p[0]
        y_pred.append(classes[pred])

    for l in labels:
        label = l[0]
        y_true.append(classes[label])

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


def test_network(sess, model, test_file_name, run_name, epoch):
    '''tests the neural network'''
    # sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})
    k = 0
    report_interval = max(1, int(len(test_files) / 100))
    cumulative_accuracy = 0.0
    cumulative_hit_at_5 = 0.0
    predictions = []
    labels = []
    offset = 0
    more_data = True
    start = time.time()
    while more_data:
        x_feed, y_feed, offset, num_samples = get_image_batch(test_file_name, 1, model.frames_per_clip, model.num_classes, offset=offset, shuffle=False)
        test_results = sess.run([eval_accuracy, y_pred_test_class, y_true_test_class, eval_correct_pred, eval_hit_5, eval_top_5, logits_test],
                                feed_dict={x_test: x_feed, y_true_test: y_feed})
        acc = test_results[0]
        y_pred_class_actual = test_results[1]
        y_true_class_actual = test_results[2]
        correct_pred_actual = test_results[3]
        hit_5_actual = test_results[4]
        top_5_actual = test_results[5]
        logits_test_out = test_results[6]
        if k != 0 and k % report_interval == 0:
            print("test [%s] correct = %s, pred/true = [%s/%s], accuracy = %s, hit@5 = %s, top 5 = %s, cum. accuracy = %s" %
                                                                            (k,
                                                                             correct_pred_actual, 
                                                                             y_pred_class_actual,
                                                                             y_true_class_actual,
                                                                             acc,
                                                                             hit_5_actual,
                                                                             top_5_actual,
                                                                             cumulative_accuracy / k))

        # add to accumulations
        if hit_5_actual[0]:
            cumulative_hit_at_5 += 1.0
        cumulative_accuracy += float(acc)
        predictions.append(y_pred_class_actual)
        labels.append(y_true_class_actual)
        k += 1
        if k > num_samples:
            more_data = False

    end = time.time()
    test_time = str(datetime.timedelta(seconds=end - start))
    
    print("Exhausted test data, time: %s" % (test_time))
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
#tfrecord_dir = sys.argv[5]

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
model = C3DModel(model_dir=model_dir)

# open the csv data file and write the header to it
run_csv_file = 'runs/%s.csv' % run_name
if sys.version_info[0] == 3:
    run_csv_fd = open(run_csv_file, 'w', newline='', buffering=1)
else:
    run_csv_fd = open(run_csv_file, 'wb', buffering=0)
run_csv_writer = csv.writer(run_csv_fd, dialect='excel')
run_csv_writer.writerow(['step_type', 'epoch', 'iteration', 'loss', 'accuracy', 'hit_at_5'])

# get the list of files for train and test
#train_files = file_split(TRAIN_SPLIT, model.tfrecord_dir, ONE_CLIP_PER_VIDEO)
#test_files = file_split(TEST_SPLIT, model.tfrecord_dir, ONE_CLIP_PER_VIDEO)

#if not all_classes:
#    train_files_filtered = []
#    test_files_filtered = []
#     for c in included_classes:
#         # print("c = %s" % c)
#         for t in train_files:
#             if c in t:
#                 train_files_filtered.append(t)

#         for t in test_files:
#             if c in t:
#                 test_files_filtered.append(t)

#     train_files = train_files_filtered
#     test_files = test_files_filtered
#     # print("train files = %s" % train_files)
#     # print("test files = %s" % test_files)
# assert len(test_files) > 0 and len(train_files) > 0, 'test = %s, train = %s' % (len(test_files), len(train_files))

# # sample from the test and train files if necessary
# if sample < 1.0:
#     sample_size = int(len(train_files) * sample)
#     train_files = random.sample(train_files, sample_size)
#     print("%s training samples" % sample_size)

#     sample_size_test = int(len(test_files) * sample)
#     test_files = random.sample(test_files, sample_size_test)
#     print("%s testing samples" % sample_size_test)

# assert len(test_files) > 0 and len(train_files) > 0

# # reset size of mini-batch based on the number of test files
# MINI_BATCH_SIZE = min(MINI_BATCH_SIZE, len(test_files))
# SHUFFLE_SIZE = min(SHUFFLE_SIZE, int(len(train_files) * 0.05))

# # throw out samples to fit the batch size
# num_samples_batch_fit = len(train_files) - (len(train_files) % BATCH_SIZE)

# if BALANCE_CLASSES:
#     # balance the classes for training
#     train_files = balance_classes(train_files)

# random.shuffle(train_files)
# train_files = train_files[0: num_samples_batch_fit]
# print("Training samples = %s, testing samples = %s" % (len(train_files), len(test_files)))
# print("Training class counts:")
# print_class_counts(train_files)
# print("Test class counts:")
# print_class_counts(test_files)

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
run_log_fd.write("ONE_CLIP_PER_VIDEO = %s" % (ONE_CLIP_PER_VIDEO))
run_log_fd.write("LEARNING_RATE_DECAY = %s" % (LEARNING_RATE_DECAY))
run_log_fd.write("OPTIMIZER = %s" % (OPTIMIZER))
# run_log_fd.write("Training samples = %s, testing samples = %s\n" % (len(train_files), len(test_files)))

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
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, model.frames_per_clip, 112, 112, 3])
    y_true = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    x_test = tf.placeholder(tf.float32, shape=[1, model.frames_per_clip, 112, 112, 3])
    y_true_test = tf.placeholder(tf.int64, shape=[1])

    y_true_one_hot = tf.one_hot(y_true, depth=model.num_classes)
    y_true_test_one_hot = tf.one_hot(y_true_test, depth=model.num_classes)
    y_true_class = tf.argmax(y_true_one_hot, axis=1)
    y_true_test_class = tf.argmax(y_true_test_one_hot, axis=1)

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
    if OPTIMIZER == 'SGD':
        learning_rate = tf.placeholder(tf.float32, shape=[])
        #learning_rate = tf.train.exponential_decay(learning_rate=model.learning_rate, global_step=global_step, 
        #                                       decay_steps=(4 * len(train_files)), decay_rate=0.96,
        #                                       staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer")
    elif OPTIMIZER == 'Adam':
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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
    report_step = 20
    print("Beginning training epochs, reporting every %s batches, mini-test batch every %s batches" % (report_step, report_step * 5))

    in_epoch = 1
    print("START EPOCH %s" % in_epoch)
    start = time.time()
    # sess.run(train_iterator.initializer, feed_dict={train_filenames: train_files})
    # if VALIDATE_WITH_TRAIN:
    #     sess.run(test_iterator.initializer, feed_dict={test_filenames: train_files})
    # else:
    #     sess.run(test_iterator.initializer, feed_dict={test_filenames: test_files})

    j = 0
    train_acc_accum = 0.0
    train_hit5_accum = 0.0
    # num_batches_per_epoch = len(train_files) / BATCH_SIZE
    while True and in_epoch <= NUM_EPOCHS:
        x_feed, y_feed, _, num_samples = get_image_batch(TRAIN_SPLIT, BATCH_SIZE, model.frames_per_clip, model.num_classes)
        if j != 0 and j % num_samples < BATCH_SIZE:
            # end of epoch
            # save a model checkpoint and report end of epoch information
            save_path = os.path.join(model.model_dir, "model_epoch_%s.ckpt" % in_epoch)
            save_path = saver.save(sess, save_path)
            end = time.time()
            train_time = str(datetime.timedelta(seconds=end - start))
            print("END EPOCH %s, steps completed = %s, epoch training time: %s" % (in_epoch, j, train_time))
            print("model checkpoint saved to %s\n\n" % save_path)

            # test the network
            test_results = test_network(sess, model, TEST_SPLIT, run_name, in_epoch)
            run_csv_writer.writerow(test_results)

            in_epoch += 1

            if in_epoch % 4 == 0 and OPTIMIZER != 'Adam':
                model.current_learning_rate = model.current_learning_rate * LEARNING_RATE_DECAY
                print("learning rate adjusted to %g" % model.current_learning_rate)

            print("START EPOCH %s" % in_epoch)
            start = time.time()

        train_result = sess.run([train_op, loss_op, accuracy, x, y_true, y_true_class, y_pred, y_pred_class, logits, hit_5], 
                                feed_dict={x: x_feed, 
                                           y_true: y_feed, 
                                           learning_rate: model.current_learning_rate})
        loss_op_out = train_result[1]
        train_acc = train_result[2]
        x_actual = train_result[3]
        y_true_actual = train_result[4]
        y_true_class_actual = train_result[5]
        y_pred_actual = train_result[6]
        y_pred_class_actual = train_result[7]
        logits_out = train_result[8]
        hit5_out = train_result[9]

        # print("train_acc = %s" % train_acc)
        # train_acc is the number of correct responses divided by the batch size
        # e.g. 3 correct responses/30 batch size = 0.1
        train_acc_accum += train_acc

        # print("hit_5_out = %s" % hit5_out)
        # count the trues in the hit_5_out array
        hit5_counter = collections.Counter(hit5_out)
        hit5_out_trues = float(hit5_counter[True])
        # print("%s trues out of %s" % (hit5_out_trues, len(hit5_out)))
        train_hit5_accum += float(hit5_out_trues / len(hit5_out))

        # report out results and run a test mini-batch every now and then
        if j != 0 and j % (report_step * BATCH_SIZE) == 0:
            #print("logits = %s" % logits_out)
            #print("x = %s" % x_actual)
            #print("y_true = %s, y_true_class = %s, y_pred = %s, y_pred_class = %s" % (y_true_actual, y_true_class_actual, y_pred_actual, y_pred_class_actual))
            #print("train_acc = %s" % train_acc)
            print("hit5_out = %s, length = %s" % (hit5_out, len(hit5_out)))
            print("%s trues out of %s, %s accuracy" % (hit5_out_trues, len(hit5_out), hit5_out_trues / len(hit5_out)))
            run_time = time.time()
            run_time_str = str(datetime.timedelta(seconds=run_time - start))
            train_step_acc = train_acc_accum / report_step * BATCH_SIZE

            # mini batch accuracy - every 5 report step iterations
            if j % (report_step * BATCH_SIZE * 5) == 0:
                mini_batch_acc = 0.0
                mini_batch_hit5 = 0.0
                for k in range(MINI_BATCH_SIZE):
                    x_feed, y_feed, _, num_samples = get_image_batch(TRAIN_SPLIT, 1, model.frames_per_clip, model.num_classes)
                    acc, hit5_out, top_5_out, x_out = sess.run([eval_accuracy, eval_hit_5, eval_top_5, x],
                                                               feed_dict={x_test: x_feed, y_true_test: y_feed})
                    # print("type(x) = %s, x = %s" % (type(x_out), x_out))
                    mini_batch_acc += acc
                    hit5_counter = collections.Counter(hit5_out)
                    hit5_out_trues = hit5_counter[True]
                    # print("%s trues out of %s" % (hit5_out_trues, len(hit_5_out)))
                    mini_batch_hit5 += float(hit5_out_trues / len(hit5_out))

                mini_batch_acc = mini_batch_acc / MINI_BATCH_SIZE
                mini_batch_hit5 = mini_batch_hit5 / MINI_BATCH_SIZE
                
                print("\tstep %s - epoch %s run time = %s, loss = %s, mini-batch accuracy = %s, hit@5 = %s, y_true = %s, top 5 = %s" %
                     (j, in_epoch, run_time_str, loss_op_out, mini_batch_acc, mini_batch_hit5, y_true_class_actual, top_5_out))
                csv_row = ['mini-batch', in_epoch, j, loss_op_out, mini_batch_acc, mini_batch_hit5]
            
            else:
                train_acc_accum = train_acc_accum / report_step
                train_hit5_accum = train_hit5_accum / report_step
                print("\tstep %s - epoch %s run time = %s, loss = %s, train accuracy = %s, hit@5 = %s" %
                     (j, in_epoch, run_time_str, loss_op_out, train_acc_accum, train_hit5_accum))
                csv_row = ['train', in_epoch, j, loss_op_out, train_acc_accum, train_hit5_accum]

            # write the csv data to 
            run_csv_writer.writerow(csv_row)
            train_acc_accum = 0.0
            train_hit5_accum = 0.0

        j += BATCH_SIZE

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
