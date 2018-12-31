# iad_svm.py
#
# Non-linear support-vector machine classifier for 
# IADs
#

import fnmatch
import os
import random
import sys
import tensorflow as tf

import analysis
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

LAYER = 5
SETTINGS = 'train'
#SETTINGS = 'test'

if SETTINGS == 'train':
    BATCH_SIZE = 10
    FILE_LIST = 'train-test-splits/train.list_expanded'
    MODEL_SAVE_DIR = 'iad_models/'
    LOAD_MODEL = None
    EPOCHS = 5
elif SETTINGS == 'test':
    BATCH_SIZE = 1
    FILE_LIST = 'train-test-splits/test.list_expanded'
    MODEL_SAVE_DIR = 'iad_models/'
    LOAD_MODEL = 'iad_models/iad_svm_model_layer_%s_step_final.ckpt' % LAYER
    EPOCHS = 1
NUM_CLASSES = 101
#CLASSES_TO_INCLUDE = ['ApplyEyeMakeup', 'Knitting', 'Lunges', 'HandStandPushups', 'Archery', 'MilitaryParade',
#                      'YoYo', 'BabyCrawling', 'BaseballPitch', 'BenchPress', 'Bowling', 'Drumming',
#                      'BalanceBeam', 'BandMarching', 'Fencing', 'FloorGymnastics', 'Haircut', 'Hammering',
#                      'HeadMassage', 'HighJump', 'HulaHoop', 'JavelinThrow', 'JumpingJack', 'Kayaking']
CLASSES_TO_INCLUDE = 'all'
TRAINING_DATA_SAMPLE = 1.0

LEARNING_RATE = 1e-3
GAMMA = -1.0

# the layer from which to load the activation map
# layer geometries - shallowest to deepest
# layer 1 - 64 features x 16 time slices
# layer 2 - 128 features x 16 time slices
# layer 3 - 256 features x 8 time slices
# layer 4 - 512 features x 4 time slices
# layer 5 - 512 features x 2 time slices
LAYER_GEOMETRY = {'1': (64, 16, 1),
                  '2': (128, 16, 1),
                  '3': (256, 8, 1),
                  '4': (512, 4, 1),
                  '5': (512, 2, 1)
                  }

#-------------General helper functions----------------#
def save_settings(run_name):
    '''saves the parameters to a file'''
    with open('runs/%s_parameters.txt' % run_name, 'w') as fd:
        fd.write("BATCH_SIZE = %s\n" % BATCH_SIZE)
        fd.write("FILE_LIST = %s\n" % FILE_LIST)
        fd.write("MODEL_SAVE_DIR = %s\n" % MODEL_SAVE_DIR)
        fd.write("LOAD_MODEL = %s\n" % LOAD_MODEL)
        fd.write("EPOCHS = %s\n" % EPOCHS)
        fd.write("CLASSES_TO_INCLUDE = %s\n" % CLASSES_TO_INCLUDE)
        fd.write("TRAINING_DATA_SAMPLE = %s\n" % TRAINING_DATA_SAMPLE)
        fd.write("LEARNING_RATE = %s\n" % LEARNING_RATE)
        fd.write("GAMMA = %s\n" % GAMMA)
        fd.write("LAYER = %s\n" % LAYER)


def list_to_filenames(list_file):
    '''converts a list file to a list of filenames'''
    filenames = []
    class_counts = {}
    class_files = {}

    with open(list_file, 'r') as list_fd:
        text = list_fd.read()
        lines = text.split('\n')

    if '' in lines:
        lines.remove('')

    for l in lines:
        found_files = 0
        sample, label = l.split()
        sample_basename = os.path.basename(sample)

        class_name = sample_basename.split('_')[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
            class_files[class_name].append(sample)
        else:
            class_counts[class_name] = 1
            class_files[class_name] = [sample]


    # balance classes if we're training
    if LOAD_MODEL is None:
        print("balancing files across classes")
        max_class_count = -1
        for k in class_counts.keys():
            if class_counts[k] > max_class_count:
                max_class_count = class_counts[k]

        # add files to filenames, add as many as possible and then
        # sample the remainder less than max_class_count
        print("oversample size = %s" % max_class_count)
        if CLASSES_TO_INCLUDE == 'all':
            keys = class_counts.keys()
        else:
            keys = [x for x in class_counts.keys() if x in CLASSES_TO_INCLUDE]

        for k in keys:
            oversample = max_class_count - class_counts[k]
            filenames.extend(class_files[k])
            if oversample <= len(class_files[k]):
                filenames.extend(random.sample(class_files[k], oversample))
            else:
                class_filenames = class_files[k]
                while len(class_filenames) < max_class_count:
                    class_filenames.append(random.sample(class_files[k], 1)[0])
                filenames.extend(class_filenames)

    else:
        if CLASSES_TO_INCLUDE == 'all':
            keys = class_counts.keys()
        else:
            keys = [x for x in class_counts.keys() if x in CLASSES_TO_INCLUDE]

        for k in sorted(keys):
            filenames.extend(class_files[k])

    return filenames


def save_model(sess, saver, step):
    save_path = os.path.join(MODEL_SAVE_DIR, "iad_svm_model_layer_%s_step_%s.ckpt" % (LAYER, step))
    saver.save(sess, save_path)

#-------------SVM Functions---------------------------#

def _parse_function(example):
    img_geom = tuple([1]) + LAYER_GEOMETRY[str(LAYER)]
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    # decode the image, get label
    img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
    img = tf.reshape(img, img_geom, "parse_reshape")

    #img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #print("img shape = %s" % img.get_shape())
    img = tf.squeeze(img, 0)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return img, label


#------------------main--------------------------------#
def main():
    '''main function'''
    if LOAD_MODEL is None:
        training = True
        run_type = 'train'
    else:
        training = False
        run_type = 'test'

    # get the run name
    run_name = sys.argv[1]
    save_settings(run_name + "_" + run_type)

    # get the list of classes
    class_list = get_class_list(CLASS_INDEX_FILE)

    # get the list of filenames
    print("loading file list from %s" % FILE_LIST)
    filenames = list_to_filenames(FILE_LIST)
    print("%s files" % len(filenames))
    if training:
        random.shuffle(filenames)
        if TRAINING_DATA_SAMPLE != 1.0:
            filenames = random.sample(filenames, int(TRAINING_DATA_SAMPLE * len(filenames)))

    # ensure filenames list is evenly divisable by batch size
    pad_filenames = len(filenames) % BATCH_SIZE
    filenames.extend(filenames[0:pad_filenames])
    print("filenames = %s..." % filenames[0:5])

    # create the TensorFlow sessions
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # dataset iterator
    input_filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    if training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat(EPOCHS)
    else:
        dataset = dataset.repeat(1)
    dataset_iterator = dataset.make_initializable_iterator()
    x, y_true = dataset_iterator.get_next()
    y_true_class = tf.argmax(y_true, axis=1)

    # placeholders
    prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    # variables
    b = tf.Variable(tf.random_normal(shape=[NUM_CLASSES, BATCH_SIZE]))
    gamma = tf.constant(GAMMA)
    dist = tf.reduce_sum(tf.square(x), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.multiply(2., tf.matmul(x, tf.transpose(x)))
    svm_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # batch multiplication
    def reshape_matmul(mat, _size):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [NUM_CLASSES, _size, 1])
        return tf.matmul(v2, v1)

    # compute SVM model
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_true_cross = reshape_matmul(y_true, BATCH_SIZE)
    second_term = tf.reduce_sum(tf.multiply(svm_kernel, tf.multiply(b_vec_cross, y_true_cross)), [1, 2])
    loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])

    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(y_true, b), pred_kernel)
    y_pred_class = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), axis=0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_true, 0)), tf.float32))

    # operations
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_step = train_op.minimize(loss)

    # initializer
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init_op)

    # start the training/testing steps
    if training:
        print("begin training")
        step = 0
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

        # loop until out of data
        while True:
            try:
                train_result = sess.run([train_op, accuracy, x])
                if step != 0 and step % 100 == 0:
                    print("step %s, accuracy = %s" % (step, train_result[1]))
                    # save the current model every 1000 steps
                    if step % 1000 == 0:
                        save_model(sess, saver, step)
                # print("x = %s, logits = %s" % (train_result[2], train_result[3]))
                step += 1
            except tf.errors.OutOfRangeError:
                print("data exhausted, saving final model")
                save_model(sess, saver, 'final')
                break
    else:
        print("begin testing")
        step = 0
        saver.restore(sess, LOAD_MODEL)
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

        cumulative_accuracy = 0.0
        predictions = []
        true_classes = []
        # loop until out of data
        while True:
            try:
                test_result = sess.run([accuracy, x, y_pred_class, y_true_class])
                cumulative_accuracy += test_result[0]
                predictions.append(test_result[2])
                true_classes.append(test_result[3])
                if step % 100 == 0:
                    print("step %s, accuracy = %s, cumulative accuracy = %s" %
                          (step, test_result[0], cumulative_accuracy / step / BATCH_SIZE))
                step += 1
            except tf.errors.OutOfRangeError:
                break

        # wrap up, provide test results
        print("data exhausted, test results:")
        print("steps = %s, cumulative accuracy = %.04f" % (step, cumulative_accuracy / step / BATCH_SIZE))
        #for i, p in enumerate(predictions):
        #    print("[%s] true class = %s, predicted class = %s" % (i, true_classes[i], p))

        cm = analysis.confusion_matrix(predictions, true_classes, class_list)
        print("confusion matrix = %s" % cm)
        analysis.plot_confusion_matrix(cm, class_list, "runs/" + run_name + ".pdf")
        print("per-class accuracy:")
        analysis.per_class_table(predictions, true_classes, class_list, "runs/" + run_name + '.csv')


if __name__ == "__main__":
    main()