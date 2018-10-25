# train_network.py
#
# TODO:
# - incorporate tfrecord data input queue
# - fix storage of labels as integers (indexes into the class index file)

import os
import tensorflow as tf
import c3d_model
import c3d
import time
import datetime

NUM_EPOCHS = 16

# get the list of files for train and test
train_files = [os.path.join(c3d_model.TRAIN_DIR, x) for x in os.listdir(c3d_model.TRAIN_DIR)]
test_files = [os.path.join(c3d_model.TEST_DIR, x) for x in os.listdir(c3d_model.TEST_DIR)]

current_learning_rate = c3d_model.LEARNING_RATE

tf.reset_default_graph()

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
    dataset = tf.data.TFRecordDataset(train_filenames)
    dataset = dataset.map(c3d_model._parse_function)
    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.batch(c3d_model.BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    x, y_true = iterator.get_next()

    # print("x = %s, shape = %s" % (x, x.get_shape().as_list()))
    # convert x to float, reshape to 5d
    # x = tf.cast(x, tf.float32)
    # print("reshaping x")
    # print("x pre-reshape = %s, shape = %s" % (x, x.get_shape().as_list()))
    # print("x pre-clip = %s, shape = %s" % (x, x.get_shape().as_list()))
    x = tf.reshape(x, [c3d_model.BATCH_SIZE, 250, 112, 112, 3])

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
    optimizer = tf.train.AdamOptimizer(learning_rate=current_learning_rate)

    train_op = optimizer.minimize(loss_op)

    # evaluate the model
    correct_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()
    sess.run(init_op)

    print("Beginning training epochs")

    for i in range(NUM_EPOCHS):
        print("START EPOCH %s" % i)
        start = time.time()
        sess.run(iterator.initializer, feed_dict={train_filenames: train_files, test_filenames: test_files})
        while True:
            try:
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
        save_path = os.path.join(c3d_model.MODEL_DIR, "model_epoch_%s.ckpt" % i)
        save_path = saver.save(sess, save_path)
        end = time.time()
        train_time = str(datetime.timedelta(seconds=end - start))
        print("END EPOCH %s, epoch training time: %s seconds" % (i, train_time))
        print("model saved to %s\n\n" % save_path)

        if i != 0 and i % 4 == 0:
            current_learning_rate = current_learning_rate / 10
            print("learning rate adjusted to %g" % current_learning_rate)

    print("end training epochs")

    coord.request_stop()
    coord.join(threads)
    sess.close()
