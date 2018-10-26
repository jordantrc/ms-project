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

    c3d_model.c3d_network(NUM_EPOCHS)

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
