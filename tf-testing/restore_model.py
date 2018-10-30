# restore model

import tensorflow as tf

def restore_model():
    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('Model')

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    sess.close()

if __name__ == "__main__":
    restore_model()