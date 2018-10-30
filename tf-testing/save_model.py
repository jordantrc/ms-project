# model save

import tensorflow as tf
import model


def save_model():
    model.create_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(101):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

        if i % 10 == 0:
            saver.save(sess, 'Model/second', global_step=i)

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    # saver.save(sess, 'Model/first')

    sess.close()

if __name__ == "__main__":
    save_model()
