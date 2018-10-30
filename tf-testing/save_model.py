# model save

import tensorflow as tf

def save_model():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

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
