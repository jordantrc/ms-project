# model

import tensorflow as tf

def create_graph():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)
