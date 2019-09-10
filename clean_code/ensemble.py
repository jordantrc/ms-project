# ensemble.py
#
# This program takes as input a set of classified IADS...
#
# Usage:
# ensemble.py <train|test> <model name> <dataset name> <num_classes> <dataset size>
#   if train is specified, the model name will be the name of the saved model
#   if test is specified, the model name will be the name of the loaded model

import argparse
import csv
import numpy as np
import os
import sys
import tensorflow as tf

# optional - specify the CUDA device to use for GPU computation
# comment this line out if you wish to use all CUDA-capable devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##############################################
# Parameters
##############################################

# trial-specific parameters
# BATCH_SIZE is the number of samples per training batch
# EPOCHS is the number of training epochs to complete
# ALPHA is the learning rate
# MODEL_NAME is the model to use to test the trained network
# USE_WEIGHTS
BATCH_SIZE = 15
EPOCHS = 30
ALPHA = 1e-4
MODEL_NAME = "ensemble"
USE_WEIGHTS = False

# method to use to aggregate the results from the model,
# options are:
# average: take an average of the softmax values from all models, choose the
# largest response
# most common: take the largest response from each model, the predicted class
# is the class most-commonly
# associated with the largest response
AGGREGATE_METHOD = "average"

# CONSENSUS_HEURISTIC is used to determine the heuristic for generating
# a consensus classification result. Options are:
# top_5_confidence_discounted: the class output is the class with the highest
#   computed confidence. Model confidence values are discounted based on their
#   accuracy. The most accurate models have the least amount of
#   confidence-score discount.
#
# TODO
CONSENSUS_HEURISTIC = "top_5_confidence_discounted"

# C3D Specific parameters
num_features = [64, 128, 256, 256, 256, 6656]  # last element is for the combined model
window_size = [16, 16, 8, 4, 2, 1]


def layered_model(features, c3d_depth, num_classes):
    """Return a layered model."""
    # input layers
    input_layer = tf.reshape(features["x_" + str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1])  # batch_size, h, w, num_channels

    # hidden layers
    flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth]*window_size[c3d_depth]])
    dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
    dense2 = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(dense2, rate=0.5, training=features["train"])

    # output layers
    return tf.layers.dense(inputs=dropout, units=num_classes)


def model(features, c3d_depth, num_classes):
    """Return a single layer softmax model."""
    # input layers
    input_layer = tf.reshape(features["x_" + str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1])  # batch_size, h, w, num_channels

    # hidden layers
    flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth]*window_size[c3d_depth]])
    dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

    # output layers
    return tf.layers.dense(inputs=dropout, units=num_classes)


def conv_model(features, c3d_depth, num_classes):
    """Return a convolutional model."""
    # input layers
    input_layer = tf.reshape(features["x_" + str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1])  # batch_size, h, w, num_channels

    # hidden layers
    num_filters = 32
    filter_width = 4
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filters,
        kernel_size=[1, filter_width],
        padding="valid",  # don't want to add padding because that changes the IAD
        activation=tf.nn.leaky_relu)
    flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth] * window_size[c3d_depth]])
    dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

    # output layers
    return tf.layers.dense(inputs=dropout, units=num_classes)


def model_consensus(result, csv_writer, true_class):
    """Return a prediction based on the ensemble model consensus heuristic."""
    consensus = -1.
    confidences = result[4]
    classes = result[5]
    confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]
    avg_confidences = [
        0.38178119,
        0.56168587,
        0.56371784,
        0.54200298,
        0.49888024,
        0.71540061
    ]

    # write csv data
    # columns - ["true_class", "model", "place", "class", "confidence"]
    for i, r in enumerate(classes[0]):
        # i is the model
        for j, c in enumerate(r):
            # j is the place
            # c is the class
            row = [true_class[0], i, j, c, confidences[0][i][j]]
            csv_writer.writerow(row)

    # consensus heuristics
    if CONSENSUS_HEURISTIC == 'top_5_count':
        counts = np.bincount(top_5_indices)
        consensus = np.argmax(counts)

    elif CONSENSUS_HEURISTIC == 'top_10_confidence':
        confidence = [0.] * 101
        for i, m in enumerate(confidences[0]):
            # i is the model
            for j, p in enumerate(m):
                # j is the place
                if j in range(10):
                    label = classes[0][i][j]
                    confidence[label] += p
        consensus = np.argmax(confidence)

    elif CONSENSUS_HEURISTIC == 'top_5_confidence':
        confidence = [0.] * 101
        for i, m in enumerate(confidences[0]):
            # i is the model
            for j, p in enumerate(m):
                # j is the place
                if j in range(5):
                    label = classes[0][i][j]
                    confidence[label] += p
        consensus = np.argmax(confidence)

    elif CONSENSUS_HEURISTIC == 'top_5_confidence_discounted':
        confidence = [0.] * 101

        for i, m in enumerate(confidences[0]):
            # i is the model
            for j, p in enumerate(m):
                # j is the place
                if j in range(5):
                    label = classes[0][i][j]
                    confidence[label] += p * confidence_discount_layer[i]

        consensus = np.argmax(confidence)

    elif CONSENSUS_HEURISTIC == 'top_3_confidence_floored':
        confidence = [0.] * 101
        for i, m in enumerate(confidences[0]):
            # i is the model
            for j, p in enumerate(m):
                # j is the place
                if j in range(3):
                    # drop confidence if less than half average
                    if p < avg_confidences[i] / 2:
                        label = classes[0][i][j]
                        confidence[label] += p
        consensus = np.argmax(confidence)

    elif CONSENSUS_HEURISTIC == 'top_2_confidence':
        confidence = [0.] * 101
        for i, m in enumerate(confidences[0]):
            # i is the model
            for j, p in enumerate(m):
                # j is the place
                if j in range(2):
                    label = classes[0][i][j]
                    confidence[label] += p
        consensus = np.argmax(confidence)

    # write csv record
    # columns - ["true_class", "model", "place", "class", "confidence"]
    row = [true_class[0], "ensemble", 0, consensus, confidence[consensus]]
    csv_writer.writerow(row)

    return consensus


def read_file(filename_list):
    """Read the data from a given numpy data file."""
    all_data, all_labels = [], []
    for file in filename_list:
        infile = np.load(file)
        data, labels = infile["data"], infile["label"]
        data[data > 1] = 1.0
        data = data - 0.5
        all_data.append(data)
        all_labels = labels

    grouped_data = np.reshape(all_data[0], (all_data[0].shape[0], -1, 1))
    for i in range(1, len(all_data)):
        flat_data = np.reshape(all_data[i], (all_data[i].shape[0], -1, 1))
        grouped_data = np.concatenate((grouped_data, flat_data), axis=1)
    all_data.append(grouped_data)

    return all_data, all_labels


def tensor_operations(num_classes):
    """Create the tensor operations to be used in training and testing, stored in a dictionary."""
    # Placeholders
    ph = {
        "y": tf.placeholder(tf.int32, shape=(None)),
        "train": tf.placeholder(tf.bool)
    }

    for c3d_depth in range(6):
        ph["x_" + str(c3d_depth)] = tf.placeholder(
            tf.float32, shape=(None, num_features[c3d_depth], window_size[c3d_depth])
        )

    # Tensor operations
    loss_arr = []
    train_op_arr = []
    predictions_arr = []
    accuracy_arr = []
    weights = {}

    # for each model generate the tensor ops
    for c3d_depth in range(6):
        # logits
        if(c3d_depth < 3):
            logits = conv_model(ph, c3d_depth, num_classes)
        else:
            logits = model(ph, c3d_depth, num_classes)

        # probabilities and associated weights
        probabilities = tf.nn.softmax(logits, name="softmax_tensor")
        if USE_WEIGHTS:
            weights[c3d_depth] = tf.get_variable("weight_%s" % c3d_depth, shape=[1], initializer=tf.ones_initializer())
            probabilities = tf.multiply(probabilities, weights[c3d_depth], "probability_weight")

        # functions for predicting class
        predictions = {
            "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
            "probabilities": probabilities
        }
        predictions_arr.append(predictions)

        # functions for training/optimizing the network
        loss = tf.losses.sparse_softmax_cross_entropy(labels=ph["y"], logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        loss_arr.append(loss)
        train_op_arr.append(train_op)

        # functions for evaluating the network
        correct_pred = tf.equal(predictions["classes"], ph["y"])
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        accuracy_arr.append(accuracy)

    # combine all of the models together for the ensemble
    all_preds = tf.stack([x["probabilities"] for x in predictions_arr])
    all_preds = tf.transpose(all_preds, [1, 2, 0])

    model_preds = tf.transpose(all_preds, [0, 2, 1])
    model_top_10_values, model_top_10_indices = tf.nn.top_k(model_preds, k=10)
    model_preds = tf.argmax(model_preds, axis=2, output_type=tf.int32)

    if AGGREGATE_METHOD == 'average':
        # average over softmaxes
        test_prob = tf.reduce_mean(all_preds, axis=2)
        test_class = tf.argmax(test_prob, axis=1, output_type=tf.int32)

    elif AGGREGATE_METHOD == 'most_common':
        print("Aggregate method most_common not implemented")
        sys.exit(1)
        test_prob = tf.argmax(all_preds, axis=1, output_type=tf.int32)
        test_class = tf.argmax(tf.bincount(test_prob_max), output_type=tf.int32)

    # verify if prediction is correct
    test_correct_pred = tf.equal(test_class, ph["y"])
    operations = dict()
    operations['ph'] = ph
    operations['loss_arr'] = loss_arr
    operations['train_op_arr'] = train_op_arr
    operations['predictions_arr'] = predictions_arr
    operations['accuracy_arr'] = accuracy_arr
    operations['weights'] = weights
    operations['logits'] = logits
    operations['all_preds'] = all_preds
    operations['model_preds'] = model_preds
    operations['model_top_10_values'] = model_top_10_values
    operations['model_top_10_indices'] = model_top_10_indices
    operations['test_prob'] = test_prob
    operations['test_class'] = test_class
    operations['test_correct_pred'] = test_correct_pred

    return operations


def train_model(model, train, test, num_classes):
    """Train a model given the dataset, dataset parameters, and a model name."""
    ops = tensor_operations(num_classes)
    saver = tf.train.Saver()

    # load data
    train_data, train_labels = read_file(train)
    eval_data, eval_labels = read_file(test)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # train the network
        num_iter = EPOCHS * len(train_labels) / BATCH_SIZE
        for i in range(num_iter):
            # setup training batch
            batch = np.random.randint(0, len(train_data[0]), size=BATCH_SIZE)
            batch_data = {}
            for d in range(6):
                batch_data[ops['ph']["x_" + str(d)]] = train_data[d][batch]
            batch_data[ops['ph']["y"]] = train_labels[batch]

            batch_data[ops['ph']["train"]] = True

            # combine training operations into one variable
            training_operations = ops['train_op_arr'] + ops['loss_arr'] + ops['accuracy_arr']
            out = sess.run(training_operations, feed_dict=batch_data)

            # print out every 2K iterations
            if i % 2000 == 0:
                print("step: ", str(i) + '/' + str(num_iter))
                for x in range(6):
                    print("depth: ", str(x), "loss: ", out[6 + x], "train_accuracy: ", out[12 + x])

                # evaluate test network
                batch = np.random.randint(0, len(eval_data[0]), size=BATCH_SIZE)
                batch_data = {}
                for d in range(6):
                    batch_data[ops['ph']["x_" + str(d)]] = eval_data[d][batch]
                batch_data[ops['ph']["y"]] = eval_labels[batch]

                batch_data[ops['ph']["train"]] = False

                cp = sess.run([ops['test_correct_pred']], feed_dict=batch_data)
                # print("test_prob_max [shape = %s] = %s" % (tpm.shape, tpm))
                # print("test_class = %s" % tc)
                correct = np.sum(cp)
                total = len(cp[0])
                print("test: ", correct / float(total), "components: ", correct, total)

        # save the model
        save_path = saver.save(sess, model)
        print("Final model saved in %s" % save_path)


def test_model(model, test, num_classes):
    """Test the model."""
    ops = tensor_operations(num_classes)
    saver = tf.train.Saver()

    eval_data, eval_labels = read_file(test)

    test_batch_size = 1
    correct, total = 0, 0
    model_correct = [0, 0, 0, 0, 0, 0]
    num_iter = int(len(eval_labels) / test_batch_size)
    model_data_fd = open("model_%s_test_data.csv" % model, 'wb')
    model_csv = csv.writer(model_data_fd, dialect='excel')
    model_csv.writerow(["true_class", "model", "place", "class", "confidence"])
    confidences = [0.] * 6

    with tf.Session() as sess:
        # restore the model
        saver.restore(sess, model)
        print("Model restored from %s" % model)

        for i in range(num_iter):
            batch = range(i * test_batch_size, min(i * test_batch_size + test_batch_size, len(eval_labels)))
            batch_data = {}
            for d in range(6):
                batch_data[ops['ph']["x_" + str(d)]] = eval_data[d][batch]
            batch_data[ops['ph']["y"]] = eval_labels[batch]

            batch_data[ops['ph']["train"]] = False
            result = sess.run([
                ops['test_correct_pred'],
                ops['test_prob'],
                ops['all_preds'],
                ops['model_preds'],
                ops['model_top_10_values'],
                ops['model_top_10_indices']
            ], feed_dict=batch_data)

            ensemble_prediction = model_consensus(result, model_csv, batch_data[ops['ph']["y"]])

            # check if model output is correct
            for j, m in enumerate(result[3][0]):
                if m == batch_data[ops['ph']["y"]]:
                    model_correct[j] += 1
            if ensemble_prediction == batch_data[ops['ph']["y"]]:
                correct += 1

            # confidence collection
            for j, row in enumerate(result[4][0]):
                confidences[j] += np.max(row)

            total += len(result[0])

            if(i % 1000 == 0):
                print("step: ", str(i) + '/' + str(num_iter), "cummulative_accuracy:", correct / float(total))
                # per_layer accuracy
                # print("ap [%s]= %s" % (result[2].shape, result[2]))
                # print("tp [%s]= %s" % (result[1].shape, result[1]))
                # print("mp [%s] = %s" % (result[3].shape, result[3]))
                # print("true class = %s" % batch_data[ph["y"]])
                # print("top 5 values [%s] = %s" % (result[4].shape, result[4]))
                # print("top 5 indices [%s] = %s" % (result[5].shape, result[5]))

    model_data_fd.close()
    print("FINAL - accuracy:", correct / float(total))
    print("Model avg. confidence:")
    for i, v in enumerate(confidences):
        print("%s: %s" % (i, v / float(total)))

    print("Model accuracy: ")
    for i, c in enumerate(model_correct):
        print("%s: %s" % (i, c / float(total)))


def main():
    """Determine if the user has specified training or testing and run the appropriate function."""
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Ensemble model processor")
    parser.add_argument('action', help='action to perform, either train or test')
    parser.add_argument('model', help='model to save (when training) or to load (when testing)')
    parser.add_argument('dataset', help='dataset name on which to train/test')
    parser.add_argument('num_classes', help='the number of classes in the dataset')
    parser.add_argument('dataset_size', help='the size of the training dataset to load, choices: 100, 75, 50, 25')

    args = parser.parse_args()
    action = args.action
    model = args.model
    dataset = args.dataset
    num_classes = args.num_classes
    dataset_size = args.dataset_size

    # define the dataset file names
    dataset_directory = "%s_%s/" % (dataset, dataset_size)
    train_dataset = []
    test_dataset = []
    for c3d_depth in range(5):
        train_dataset.append("%s/%s_%s_train_%d.npz" % (dataset_directory, dataset, dataset_size, c3d_depth))
        test_dataset.append("%s/%s_%s_test_%d.npz" % (dataset_directory, dataset, dataset_size, c3d_depth))

    if action == 'train':
        train_model(model, train_dataset, test_dataset, num_classes)
    elif action == 'test':
        test_model(model, test_dataset, num_classes)
    else:
        print("Unsupported action: %s, exiting" % action)
        sys.exit(1)


if __name__ == "__main__":
    main()
