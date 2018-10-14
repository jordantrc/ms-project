import tensorflow as tf
import numpy as np

from file_io import read_files_in_dir, generate_model_input
import c3d
import itr_dqn_model as model_def
from thresholding_methods import thresholding


C3D_NETWORK_VARIABLE_FILE = "../../../C3D-tensorflow/sports1m_finetuning_ucf101.model"
CHKPT_NAME = ""


def threshold_activation_map(ph_values, placeholders, information_values, activation_map, thresholding_approach="norm"):
    '''
    Applies the specified thresholding method to the provided activation map and
    stores the result in the ITR input placeholder.
        - ph_values: a dictionary containing the placeholders
        -placeholders: the list of placeholders used by the network
        -information_values: a dictionary containing extraenous information about
                the input
        -activation_map: the C3D activation map
        -thresholding_approach: the thresholding method to use a string that is
                either "mean", "histogram", "entropy", or "norm"
    '''

    # Apply thresholding to the provided activation map
    thresholded_map = thresholding(activation_map[0], information_values["data_ratio"], thresholding_approach)

    # multiply values by 255 to match the scale of values being used by the Audio network
    thresholded_map *= 255
    thresholded_map = np.reshape(thresholded_map,
                                 [1, thresholded_map.shape[0], thresholded_map.shape[1], 1])

    ph_values[placeholders["itr_in"]] = thresholded_map
    return ph_values


def obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach):
    '''
    Provides the training input for the ITR network by generating an IAD from the
    activation map of the C3D network. Outputs two dictionaries. The first contains
    the placeholders that will be used when evaluating the full ITR model. The second
    contains information about the observation being read (ie. true labels, number of
    prompts, file name, etc).
        -placeholders: the list of placeholders used by the network
        -tf_records: the TFRecord data source to read from
        -sess: the tensorflow Session
        -c3d_model: the c3d network model
    '''

    # Obtain activation amp from C3D network
    ph_values, info_values = generate_model_input(placeholders, tf_records, sess)
    c3d_activation_map = sess.run(c3d_model, feed_dict=ph_values)

    # Perform thresholding on C3D activation map
    threshold_map = threshold_activation_map(ph_values, placeholders, info_values, c3d_activation_map,
                                             thresholding_approach)

    return threshold_map, info_values


def train(placeholders, tf_records, sess, c3d_model, thresholding_approach, optimizer):
    # Optimize variables in the rest of the network
    ph_values, info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach)
    sess.run(optimizer, feed_dict=ph_values)


def identify_response(file_name):
    # Identify what responses the given file name were depicting
    tag = file_name
    while (tag.find('_') >= 0):
        tag = tag[tag.find('_') + 1:]
    return tag[:-1]


def evaluate(placeholders, tf_records, sess, c3d_model, thresholding_approach, classifier, dataset_size, verbose=False):
    '''
    Evaluates the ITR model on the given dataset. Two outputs are generated: a
    confusion matrix indicating the models correct and predicted guesses and a
    dictionary containing the accuracy given the number of prompts executed and
    the response observed in the video.
    '''
    confusion_matrix = np.zeros((model_def.NUM_LABEL, model_def.NUM_LABEL))
    responses = {}

    for i in range(dataset_size):
        # Generate predicted values
        ph_values, info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach)
        predicted_action = sess.run(classifier, feed_dict=ph_values)[0]

        # Identify the correct action for the observation
        labels = info_values["label"][0]
        correct_action = np.unravel_index(np.argmax(labels, axis=None), labels.shape)

        # Update the confusion matrix
        confusion_matrix[correct_action][predicted_action] += 1

        # Determine the composition of the observation
        response_depicted = identify_response(info_values["example_id"][0])
        if(response_depicted not in responses):
            responses[response_depicted] = [[0, 0], [0, 0]]

        # Determine how many prompts have been delivered in the observation
        number_of_prompts = info_values["pt"]

        # update the response accuracy
        if correct_action == predicted_action:
            responses[response_depicted][number_of_prompts][0] += 1
        responses[response_depicted][number_of_prompts][1] += 1

        if(verbose):
            print(i, info_values["example_id"][0], number_of_prompts, correct_action, predicted_action)

    return confusion_matrix, responses


def get_accuracy(confusion_matrix):
    # Calculate the accuracy for a given confusion matrix
    correct = 0
    for n in range(confusion_matrix.shape[0]):
        correct += confusion_matrix[n][n]
    return correct / float(np.sum(confusion_matrix))


def run_model(num_train_iterations=10,
              c3d_depth=0,
              thresholding_approach="norm",
              training_dir='',
              training_dir_dataset_limit=0,
              validate_dir='',
              testing_dir='',
              train_print_freq=0,
              validation_freq=0,
              save_freq=0,
              recursive=False):

    # ----------  setup variables ------------

    # setup variables
    placeholders = model_def.get_placeholders(c3d_depth=c3d_depth)
    weights, biases = model_def.get_variables(c3d_depth=c3d_depth)
    variable_name_dict = model_def.list_variables(weights, biases)

    # define model
    c3d_model = c3d.generate_activation_map(placeholders["c3d_in"], weights["c3d"], biases["c3d"], depth=c3d_depth)
    model = model_def.get_predicted_values(placeholders, weights, biases, c3d_depth=c3d_depth)
    classifier = model_def.classifier(model)
    optimizer = model_def.optimizer(placeholders, model, alpha=1e-3)

    with tf.Session() as sess:

        # ----------  file I/O ------------

        # define files for training/testing

        training_records, testing_records, validate_records = None, None, None
        test_iter, valid_iter = 0, 0

        if(training_dir != ''):
            training_records, _ = read_files_in_dir(training_dir, randomize=True,
                                                    limit_dataset=training_dir_dataset_limit,
                                                    recursive=recursive)
        if(testing_dir != ''):
            testing_records, test_iter = read_files_in_dir(testing_dir, randomize=False, recursive=recursive)
        if(validate_dir != ''):
            validate_records, valid_iter = read_files_in_dir(validate_dir, randomize=False, recursive=recursive)

        # ----------  restore variables (update) ------------

        saver = tf.train.Saver(variable_name_dict["c3d"])
        restore_filename = C3D_NETWORK_VARIABLE_FILE

        if(CHKPT_NAME != ''):
            restore_filename = CHKPT_NAME
            saver = tf.train.Saver(
                list(set(variable_name_dict["c3d"] +
                         variable_name_dict["itr"] +
                         variable_name_dict["aud"] +
                         variable_name_dict["system"])))

        # ----------  initalize variables ------------

        # setup variables
        if(train):
            sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if(not train):
            saver.restore(sess, restore_filename)

        # ----------  finalize model ------------

        # ensure no additional changes are made to the model
        sess.graph.finalize()

        # start queue runners in order to read ipnut files
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # ----------  train network ------------

        for iteration in range(num_train_iterations):
            train(placeholders, training_records, sess, c3d_model, thresholding_approach, optimizer)

            if(train_print_freq > 0 and iteration % train_print_freq == 0):
                print(iteration)

            if(validation_freq > 0 and iteration % validation_freq == 0):
                # test the system on the validation dataset
                confusion_matrix, responses = evaluate(placeholders, validate_records, sess, c3d_model,
                                                       thresholding_approach, classifier, valid_iter, verbose=False)
                print("VAL " + str(iteration) + " accuracy: " + str(get_accuracy(confusion_matrix)) + '\n')

            if(save_freq > 0 and iteration % save_freq == 0):
                # save the model to file
                saver.save(sess, 'itr_step/model.ckpt', global_step=iteration)
        # ----------  test network ------------

        # test the system on the testing dataset
        confusion_matrix, responses = evaluate(placeholders, testing_records, sess, c3d_model, thresholding_approach,
                                               classifier, test_iter, verbose=True)
        print("TEST accuracy: " + str(get_accuracy(confusion_matrix)) + '\n')
        print(confusion_matrix)

        for k in responses:
            print(k, responses[k])

        # save final model to chekpoint file
        saver.save(sess, 'itr_final/model.ckpt')

        # ----------  close session ------------

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    run_model(num_train_iterations=3,
              c3d_depth=0,
              thresholding_approach="norm",
              training_dir="/home/jordanc/datasets/UCF-101/tfrecords",
              training_dir_dataset_limit=0,
              validate_dir='',
              testing_dir="/home/jordanc/datasets/UCF-101/tfrecords",
              train_print_freq=0,
              validation_freq=0,
              save_freq=0)

'''
    train_len = 10000

    train = True
    if(train):
        #01 03 05 07
        #09 11 13 15
        #17 19 21 23
        for i in range(11, 12, 2):
            #i=0
            print("Dataset Size: ", i, i*18)
            itr_exec = DQNExecutor(
                            train_len=train_len, 
                            training_dir=INP_FILE, 
                            testing_dir=TEST_FILE,
                            validate_dir=VAL_FILE, 
                            #test_len=60, #20 from each label 
                            c3d_depth=0,
                            thresh_method="norm",
                            dataset_limit=i)
            itr_exec.run_global_initalizers()
            itr_exec.finalize()
            itr_exec.run()
            tf.reset_default_graph()
'''
