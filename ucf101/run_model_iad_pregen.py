import tensorflow as tf 
import numpy as np 

from file_io import read_files_in_dir
import itr_3d as itr
import itr_3d_model as model_def

CHKPT_NAME = ""

NUM_CHANNELS = 1

def generate_model_input(placeholders, data_source, sess, c3d_map_depth):
	'''
	Reads a single entry from the specified data_source and stores the entry in
	the provided placeholder for use in sess
		-placeholders: a dictionary of placeholders
		-data_source: a tensor leading to a collection of TFRecords as would be 
			generated with the "input_pipeline" function
		-sess: the current session
	'''

	# read a single entry of the data_source into numpy arrays
	input_tensor = [sess.run(data_source)][0]
	np_values = {"img": input_tensor[0],
				"label": input_tensor[1],
				"example_id": input_tensor[2]}

	img_data = np_values["img"].reshape((-1, 
								itr.NUM_FEATURES[c3d_map_depth], 
								itr.LENGTH_OF_FRAME[c3d_map_depth],
								NUM_CHANNELS))

	#place in dict
	placeholder_values = {
		placeholders["itr_in"]: img_data}

	information_values = {
		"label": np_values["label"],
		"example_id": np_values["example_id"]}#,
		#"data_ratio": data_ratio}

	return placeholder_values, information_values

def obtain_IAD_input(placeholders, tf_records, sess, c3d_map_depth):
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
	ph_values, info_values = generate_model_input(placeholders, tf_records, sess, c3d_map_depth)

	new_label = np.unravel_index(np.argmax(info_values["label"], axis=1), info_values["label"].shape)[1] - 1
	if(new_label < 0):
		new_label = 0
	
	new_label_array = np.zeros(3).reshape((1,3))
	new_label_array[0, new_label] = 1
	info_values["label"] = new_label_array

	ph_values[placeholders["system_out"]] = new_label_array

	# Perform thresholding on C3D activation map
	return ph_values, info_values

def train(placeholders, tf_records, sess, optimizer, c3d_map_depth):
	# Optimize variables in the rest of the network
	ph_values, info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_map_depth)
	sess.run(optimizer, feed_dict=ph_values) 

def identify_response(file_name):
	# Identify what responses the given file name were depicting
	tag = file_name
	while (tag.find('_') >= 0):
		tag = tag[tag.find('_')+1:]
	return tag[:-1]

def evaluate(placeholders, tf_records, sess, classifier, dataset_size, c3d_map_depth, verbose=False, model=None):
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
		ph_values, info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_map_depth)
		predicted_action = sess.run(classifier, feed_dict=ph_values)[0]

		# Identify the correct action for the observation
		labels = info_values["label"][0]
		correct_action = np.unravel_index(np.argmax(labels, axis=None), labels.shape)

		# Update the confusion matrix
		confusion_matrix[correct_action][predicted_action] += 1

		if(model != None):
			predicted_values = sess.run(model, feed_dict=ph_values)[0]
			print(i, info_values["example_id"][0], predicted_values, predicted_action, correct_action)

	return confusion_matrix, responses

def get_accuracy(confusion_matrix):
	# Calculate the accuracy for a given confusion matrix
	correct = 0
	for n in range(confusion_matrix.shape[0]):
		correct += confusion_matrix[n][n]
	return correct/float(np.sum(confusion_matrix))

def run_model(
	num_train_iterations=10, 
	c3d_depth=0, 
	training_dir='', 	
	training_dir_dataset_limit=0,
	validate_dir='', 
	testing_dir='',
	train_print_freq=0,
	validation_freq=0,
	save_freq=0):

	# ----------  setup variables ------------
	# setup variables
	placeholders = model_def.get_placeholders(c3d_depth=c3d_depth, num_channels=NUM_CHANNELS)
	weights, biases = model_def.get_variables(c3d_depth=c3d_depth, num_channels=NUM_CHANNELS)
	variable_name_dict = model_def.list_variables(weights, biases)

	# define model
	model = model_def.get_predicted_values(placeholders, weights, biases, c3d_depth=c3d_depth)
	classifier = model_def.classifier(model)
	optimizer = model_def.optimizer(placeholders, model, alpha=1e-3)

	with tf.Session() as sess:

		# ----------  file I/O ------------

		# define files for training/testing

		training_records, testing_records, validate_records = None, None, None
		test_iter, valid_iter = 0, 0

		if(training_dir != ''):
			training_records, _ = read_files_in_dir(training_dir, randomize=True, recursive=False)
		
		if(testing_dir != ''):
			testing_records, test_iter = read_files_in_dir(testing_dir, randomize=False, recursive=False)
		
		if(validate_dir != ''):
			validate_records, valid_iter = read_files_in_dir(validate_dir, randomize=False, recursive=False)

		# ----------  restore variables (update) ------------

		saver = tf.train.Saver( 
				list( set(variable_name_dict["itr"] + \
					variable_name_dict["system"]) ))

		if(CHKPT_NAME != ''):
			saver.restore(sess, CHKPT_NAME)
		else:
			sess.run(tf.global_variables_initializer())

		# ----------  initalize variables ------------

		# setup variables
		sess.run(tf.local_variables_initializer())
		
		# ----------  finalize model ------------

		# ensure no additional changes are made to the model
		sess.graph.finalize()

		# start queue runners in order to read ipnut files
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		# ----------  train network ------------
		
		for iteration in range(num_train_iterations):	
			train(placeholders, training_records, sess, optimizer, c3d_depth)
			
			if(train_print_freq > 0 and iteration % train_print_freq == 0):
				print(iteration)
			
			if(validation_freq > 0 and iteration % validation_freq == 0): 
				# test the system on the validation dataset				
				confusion_matrix, responses = evaluate(placeholders, validate_records, sess, classifier, valid_iter, c3d_depth, verbose=False)
				print("VAL "+str(iteration)+" accuracy: "+str(get_accuracy(confusion_matrix))+'\n')

			if(save_freq > 0 and iteration % save_freq == 0 and iteration > 0):
				# save the model to file
				saver.save(sess, 'simul_step_'+str(c3d_depth)+'/model.ckpt', global_step=iteration)
			
		# ----------  test network ------------

		# test the system on the testing dataset
		confusion_matrix, responses = evaluate(placeholders, testing_records, sess, classifier, test_iter, c3d_depth, verbose=True, model=model)
		print("TEST accuracy: "+str(get_accuracy(confusion_matrix))+'\n')
		print(confusion_matrix)

		for k in responses:
			print(k, responses[k])

		# save final model to chekpoint file
		saver.save(sess, 'simul_final_'+str(c3d_depth)+'/model.ckpt')

		# ----------  close session ------------

		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':

	for depth in range(5):
		run_model(	
			num_train_iterations=5000, 
			c3d_depth=depth, 
			training_dir="../iad_3d_tfrecords_max/"+str(depth)+"/test", 	
			training_dir_dataset_limit=0,
			validate_dir='', 
			testing_dir="../iad_3d_tfrecords_max/"+str(depth)+"/train",
			train_print_freq=100,
			validation_freq=0,
			save_freq=1000)
		tf.reset_default_graph()
