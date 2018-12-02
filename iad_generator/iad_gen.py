import tensorflow as tf
import numpy as np

import c3d 
from file_io import input_pipeline
import os
from os.path import join, isdir, isfile

from thresholding_3d import thresholding

C3D_NETWORK_VARIABLE_FILE = "../../../cnn3d-str/C3D-tensorflow/sports1m_finetuning_ucf101.model"


def read_files_in_dir(directory):
	contents = [join(directory, f) for f in os.listdir(directory)]

	all_files = []
	for f in contents:
		if isfile(f):
			all_files += [f]
		elif isdir(f):
			all_files += read_files_in_dir(f)
	
	return all_files

def make_sequence_example(
	img_raw, 
	first_action,
	example_id,
	c3d_depth,
	num_channels):

	print(first_action, example_id)

	# The object we return
	ex = tf.train.SequenceExample()

	# ---- descriptive data ----

	ex.context.feature["length"].int64_list.value.append(img_raw.shape[1])
	ex.context.feature["example_id"].bytes_list.value.append(example_id)

	# ---- label data ----

	ex.context.feature["total_lab"].int64_list.value.append(first_action)
	ex.context.feature["c3d_depth"].int64_list.value.append(c3d_depth)
	ex.context.feature["num_channels"].int64_list.value.append(num_channels)
	
	# ---- data sequences ----

	def load_array(example, name, data, dtype):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		print("newShape:", np.asarray(data).astype(dtype).shape)
		fl_data.append(np.asarray(data).astype(dtype).tostring())

	load_array(ex, "img_raw", img_raw, np.float32)

	return ex

def generate_model_input(placeholders, data_source, sess):
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

	# pad and reshape the img data for use with the C3D network
	input_length = np_values["img"].shape[1]

	size_of_single_img_frame = c3d.INPUT_DATA_SIZE["h"]*c3d.INPUT_DATA_SIZE["w"]*c3d.INPUT_DATA_SIZE["c"]

	
	data_ratio = ( float(input_length)/ size_of_single_img_frame ) / c3d.INPUT_DATA_SIZE["t"]
	buffer_len = (c3d.INPUT_DATA_SIZE["t"] * size_of_single_img_frame) - input_length

	img_data = np.pad(np_values["img"], 
									((0,0), (0,buffer_len)), 
									'constant', 
									constant_values=(0,0))
	img_data = img_data.reshape((-1, 
								c3d.INPUT_DATA_SIZE["h"],
								c3d.INPUT_DATA_SIZE["w"],
								c3d.INPUT_DATA_SIZE["c"]))
	img_data = np.expand_dims(img_data, axis=0)

	#place in dict
	placeholder_values = {
		placeholders: img_data}

	information_values = {
		"label": np.unravel_index(np.argmax(np_values["label"]), dims =(4)),
		"example_id": np_values["example_id"],
		"data_ratio": data_ratio}

	return placeholder_values, information_values

def convert_to_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach, compression_method, video_name, c3d_depth):
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
	print(c3d_activation_map.shape)

	thresholded_data = thresholding(c3d_activation_map[0], info_values["data_ratio"], compression_method, thresholding_approach)

	print(thresholded_data)


	ex = make_sequence_example(thresholded_data, info_values["label"][0], info_values["example_id"][0], c3d_depth, compression_method["value"])
	print("write to: ", video_name)
	writer = tf.python_io.TFRecordWriter(video_name)
	writer.write(ex.SerializeToString())
	writer.close()

if __name__ == '__main__':

	# open the files 
 
	compression_method={"type":"peaks", "value":10, "num_channels":10}

	# setup variables
	placeholders = c3d.get_input_placeholder(1)
	weights, biases = c3d.get_variables()
	variable_name_dict = list( set(weights.values() + biases.values()))

	cur_dir = "../one_person_tfrecords"
	filenames = read_files_in_dir(cur_dir)
	for f in filenames:
		print(f)

	

	for c3d_depth in range(5):
		new_dir = "../iad_3d_tfrecords/"+str(c3d_depth)+"/"

		# define model
		c3d_model = c3d.generate_activation_map(placeholders, weights, biases, depth=c3d_depth)

		with tf.Session() as sess:


			saver = tf.train.Saver(variable_name_dict)
			

			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			
			saver.restore(sess, C3D_NETWORK_VARIABLE_FILE)

			#setup file io
			src = input_pipeline(filenames)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord, sess=sess)
			
			#process files
			for f in filenames:
				print(f)

				new_name = new_dir+f[len(cur_dir)+1:]

				new_dir_name = new_name
				while (new_dir_name[-1] != '/'):
					new_dir_name = new_dir_name[:-1]

				print(new_dir_name)
				if not os.path.exists(new_dir_name):
					os.makedirs(new_dir_name)




				
				convert_to_IAD_input(placeholders, src, sess, c3d_model, "norm", compression_method, new_name, c3d_depth)

			coord.request_stop()
			coord.join(threads)
		
		#tf.reset_default_graph()

