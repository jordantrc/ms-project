import tensorflow as tf
import numpy as np

import c3d 
from file_io import input_pipeline, generate_model_input
import os
from os.path import join, isdir, isfile

from thresholding_3d import thresholding

from multiprocessing import RawArray
import multiprocessing
from threading import Thread, Semaphore

import time, tempfile

batch_size=1

C3D_NETWORK_VARIABLE_FILE = "../../catkin_ws/src/cnn3d-str/C3D-tensorflow/sports1m_finetuning_ucf101.model"

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

	info_values,

	c3d_depth,
	num_channels):


	# The object we return
	ex = tf.train.SequenceExample()

	# ---- descriptive data ----

	ex.context.feature["length"].int64_list.value.append(img_raw.shape[1])
	ex.context.feature["uid"].int64_list.value.append(info_values["uid"])

	# ---- label data ----

	ex.context.feature["verb"].int64_list.value.append(info_values["verb"])
	ex.context.feature["noun"].int64_list.value.append(info_values["noun"])


	ex.context.feature["c3d_depth"].int64_list.value.append(c3d_depth)
	ex.context.feature["num_channels"].int64_list.value.append(num_channels)
	
	# ---- data sequences ----

	def load_array(example, name, data, dtype):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		fl_data.append(np.asarray(data).astype(dtype).tostring())

	load_array(ex, "img_raw", img_raw, np.float32)

	return ex

##### Get the Maximum and Minimum Values of IADs #####

class Record:
	def __init__(self, filename, info_values):
		self.filename = filename
		self.uid = info_values["uid"]

		self.info_values=info_values

def get_row_min_max(c3d_activation_map, info_values, sem, max_vals, min_vals, records):
	'''
	Get the IAD for the activation map and record the highest and lowest observed values.
	'''
	
	thresholded_data = thresholding(c3d_activation_map[0], info_values["data_ratio"], compression_method, "none")

	local_max_values = np.max(thresholded_data, axis=1)
	local_min_values = np.min(thresholded_data, axis=1)

	for i in range(len(local_max_values)):

		if(local_max_values[i] > max_vals[i]):
			max_vals[i] = local_max_values[i]

		if(local_min_values[i] < min_vals[i]):
			min_vals[i] = local_min_values[i]


	# save current un-thresholded data to file
	filename = tempfile.NamedTemporaryFile().name+".npy"
	np.save(filename, thresholded_data)
	records.put(Record(filename, info_values))
	
	sem.release()

def identify_min_maxes(filenames, records):
	placeholders = c3d.get_input_placeholder(batch_size)
	weights, biases = c3d.get_variables()
	variable_name_dict = list( set(weights.values() + biases.values()))

	sem = Semaphore(4)

	for c3d_depth in range(1):#5):

		max_vals, min_vals = RawArray('d', 64), RawArray('d', 64)
		for i in range(64):
			max_vals[i] = float("-inf")
			min_vals[i] = float("inf")

		# define model
		c3d_model = c3d.generate_activation_map(placeholders, weights, biases, depth=c3d_depth)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:

				saver = tf.train.Saver(variable_name_dict)
				
				sess.run(tf.global_variables_initializer())
				sess.run(tf.local_variables_initializer())
				
				saver.restore(sess, C3D_NETWORK_VARIABLE_FILE)

				#setup file io
				tf_records = input_pipeline(filenames, batch_size=batch_size)
				sess.graph.finalize()
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord, sess=sess)
				
				#process files
				for i in range(len(filenames)):
					if(i %1000 == 0 ):
						print("Converted "+str(i)+" files")

					
					ph_values, info_values = generate_model_input(placeholders, tf_records, sess)

					all_procs = []

					if(ph_values != 0):
						#generate activation map from 3D-CNN
						c3d_activation_map = sess.run(c3d_model, feed_dict=ph_values)
						
						# try to acquire a ticket, if ticket is available converting activation map to IAD
						# have to use Semaphores here because activation maps are save on GPU. ATTempting to start multiple threads 
						# means that the GPU never relesaes the memory used for activation maps.
						sem.acquire()
						p = Thread(target=get_row_min_max, args=(c3d_activation_map, info_values,sem, max_vals, min_vals, records))
						p.start()
						all_procs.append(p)
						
				for p in all_procs:
					p.join()
					
				coord.request_stop()
				coord.join(threads)

	return max_vals, min_vals

##### Convert Activation Map to IAD #####

def rethreshold_iad(iad, max_vals, min_vals):
	'''
	re-threshold the given iad with the new max and min values
	'''

	for index in range(data.shape[0]):
		data_row = data[index]

		max_val_divider = max_vals[index] - min_vals[index]
		data_row -= max_vals[index]
		if(max_val_divider != 0):
			data_row /= max_val_divider
		else:
			data_row = list(np.zeros_like(data_row))
		
		floor_values = data_row > 1.0
		data_row[floor_values] = 1.0

		ceil_values = data_row < 0.0
		data_row[ceil_values] = 0.0

		iad[index] = data_row
	return iad

def repeat_threshold(unthresholded_data, record, sem, max_vals, min_vals):
	'''
	Convert activation map into IAD. Takes the following parameters:
		- c3d_activation_map
		- info_values - as defined by generate_model_input
		- sem - a semaphore
		- max_vals - an array of maximum values for available features
		- min_vals - an array of minimum values for available features
	'''
	
	thresholded_data = rethreshold_iad(unthresholded_data, max_vals, min_vals)

	print("thresholded_data.shape:", thresholded_data.shape)
	info_values = record.info_values
	
	ex = make_sequence_example(thresholded_data, info_values, c3d_depth, compression_method["num_channels"])

	video_name = new_dir + str(record.uid).zfill(6)+".tfrecord"

	print(thresholded_data.shape)
	
	writer = tf.python_io.TFRecordWriter(video_name)
	writer.write(ex.SerializeToString())
	writer.close()
	
	sem.release()

def process_file(records, max_vals, min_vals):
	'''
	opens an unthreshodled IAD and thresholds given the new values
	'''

	NUM_THREADS = 4
	sem = Semaphore(NUM_THREADS)

	all_procs = []
	i = 0

	while(not records.empty()):

		if(i %1000 == 0 ):
			print("A Processes has converted "+str(i)+" files")

		r = records.get()
		unthresholded_data = np.load(r.filename)
		os.system("rm "+r.filename)

		sem.acquire()
		p = Thread(target=repeat_threshold, args=(unthresholded_data, r,sem, max_vals, min_vals))
		p.start()
		all_procs.append(p)
		i+=1
			
	for p in all_procs:
		p.join()
					
			
##### Main #####

if __name__ == '__main__':
 
 	thresholding_approach = "min_max_norm"
 	compression_method={"type":"max", "value":1, "num_channels":1}

	cur_dir = "../train_records/"
	filenames = read_files_in_dir(cur_dir)

	

	num_procs = 1
	t_s = time.time()

	#store thresholded info in temporary files in a a queue
	records = multiprocessing.Queue()

	for c3d_depth in range(1):#5):
		all_procs = []
		new_dir = "../iad_records/"+"min_max_norm"+"/"+str(c3d_depth)+"/"
		print(new_dir)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)

		# get the maximum and mimimum activation values for each iad row
		max_vals, min_vals = identify_min_maxes(filenames, records)

		#need to reset graph after each map generation because graphs are read-only
		tf.reset_default_graph()

		# generate IADs using the earlier identified values
		process_file(records, max_vals, min_vals)

		tf.reset_default_graph()
	print("completed_in: ", time.time()-t_s)
			

