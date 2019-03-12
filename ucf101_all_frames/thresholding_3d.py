import numpy as np
import threading, math

import peak_finding

THRESHOLDING_OPTIONS = ["mean", "histogram", "entropy", "norm", "none", "min_max_norm"]

class CountDownLatch(object):
	#A Count down latch class to assist in multithreading
	def __init__(self, count=1):
		self.count = count
		self.lock = threading.Condition()

	def count_down(self):
		self.lock.acquire()
		self.count -= 1
		if self.count <= 0:
			self.lock.notifyAll()
		self.lock.release()

	def await(self):
		self.lock.acquire()
		while self.count > 0:
			self.lock.wait()
		self.lock.release()

def threshold_activations(activations, out, latch, thresholding_method, index = None, max_val = None, min_val = None):
	'''
	Perform the specified thresholding method on the activations. 
		-activations: a numpy array the compressed spatial expression of filters
		-out: a list to place the thresholded output in
		-latch: a count down latch to assist multithreading
		-thresholding_method: a string either "mean", "histogram", "entropy",
				or "norm"that designates the type of thresholding to perform
	'''

	# Choose the maximum expresion that occurs spatially in each frame for each 
	#filter 
	returned_values = None

	if(thresholding_method == "norm"):
		# Perform scaling/normalization
		activations -= activations.min()
		if(activations.max() != 0):
			activations /= activations.max()
		else:
			activations = list(np.zeros_like(activations))
		returned_values = activations
	elif(thresholding_method == "min_max_norm"):
		# Perform scaling/normalization

		#print("index:", index)
		#if(index == 1 and np.min(activations) != min_val[index]):
		#	print("mismatch:", min_val[index], np.min(activations))
		
		max_val_divider = max_val[index] - min_val[index]
		activations -= min_val[index]
		if(max_val_divider != 0):
			activations /= max_val_divider
		else:
			activations = list(np.zeros_like(activations))
		
		#ceil and floor all values outside of the max min value range
		#pre_cf  = np.copy(activations)

		floor_values = activations > 1.0
		activations[floor_values] = 1.0


		ceil_values = activations < 0.0
		

		#print(ceil_values)
		#print(activations)

		activations[ceil_values] = 0.0
		
		#if(pre_cf.all() != activations.all()):
		#	print("activations, don't match "+str(min_val[index])+ ' '+str(max_val[index]))





		returned_values = activations

	else:
		threshold = 0.0
		if(thresholding_method == "mean"):
			# Perform mean thresholding
			threshold = np.mean(activations)

		else:
			# Set up histograms for iterative methods
			hist, bin_edges = np.histogram(activations, bins, normed=False)
			pmf = [float(i)/np.sum(hist) for i in hist]
			
			metric, min_val = None, 0

			if(thresholding_method == "histogram"):
				# Perform histogram-based thresholding
				for i in range (1,len(hist)):

					pmf_t = np.sum(pmf[:i])

					chunk = np.copy(hist[:i]) * range(1, i+1)
					b1 = np.sum(chunk) / pmf_t

					chunk = np.copy(hist[i:]) * range(i+1, len(hist)+1)
					b2 = np.sum(chunk) * (1 - pmf_t)
					sum_b1 = [(b1 - j)**2 for j in range(0, i)]
					sum_b1 = np.sum(sum_b1)

					sum_b2 = [(b2 - j)**2 for j in range(i+1, len(hist))]
					sum_b2 = np.sum(sum_b2)

					eta_t = sum_b1 + sum_b2

					if(metric == None or eta_t < metric):
						metric = eta_t
						min_val = i

			elif(thresholding_method == "entropy"):
				# Perform entropy-based thresholding
				for i in range (1,len(hist)):

					moment0a = np.sum(hist[:i])
					chunk = np.copy(hist[:i]) * range(1, i+1)
					moment1a = np.sum(chunk)

					moment0b = np.sum(hist[i:])
					chunk = np.copy(hist[i:]) * range(i+1, len(hist)+1)
					moment1b = np.sum(chunk)

					if(moment0a != 0 and moment0b != 0):

						mean_a = moment1a/moment0a
						mean_b = moment1b/moment0b

						if(mean_a != 0 and mean_b != 0):

							eta_t = - (moment1a * math.log(mean_a)) - (moment1b * math.log(mean_b))
							if(metric == None or eta_t < metric):
								metric = eta_t
								min_val = i

			# select the threshold number that performed the best
			threshold = bin_edges[min_val]
			returned_values = activations > threshold

	out.append(returned_values)
	latch.count_down()

def compress_activations(activations, out, compression_method, compression_latch, thresholding_method, index = None, max_val=None, min_val=None):
	#separate the activations into spatial subdivisions
	activation_divisions = []

	if(compression_method["type"] == "max"):
		activations = activations.reshape((activations.shape[0], -1))
		activation_divisions = [np.max(activations, axis=1)]

	elif(compression_method["type"] == "even"):
		step = activations.shape[1]/compression_method["value"]
		
		for n in range(0, activations.shape[1], step):
			for m in range(0, activations.shape[2], step):
				restruct = activations[:, n:n+step, m:m+step].reshape((activations.shape[0], -1))
				activation_divisions.append(np.max(restruct, axis=1))

	elif(compression_method["type"] == "peaks"):
		activation_divisions = peak_finding.peak_finding_fast(activations, n = compression_method["value"])

	#perform thresholding on each of the spatial subdivisions


	thresholded_activations = []
	if (thresholding_method != "none"):
		
		for i in range(len(activation_divisions)):
			thresholded_activations.append([])

		activation_latch = CountDownLatch(len(activation_divisions))
		list_of_threads = []
		for i in range(len(activation_divisions)):
			#print("start")
			if(len(activation_divisions[i]) > 0):
				t = None
				if(thresholding_method == "min_max_norm"):
					assert max_val != None, "max_val variable must be set"
					assert min_val != None, "max_val variable must be set"
					t = threading.Thread(target = threshold_activations, args = (activation_divisions[i], thresholded_activations[i], activation_latch, thresholding_method,index+i, max_val,min_val,))
				else:
					t = threading.Thread(target = threshold_activations, args = (activation_divisions[i], thresholded_activations[i], activation_latch, thresholding_method,))
				list_of_threads.append(t)
				t.start()
			else:
				#print("err: ", i, activation_divisions[i])
				activation_latch.count_down()

		# wait for all threads to finish
		activation_latch.await()

	else:
		thresholded_activations = activation_divisions
	

	#print(thresholded_activations[0][:5])
	out.append( np.array(thresholded_activations).squeeze() )
	compression_latch.count_down()

def thresholding(activation_map, data_ratio, compression_method={"type":"max", "value":-1}, thresholding_method="none", max_val=None, min_val=None):
	'''
	Perform the specified thresholding method on the entire activation map provided. 
		-activation_map: a 4D numpy array containing the output of a 3D-CNN convolution
		-data_ratio: a float indicating the amount of usefule (un-padded) frames
				in the original input, the rest can be ignored
		-thresholding_method: a string either "mean", "histogram", "entropy", or "norm"
				that designates the type of thresholding to perform
	'''

	# asserts to make sure input is correctly formed
	assert len(activation_map.shape) == 4, "input to 'feature_to_event' must be 4D matrix, input has "+str(len(activation_map.shape))+" dimensions"
	#assert data_ratio > 0.0 and data_ratio <= 1.0, "Data ratio parmater must be a float 0 < x <= 1, is :" + str(data_ratio)
	assert thresholding_method in THRESHOLDING_OPTIONS, "Thresh_method must be in "+str(THRESHOLDING_OPTIONS)+", is: "+ str(thresholding_method)

	num_events = activation_map.shape[-1]
	initial_length = activation_map.shape[0]
	#print("num_events = %s, initial_length = %s" % (num_events, initial_length))

	#only perform thresholding on the un-padded region of the input
	#activation_map = np.array(activation_map)[:int(data_ratio*activation_map.shape[0])]
	activation_map = np.array(activation_map)

	#set up a list to contain the identified start and stop times
	thresholded_activations = []
	for i in range(num_events):
		thresholded_activations.append([])
	
	#setup multithread processes to perform the threading operation
	latch = CountDownLatch(num_events)

	list_of_threads = []
	for i in range(num_events):
		# isolate the activations for filter 'i' and perform the thresholding for 
		# this feature in its own thread
		#activations = np.reshape(activation_map[...,i], (activation_map.shape[0], -1)) 
		activations = activation_map[...,i]
		t = threading.Thread(target = compress_activations, args = (activations, thresholded_activations[i], compression_method, latch, thresholding_method, i, max_val, min_val, ))

		list_of_threads.append(t)
		t.start()
		
	# wait for all threads to finish
	latch.await()
	#print("compression_finished:", np.array(thresholded_activations).squeeze().shape)

	# pad and resize array for use in the ITR network
	#thresholded_activations *= 255

	#NEED TO ORGANIZE 3D-IAD


	thresholded_activations = np.array(thresholded_activations).squeeze()#.astype(np.int64)
	#print("thresholded_activations[0]:", thresholded_activations[0][:5])

	#print(thresholded_activations[0][:5])
	
	
	if(len(thresholded_activations.shape) > 2):
		thresholded_activations = np.transpose(thresholded_activations, (0,2,1))
	'''
		thresholded_activations = np.pad(thresholded_activations, 
										((0,0), (0,initial_length-thresholded_activations.shape[1]), (0,0)), 
										'constant', 
										constant_values=(0,0))
	else:
		thresholded_activations = np.pad(thresholded_activations, 
										((0,0), (0,initial_length-thresholded_activations.shape[1])), 
										'constant', 
										constant_values=(0,0))
	print("thresholded_activations: ", thresholded_activations.shape)
	'''

	#print("thresholded_activations: ", thresholded_activations.shape)
	
	return thresholded_activations 