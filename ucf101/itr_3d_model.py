import numpy as np
import tensorflow as tf 

import dqn_code.c3d as c3d
import itr_3d as itr
from dqn_code.basic_tf_commands import weight_variable, bias_variable

NUM_LABEL = 101

# define placeholders
def get_placeholders(batch_size=1, c3d_depth=0, num_channels=1):
	'''
	Define the placeholders used by the network. 
		-c3d_depth: the depth at which the c3d activation map should be drawn
	'''

	return {
		# imported placeholders
		"c3d_in": c3d.get_input_placeholder(batch_size),
		"itr_in": itr.get_input_placeholder(batch_size, c3d_depth, num_channels=num_channels),

		"system_out": tf.placeholder(tf.float32,shape=(batch_size, NUM_LABEL),name="out_ph")
		}


def get_variables(c3d_depth=0, num_channels=1):
	'''
	Define the variables used by the network. 
		-c3d_depth: the depth at which the c3d activation map should be drawn
	'''

	# imported variables
	c3d_w, c3d_b = c3d.get_variables()
	itr_w, itr_b = itr.get_variables(num_channels=num_channels)

	# system specific variables
	system_w = {"W_0": weight_variable("W_0", [itr.calculate_output_size(c3d_depth), NUM_LABEL])}
	system_b = {"b_0": bias_variable("b_0", [NUM_LABEL])}

	#coalated variables
	weights = {"c3d": c3d_w,
						"itr": itr_w,
						"system": system_w}
	biases = {"c3d": c3d_b,
						"itr": itr_b,
						"system": system_b}

	return weights, biases

def list_variables(weights, biases):
	return {
		"c3d": list( set(weights["c3d"].values() + biases["c3d"].values()) ),
		"itr": list( set(weights["itr"].values() + biases["itr"].values()) ),
		"system": list( set(weights["system"].values() + biases["system"].values()) )
		}

def get_predicted_values(placeholders, _weights, _biases, c3d_depth=0):
	'''
	Define the full structure of the network. 
		-placeholders: a set of placeholder variables established by running the
			"get_placeholders" function
		-weights: a set of weights established by running the
			"get_variables" function
		-weights: a set of biases established by running the
			"get_variables" function
	'''

	# Generate the output of the ITR network
	itr_activation_map = itr.generate_activation_map(placeholders["itr_in"], _weights["itr"], _biases["itr"])
	itr_activation_map = tf.reshape(itr_activation_map, (1, itr.calculate_output_size(c3d_depth)))

	# Generate pre_q values using a fully connected layer
	return tf.matmul(itr_activation_map, _weights["system"]["W_0"], name="pre_q_matmul") + _biases["system"]["b_0"]

def classifier(model):
	'''
	Outputs the best available action
		-model: a TF model as would be generated via the "get_predicted_values" 
			function
	'''
	return tf.argmax(model,1)

def optimizer(placeholders, model, alpha=1e-3):
	'''
	Optimizes the network
		-placeholders: a set of placeholder variables established by running the
			"get_placeholders" function
		-model: a TF model as would be generated via the "get_predicted_values" 
			function
		-alpha: the learning rate for the Adam optimizer
	'''
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=placeholders["system_out"], logits=model)
	return tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)




