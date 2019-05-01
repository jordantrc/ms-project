import tensorflow as tf 
import numpy as np

# only run on one compute server GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

##############################################
# Parameters
##############################################

#trial specific
batch_size = 15
epochs = 16
alpha = 1e-4
model_name = "ensemble"
use_weights = False

# method to use to aggregate the results from the model,
# options are:
# average: take an average of the softmax values from all models, choose the largest response
# most common: take the largest response from each model, the predicted class is the class most-commonly
# associated with the largest response
aggregate_method = "most_common"

#dataset specific
dataset_size = 50
dataset_name = "ucf"
num_classes = 101

train_dataset = []
test_dataset = []
for c3d_depth in range(5):
  train_dataset.append(dataset_name+"_"+str(dataset_size)+"/"+dataset_name+"_"+str(dataset_size)+"_train_%d.npz" % c3d_depth)
  test_dataset.append(dataset_name+"_"+str(dataset_size)+"/"+dataset_name+"_"+str(dataset_size)+"_test_%d.npz" % c3d_depth)

#C3D specific
num_features = [64,128,256,256,256,6656] # last element is for the combined model
window_size = [16,16,8,4,2,1]

##############################################
# Model Spec
##############################################

def layered_model(features, c3d_depth):
  #input layers
  input_layer = tf.reshape(features["x_"+str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1]) # batch_size, h, w, num_channels

  #hidden layers
  flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth]*window_size[c3d_depth]])
  dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)

  #dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.leaky_relu)

  dropout = tf.layers.dropout(dense, rate=0.8, training=features["train"])

  #output layers
  return tf.layers.dense(inputs=dropout, units=num_classes)

def model(features, c3d_depth):
  #input layers
  input_layer = tf.reshape(features["x_"+str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1]) # batch_size, h, w, num_channels

  #hidden layers
  flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth]*window_size[c3d_depth]])
  dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
  dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

  #output layers
  return tf.layers.dense(inputs=dropout, units=num_classes)

def conv_model(features, c3d_depth):
  #input layers
  input_layer = tf.reshape(features["x_"+str(c3d_depth)], [-1, num_features[c3d_depth], window_size[c3d_depth], 1]) # batch_size, h, w, num_channels

  #hidden layers
  num_filters=32
  filter_width=4
  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=num_filters,
    kernel_size=[1, filter_width],
    padding="valid", # don't want to add padding because that changes the IAD
    activation=tf.nn.leaky_relu)
  flatten = tf.reshape(input_layer, [-1, num_features[c3d_depth]*window_size[c3d_depth]])
  dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
  dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

  #output layers
  return tf.layers.dense(inputs=dropout, units=num_classes)

#################### Placeholders ##########################
ph = {  "y" : tf.placeholder(tf.int32, shape=(None)),
        "train" : tf.placeholder(tf.bool)   }

for c3d_depth in range(6):
  ph["x_"+str(c3d_depth)] = tf.placeholder(
    tf.float32, shape=(None,num_features[c3d_depth],window_size[c3d_depth])
  )

#################### Tensor Ops ##########################

loss_arr, train_op_arr, predictions_arr, accuracy_arr = [], [], [], []
weights = {}

# for each model generate the tensor ops
for c3d_depth in range(6):

  # logits
  if(c3d_depth < 3):
    logits = conv_model(ph, c3d_depth)
  else:
    #logits = model(ph, c3d_depth)
    logits = model(ph, c3d_depth)

  # probabilities and associated weights

  probabilities = tf.nn.softmax(logits, name="softmax_tensor")
  if use_weights:
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

  optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
  train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())

  loss_arr.append(loss)
  train_op_arr.append(train_op)

  # functions for evaluating the network
  correct_pred = tf.equal(predictions["classes"], ph["y"])
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  accuracy_arr.append(accuracy)

# combine all of the models together for the ensemble
all_preds = tf.stack([x["probabilities"] for x in predictions_arr])
all_preds = tf.transpose(all_preds, [1,2,0])
test_prob = tf.reduce_mean(all_preds, axis = 2)

if aggregate_method == 'average':
  # average over softmaxes
  test_class = tf.argmax(test_prob, axis=1, output_type=tf.int32)

elif aggregate_method == 'most_common':
  test_prob_max = tf.argmax(test_prob, axis=1, output_type=tf.int32)
  test_class = tf.argmax(tf.bincount(test_prob_max), output_type=tf.int32)

# verify if prediction is correct
test_correct_pred = tf.equal(test_class, ph["y"])


##############################################
# File IO
##############################################


def read_file(filename_list):
  all_data, all_labels = [], []
  for file in filename_list:
    infile = np.load(file)
    data, labels = infile["data"], infile["label"]
    data[data > 1] = 1.0
    data = data-0.5
    all_data.append(data)
    all_labels = labels

  grouped_data = np.reshape(all_data[0], (all_data[0].shape[0], -1, 1))
  for i in range(1,len(all_data)):
    flat_data = np.reshape(all_data[i], (all_data[i].shape[0], -1, 1))
    grouped_data = np.concatenate((grouped_data, flat_data), axis = 1)
  all_data.append(grouped_data)
  assert len(all_data) == len(all_labels), "data length does not match label length"
  return all_data, all_labels

train_data, train_labels = read_file(train_dataset)
eval_data, eval_labels = read_file(test_dataset)

##############################################
# Run Model
##############################################

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  # Train the network
  num_iter = epochs * len(train_labels) / batch_size
  for i in range(num_iter):
    
    #setup training batch
    batch = np.random.randint(0, len(train_data[0]), size=batch_size)
    batch_data = {}
    for d in range(6):
      batch_data[ph["x_"+str(d)]] = train_data[d][batch]
    batch_data[ph["y"]] = train_labels[batch]

    batch_data[ph["train"]] = True


    training_operations = train_op_arr + loss_arr + accuracy_arr
    out = sess.run(training_operations, feed_dict=batch_data)

    # print out every 2K iterations
    if(i % 2000 == 0):
      print("step: ", str(i)+'/'+str(num_iter))
      for x in range(6):
        print("depth: ", str(x),"loss: ", out[6+x], "train_accuracy: ", out[12+x])

      # evaluate test network
      batch = np.random.randint(0, len(eval_data[0]), size=batch_size)
      batch_data = {}
      for d in range(6):
        batch_data[ph["x_"+str(d)]] = eval_data[d][batch]
      batch_data[ph["y"]] = eval_labels[batch]

      batch_data[ph["train"]] = False


      cp = sess.run([test_correct_pred], feed_dict=batch_data)

      correct = np.sum(cp)
      total = len(cp[0])
      print("test:", correct / float(total))

  #Test the finished network
  correct, total = 0,0
  num_iter = int(len(eval_labels) / batch_size)

  for i in range(0, num_iter+1):

    batch = range(i*batch_size, min(i*batch_size+batch_size, len(eval_labels)))
    batch_data = {}
    for d in range(6):
      batch_data[ph["x_"+str(d)]] = eval_data[d][batch]
    batch_data[ph["y"]] = eval_labels[batch]
    
    batch_data[ph["train"]] = False
    cp = sess.run([test_correct_pred], feed_dict=batch_data)
    
    correct += np.sum(cp)
    total += len(cp[0])

    if(i % 1000 == 0):
      print("step: ", str(i)+'/'+str(num_iter), "cummulative_accuracy:", correct / float(total))

  print("FINAL - accuracy:", correct/ float(total))