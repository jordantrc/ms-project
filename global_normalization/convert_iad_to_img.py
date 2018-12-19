import cv2
import numpy as np 
import sys
import tensorflow as tf

from iad_cnn_2d.file_io import input_pipeline
import iad_cnn_2d.itr_3d as itr

NUM_CHANNELS=1

def generate_model_input_iad(data_source, sess, c3d_map_depth):
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
				"uid": input_tensor[1],
				"length": input_tensor[2],
				"verb": input_tensor[3],
				"noun": input_tensor[4]}
	
	# pad and reshape the img data for use with the C3D network
	num_features = itr.NUM_FEATURES[c3d_map_depth]
	time_frames = itr.LENGTH_OF_FRAME[c3d_map_depth]

	print('num_features:', num_features)
	print('time_frames:', time_frames)
	print('np_values["length"]:', np_values["length"])

	if (np_values["length"] > time_frames):
		return 0, 0

	data_ratio = float(np_values["length"][0])/time_frames
	buffer_len = (time_frames-np_values["length"][0])# * num_features

	img_data = np_values["img"].reshape((num_features, np_values["length"][0], 1))
	print(img_data.shape)
	img_data = np.pad(img_data, 
									((0,0), (0,buffer_len), (0,0)), 
									'constant', 
									constant_values=(0,0))
	
	img_data = np.expand_dims(img_data, axis=0)

	print('np_values["img"].shape:', np_values["img"].shape)
	print('img_data.shape:', img_data.shape)

	dr = np.reshape(np_values["img"], (num_features, np_values["length"][0]))

	print("data:", dr[:,0])

	information_values = {
		"uid": np_values["uid"][0],
		"length": np_values["length"][0],
		"verb": np_values["verb"][0],
		"noun": np_values["noun"][0],
		"data_ratio": data_ratio}

	#print(img_data.shape)
	#print(information_values["verb"].shape, information_values["verb"])

	return img_data
	
def convert_iad_to_img(iad, outname):
	img = iad

	img.astype(np.uint8)

	print(img.shape)

	img_new = []
	start_shape = img.shape[2]
	

	for i in range(min(img.shape[-1], 3)):
		cur_frame = img[0,:,:,i]
		print(cur_frame)
		buf = np.zeros((img.shape[1],1))

		print(type(cur_frame), type(buf))
		print("cur_frame: ", cur_frame.shape, buf.shape)
		cur_frame = np.concatenate((cur_frame, buf), axis=1)

		if(i == 0):
			img_new = cur_frame
		else:
			img_new = np.concatenate((img_new, cur_frame), axis=1)

	print("img_new: ", img_new.shape)
	img = img_new
	img *= 255
	img.astype(np.uint8)
	#img = np.reshape(img, (img.shape[1], img.shape[2]))

	#val = int(img.shape[1]*.5)
	#img = img[:, :val]

	img -= 1
	img = np.stack((img,img,img), axis = 2)
	for n in range(0, img.shape[1], 10):
		img[:,n, 0] =.25

	for n in range(0, img.shape[1], start_shape):
		img[:,n, 1] =.25
	#img *= -255

	sizer = 6
	img = cv2.resize(img, None,fx=sizer, fy=sizer, interpolation=cv2.INTER_NEAREST)
	line = np.zeros_like(img.shape[1])

	for x in range(img.shape[0], 0, -sizer):
		img = np.insert(img, x, line, axis=0)

	for x in range(img.shape[1], 0, -sizer):
		img = np.insert(img, x, line, axis=1)

	cv2.imwrite(outname, img)

if __name__ == '__main__':

	if(len (sys.argv) == 3):
		infile = sys.argv[1]
		c3d_map_depth = int(sys.argv[2])

		with tf.Session() as sess:
			tfrecords = input_pipeline([infile])

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord, sess=sess)

			iad = generate_model_input_iad(tfrecords, sess, c3d_map_depth)

			#print(iad[:30])
			#print(iad[0,0,:20,:])

			outfile = infile[:-len(".tfrecord")]+".png"
			print("written to:", outfile)
			convert_iad_to_img(iad, outfile)

			coord.request_stop()
			coord.join(threads)
	else:
		print("Usage: python convert_iad_to_img.py <input_filename> <c3d_depth>")