import cv2
import numpy as np 
import sys
import tensorflow as tf

from iad_3d.file_io import input_pipeline
import iad_3d.itr_3d as itr

NUM_CHANNELS=10

def generate_model_input(data_source, sess, c3d_map_depth):
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

	return np_values["img"].reshape((-1, 
								itr.NUM_FEATURES[c3d_map_depth], 
								itr.LENGTH_OF_FRAME[c3d_map_depth],
								NUM_CHANNELS))
	
def convert_iad_to_img(iad, outname):
	img = iad

	img.astype(np.uint8)

	print(img.shape)

	img_new = []
	start_shape = img.shape[2]
	for i in range(3):#img.shape[-1]):
		cur_frame = img[:,:,:,i]
		print(cur_frame)
		buf = np.zeros((img.shape[0],img.shape[1], 1))

		print(type(cur_frame), type(buf))
		print("cur_frame: ", cur_frame.shape, buf.shape)
		cur_frame = np.concatenate((cur_frame, buf), axis=2)

		if(i == 0):
			img_new = cur_frame
		else:
			img_new = np.concatenate((img_new, cur_frame), axis=2)

	print("img_new: ", img_new.shape)
	img = img_new[0]
	img *= 255
	img.astype(np.uint8)
	#img = np.reshape(img, (img.shape[1], img.shape[2]))

	val = int(img.shape[1]*.5)
	img = img[:, :val]


	img_frame = np.copy(img)
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

			iad = generate_model_input(tfrecords, sess, c3d_map_depth)

			outfile = infile[:-len(".tfrecord")]+".png"
			convert_iad_to_img(iad, outfile)

			coord.request_stop()
			coord.join(threads)