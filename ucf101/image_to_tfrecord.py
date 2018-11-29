# image_to_tfrecord.py
#
# converts a directory of images into tfrecord format

import os
import sys
import tensorflow as tf

from PIL import Image
from tfrecord_gen import get_class_list, _int64_feature, _bytes_feature

def main():
	'''main function'''
	directory = '/home/jordanc/datasets/UCF-101/UCF-101/'
	tfrecord_dir = '/home/jordanc/datasets/UCF-101/tfrecords'
	classes = get_class_list(CLASS_INDEX_FILE)

	for root, dirs, files in os.walk(directory):
		if "v_" in root:
			# create a tfrecord from the jpegs in this directory
			# directory format is:
			# /home/jordanc/datasets/UCF-101/UCF-101/BreastStroke/v_BreastStroke_g23_c02
			files.sort()
			video_name = os.path.basename(root)
			tfrecord_path = os.path.join(root, video_name + ".tfrecord")
			tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_path)
			assert tfrecord_writer is not None, "tfrecord_writer instantiation failed"

			# get the label and number of frames, create features dict
			label = video_name.split('_')[1]
			label_int = classes.index(label)
			num_frames = len(files)
			features = {'label': _int64_feature(label_int), 'num_frames': _int64_feature(num_classes)}

			# read each image and add it to the tfrecord
			for f in files:
				path = os.path.join(root, f)
				frame = np.asarray(Image.open(path)).astype(np.float32)
				frame_raw = frame.tostring()
				frame_index, _ = f.split('.')
				features['frames/%s' % frame_index] = _bytes_feature(frame_raw)

			example = tf.train.Example(features=tf.train.Features(feature=features))
			tfrecord_writer.write(example.SerializeToString())
			tfrecord_writer.close()
			print("%s images of shape %s written to tfrecord file %s" % (num_frames, frame.shape, tfrecord_path))


if __name__ == "__main__":
	main()