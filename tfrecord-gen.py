#
# tfrecord-gen.py
#
# Converts various datasets to
# tfrecords.
#
# Some code taken from https://github.com/ferreirafabio/video2tfrecord/blob/master/video2tfrecord.py
#
# Usage:
#
# tfrecord-gen.py <Dataset> <root directory> <output directory>
#

from __future__ import print_function

import cv2
import os
import random
import sys


def main():
    # parse arguments

    if len(sys.argv) != 4:
    	print_help()
    	sys.exit(1)

    dataset_name = sys.argv[1]
    root_directory = sys.argv[2]
    output_directory = sys.argv[3]

    if dataset_name == "UCF101":
    	ucf101_dataset(root_directory, output_directory)


def ucf101_dataset(root, output):
	"""generates tfrecords for the ucf101 dataset"""
	# sampling information - number of samples of sample_length
	# in seconds
	sample_randomly = True
	num_samples = 5
	sample_length = 2

	assert os.path.isdir(root) and os.path.isdir(output)
	class_index_file = os.path.join(root, "classInd.txt")
	video_directory = os.path.join(root, "UCF-101")

	# read class index file to get the class
	# index information
	classes = {}
	with open(class_index_file) as class_index_file_fd:
		print("Opening %s" % (class_index_file))
		lines = class_index_file_fd.read().split('\n')
		#print(lines)
		for l in lines:
			if len(l) > 0:
				i, c = l.split(" ")
				classes[i] = [c]
	assert len(classes) > 0

	# get count of each class
	smallest_class = None
	smallest_class_num = -1
	for k in classes.keys():
		class_name = classes[k][0]
		class_directory = os.path.join(video_directory, class_name)
		videos = os.listdir(class_directory)
		videos = [ os.path.join(class_directory, v) for v in videos ]

		if smallest_class is None:
			smallest_class_num = len(videos)
			smallest_class = class_name
		elif len(videos) < smallest_class_num:
			smallest_class_num = len(videos)
			smallest_class = class_name

		# append the count to the class_indices
		classes[k].append(videos)
		print("%s - %s videos" % (class_name, len(videos)))

	print("smallest class = %s (%s videos)" % (smallest_class, smallest_class_num))
	
	# open each video and sample frames
	for k in classes.keys():
		class_name = classes[k][0]
		videos = classes[k][1]
		for v in videos:
			video_sample = sample_video(v, num_samples, sample_length, sample_randomly)


def sample_video(path, num_samples, sample_length, sample_randomly):
	"""samples a video given function arguments, returns a numpy array
	containing all three color channels, audio, and optical flow channel"""
	assert os.path.isfile(path), "unable to open %s" % (path)

	# sample the video
	v = video_file_to_ndarray(path, num_samples, sample_length, sample_randomly)


def video_file_to_ndarray(path, num_samples, sample_length, sample_randomly):
	"""returns an ndarray of samples from a video"""
	cap = cv2.VideoCapture(path)
	assert cap is not None, "Video capture failed for %s" % (path)

	# get metadata of video
	if hasattr(cv2, 'cv'):
		frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
		height = int(cap.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
		width = int(cap.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
		fps = float(cap.get(cv2.cv.CAP_PROP_FPS))
	else:
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		fps = float(cap.get(cv2.CAP_PROP_FPS))

	length = frame_count / fps
	print("%s length = %s" % (path, length))
	
	video_seconds = list(range(0, int(length)))
	if sample_randomly:
		sample_start_times = random_sample_start_times(video_seconds, num_samples, sample_length)
	else:
		for n in range(0, num_samples):
			sample_start_times.append(n * sample_length)

	print("%s - start times: %s" % (path, sample_start_times))

	# capture the video frames in 
	

def random_sample_start_times(seconds, num, length):
	"""samples the seconds list randomly, if the list is smaller
	than the number of samples compounded by length, the list is
	samples sequentially, looping back around to the beginning"""

	start_times = []

	if len(seconds[0::length]) > num:
		# generate a sample from the sample_seconds list
		start_times = random.sample(seconds[0::length], num)
		start_times.sort()
	else:
		# oversampling is needed
		index = 0
		while len(start_times) < num:
			start_times.append(seconds[index])
			index += 1
			if index == len(seconds):
				index = 0

	return start_times


def print_help():
	"""prints a help message"""
	print("Usage:\ntfrecord-gen.py <Dataset> <root directory>")


if __name__ == "__main__":
    main()
