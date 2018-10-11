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
import numpy as np
import os
import random
import sys
import tensorflow as tf


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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def ucf101_dataset(root, output):
    """generates tfrecords for the ucf101 dataset"""
    # sampling information - number of samples of sample_length
    # in seconds
    sample_randomly = True
    num_samples = 5
    sample_length = 2

    # image resizing settings
    flip_horizontally = 0.5
    resize_height = 128 
    resize_width = 171
    crop_height = 112 
    crop_width = 112


    assert os.path.isdir(root) and os.path.isdir(output)
    class_index_file = os.path.join(root, "classInd.txt")
    video_directory = os.path.join(root, "UCF-101")

    # read class index file to get the class
    # index information
    classes = {}
    with open(class_index_file) as class_index_file_fd:
        print("Opening %s" % (class_index_file))
        lines = class_index_file_fd.read().split('\n')
        # print(lines)
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
        videos = [os.path.join(class_directory, v) for v in videos]

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
            print("######\nProcessing %s:\n" % v)
            features = {}
            output_path = os.path.join(output, v + ".tfrecord")
            print("output_path = %s" % output_path)
            writer = tf.python_io.TFRecordWriter(output_path)
            assert writer is not None

            # get video data from the video
            video_data = video_file_to_ndarray(v, num_samples, sample_length, sample_randomly)
            image_width = video_data[1]
            image_height = video_data[2]
            images_raw = video_data[3].tostring()

            features[v] = _bytes_feature(images_raw)
            features['height'] = _int64_feature(image_height)
            features['width'] = _int64_feature(image_width)
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            print("done writing data to tfrecord file")


def video_class(path):
    """returns the class of the video given the path"""
    filename = os.path.basename(path)
    parts = filename.split('_')
    classname = parts[1]
    return classname


def video_file_to_ndarray(path, num_samples, sample_length, sample_randomly):
    """returns an ndarray of samples from a video"""
    assert os.path.isfile(path), "unable to open %s" % (path)
    cap = cv2.VideoCapture(path)
    assert cap is not None, "Video capture failed for %s" % (path)

    # get class information for the video
    video_class_name = video_class(path)

    # get metadata of video
    if hasattr(cv2, 'cv'):
        print("using cv2.cv for meta")
        frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        fps = float(cap.get(cv2.cv.CAP_PROP_FPS))
    else:
        print("using cv2 for meta")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

    print("video metadata:")
    print("frame_count = %d" % (frame_count))
    print("height = %d" % (height))
    print("width = %d" % (width))
    print("fps = %d" % (fps))
    print("length = %f" % (frame_count / fps))

    video_seconds = list(range(0, int(frame_count / fps)))
    sample_times = []
    oversample = False

    if len(video_seconds[::sample_length]) > num_samples:
        sample_times = random.sample(video_seconds[::sample_length], num_samples)
        add_times = []
        for s in sample_times:
            for i in range(1, sample_length):
                add_times.append(s + i)
        sample_times.extend(add_times)
        sample_times.sort()
    else:
        # oversampling is needed
        oversample = True
        index = 0
        while len(sample_times) < (num_samples * sample_length):
            sample_times.append(video_seconds[index])
            index += 1
            if index == len(video_seconds):
                index = 0

    print("sample_times = %s" % (sample_times))
    success, image = cap.read()

    buf = np.empty((int(fps * sample_length * num_samples), height, width, 3), np.dtype('uint8'))
    in_second = 0
    count = 0
    sequence = 0
    while success:
        if in_second in sample_times:
            # image transformations - resize to 224x224, convert to float
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

            if oversample:
                num_samples = sample_times.count(in_second)
                indices = [i for i, x in enumerate(sample_times) if x == in_second]
            else:
                num_samples = 1

            if num_samples == 1:
                # cv2.imwrite("./frames/frame%d-%d.jpg" % (sequence, count), image)
                buf[sequence] = image
                sequence += 1
            else:
                for s in range(num_samples):
                    index = indices[s]
                    sequence = int((fps * index) + (count % fps))
                    buf[sequence] = image
                    # print("index = %s sequence = %s" % (index, sequence))
                    # print("frame index = %s" % frame_index)
                    # cv2.imwrite("./frames/frame%d-%d.jpg" % (sequence, count), image)

        success, image = cap.read()
        # print('read new frame: ', success)
        count += 1
        in_second = int(count / fps)

    print("np buffer shape, sample data:")
    print(buf.shape)
    print(buf[0])

    return [video_class_name, width, height, buf]


def print_help():
    """prints a help message"""
    print("Usage:\ntfrecord-gen.py <Dataset> <root directory> <tfrecord output directory>")


if __name__ == "__main__":
    main()
