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

CLASS_INDEX_FILE = "UCF101-class-index.txt"
TRAIN_SET_FILE = "train-test-splits/trainlist01.txt"
TEST_SET_FILE = "train-test-splits/testlist01.txt"


def main():
    # parse arguments

    if len(sys.argv) != 4:
        print_help()
        sys.exit(1)

    dataset_name = sys.argv[1]
    root_directory = sys.argv[2]
    output_directory = sys.argv[3]

    print("#####\nGot arguments:")
    print("dataset_name = %s" % dataset_name)
    print("root_directory = %s" % root_directory)
    print("output_directory = %s" % output_directory)

    if dataset_name == "UCF101":
        ucf101_dataset(root_directory, output_directory)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def integer_label(classes, label):
    # print("integer_label: label = %s, classes = %s" % (label, classes))
    int_label = -1
    for k, v in classes.items():
        # print("k = %s, v = %s" % (k, v))
        if v == label:
            int_label = int(k)
            break

    return int_label


def split_file_list(filepath):
    '''returns a list of the splits contained in the filepath file'''

    videos = []
    with open(filepath) as fd:
        text = fd.read()
        lines = text.split('\n')

        for l in lines:
            space_split = l.split(' ')
            video = space_split[0]
            video = video.split('/')[1]
            videos.append(video)

    return videos


def ucf101_dataset(root, output):
    """generates tfrecords for the ucf101 dataset"""
    # sampling information - number of samples of sample_length
    # in seconds
    sample_randomly = True
    num_samples = 5
    sample_length = 2

    # image resizing settings
    flip_horizontally = 0.5
    resize_height = 112
    resize_width = 112
    # crop_height = 112
    # crop_width = 112

    assert os.path.isdir(root) and os.path.isdir(output)
    video_directory = os.path.join(root, "UCF-101")

    # read class index file to get the class
    # index information
    classes = {}
    class_indexes = {}
    with open(CLASS_INDEX_FILE) as class_index_file_fd:
        print("Opening %s" % (CLASS_INDEX_FILE))
        lines = class_index_file_fd.read().split('\n')
        # print(lines)
        for l in lines:
            if len(l) > 0:
                i, c = l.split(" ")
                classes[i] = [c]
                class_indexes[i] = c
    assert len(classes) > 0

    # read the training and test list files to get the list of training
    # and test samples
    train_files = split_file_list(TRAIN_SET_FILE)
    test_files = split_file_list(TEST_SET_FILE)

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

    train_output_path = os.path.join(output, "train.tfrecord")
    test_output_path = os.path.join(output, "test.tfrecord")
    train_writer = tf.python_io.TFRecordWriter(train_output_path)
    test_writer = tf.python_io.TFRecordWriter(test_output_path)
    assert train_writer is not None and test_writer is not None

    # open each video and sample frames
    for k in classes.keys():
        class_name = classes[k][0]
        videos = classes[k][1]
        for i, v in enumerate(videos):
            print("######\nProcessing %s [%d of %d]:\n" % (v, i, len(videos)))
            video_file_name = os.path.basename(v)
            features = {}

            # get video data from the video
            label, image_width, image_height, images = video_file_to_ndarray(v,
                                                                             num_samples,
                                                                             sample_length,
                                                                             sample_randomly,
                                                                             flip_horizontally,
                                                                             resize_height,
                                                                             resize_width)
            images = images / 255.0
            images_raw = images.tostring()

            label_int = integer_label(class_indexes, label)
            assert label_int >= 0

            # TODO - write train/test splits to individual files - train.tfrecords, test.tfrecords
            features['label'] = _int64_feature(label_int)
            features['img_raw'] = _bytes_feature(images_raw)
            # features['height'] = _int64_feature(image_height)
            # features['width'] = _int64_feature(image_width)
            example = tf.train.Example(features=tf.train.Features(feature=features))
            if video_file_name in train_files:
                train_writer.write(example.SerializeToString())
                print("images shape: %s written to tfrecord file %s" % (images.shape, train_output_path))
            elif video_file_name in test_files:
                test_writer.write(example.SerializeToString())
                print("images shape: %s written to tfrecord file %s" % (images.shape, test_output_path))
            else:
                # error condition
                assert False, "video_file_name %s not found in test or train sets" % video_file_name

    train_writer.close()
    test_writer.close()


def video_class(path):
    """returns the class of the video given the path"""
    filename = os.path.basename(path)
    parts = filename.split('_')
    classname = parts[1]
    return classname


def video_file_to_ndarray(path, num_samples, sample_length, sample_randomly, flip, resize_height, resize_width):
    """returns an ndarray of samples from a video"""
    assert os.path.isfile(path), "unable to open %s" % (path)
    cap = cv2.VideoCapture(path)
    assert cap is not None, "Video capture failed for %s" % (path)

    # get class information for the video
    video_class_name = video_class(path)

    # determine if video should be flipped
    flip_video = False
    if random.random() <= flip:
        flip_video = True

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

    buf = np.empty((int(fps * sample_length * num_samples), resize_height, resize_width, 3), np.dtype('uint8'))
    in_second = 0
    count = 0
    sequence = 0
    max_sequence = (fps * sample_length * num_samples) - 1
    while success:
        if in_second in sample_times:
            # image transformations - resize to 224x224, convert to float
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
            if flip_video:
                image = cv2.flip(image, 0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

            if oversample:
                num_samples = sample_times.count(in_second)
                indices = [i for i, x in enumerate(sample_times) if x == in_second]
            else:
                num_samples = 1

            if num_samples == 1:
                # cv2.imwrite("./frames/frame%d-%d.jpg" % (sequence, count), image)
                # print("image = %s, shape = %s" % (image, image.shape))
                if sequence <= max_sequence:
                    buf[sequence] = image
                sequence += 1
            else:
                for s in range(num_samples):
                    index = indices[s]
                    sequence = int((fps * index) + (count % fps))
                    if sequence <= max_sequence:
                        buf[sequence] = image
                    # print("index = %s sequence = %s" % (index, sequence))
                    # print("frame index = %s" % frame_index)
                    # cv2.imwrite("./frames/frame%d-%d.jpg" % (sequence, count), image)

        success, image = cap.read()
        # print('read new frame: ', success)
        count += 1
        in_second = int(count / fps)

    return video_class_name, width, height, buf


def print_help():
    """prints a help message"""
    print("Usage:\ntfrecord-gen.py <Dataset> <root directory> <tfrecord output directory>")


if __name__ == "__main__":
    main()
