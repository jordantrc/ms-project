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
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf

import c3d_model

from itertools import cycle

CLASS_INDEX_FILE = "UCF101-class-index.txt"
TRAIN_SET_FILE = "train-test-splits/trainlist01.txt"
TEST_SET_FILE = "train-test-splits/testlist01.txt"
DATA_SAMPLE = 1.0


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


def get_class_list(file):
    '''get the list of classes from the provided file'''
    classes = []
    with open(file) as class_index_file_fd:
        print("Opening %s" % (CLASS_INDEX_FILE))
        lines = class_index_file_fd.read().split('\n')
        # print(lines)
        for l in lines:
            if len(l) > 0:
                _, c = l.split(" ")
                classes.append(c)
    assert len(classes) > 0

    return classes


def integer_label(classes, label):
    # print("integer_label: label = %s, classes = %s" % (label, classes))
    int_label = -1
    for k, v in classes.items():
        # print("k = %s, v = %s" % (k, v))
        if v.lower() == label.lower():
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
            if len(l) > 0:
                space_split = l.split(' ')
                video = space_split[0]
                # print("video = %s" % video)
                video = video.split('/')[1]
                videos.append(video.strip())

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
    classes = get_class_list(CLASS_INDEX_FILE)

    # read the training and test list files to get the list of training
    # and test samples
    train_files = split_file_list(TRAIN_SET_FILE)
    test_files = split_file_list(TEST_SET_FILE)
    assert len(train_files) > 0 and len(test_files) > 0, "test/train file zero length"

    # print("train files: [%s]\n\ntest_files[%s]" % (train_files, test_files))

    # get count of each class
    smallest_class = None
    smallest_class_num = -1
    class_videos = {}
    for c in classes:
        class_directory = os.path.join(video_directory, c)
        videos = os.listdir(class_directory)
        videos = [os.path.join(class_directory, v) for v in videos]

        if smallest_class is None:
            smallest_class_num = len(videos)
            smallest_class = c
        elif len(videos) < smallest_class_num:
            smallest_class_num = len(videos)
            smallest_class = c

        class_videos[c] = videos
        print("%s - %s videos" % (c, len(videos)))

    print("smallest class = %s (%s videos)" % (smallest_class, smallest_class_num))

    # open each video and sample frames
    for c in classes:
        videos = class_videos[c]

        # sample videos based on DATA_SAMPLE
        videos = random.sample(videos, int(len(videos) * DATA_SAMPLE))

        for i, v in enumerate(videos):
            print("######\nProcessing %s [%d of %d]:\n" % (v, i, len(videos)))
            label, image_width, image_height, images = video_file_to_ndarray(v,
                                                                             num_samples,
                                                                             sample_length,
                                                                             sample_randomly,
                                                                             flip_horizontally,
                                                                             resize_height,
                                                                             resize_width)
            label_int = classes.index(label)
            print("label_int = %s, class name = %s" % (label_int, classes[label_int]))
            assert label_int >= 0

            featues = {}
            features['label'] = _int64_feature(label_int)

            # package up the frames from the video
            for i in range(images.shape[0]):
                frame = images[i]
                # frame = tf.image.encode_jpeg(frame, quality=100)
                frame_raw = frame.tostring()
                features['frames/{:04d}'.format(i)] = _bytes_feature(frame_raw)

            # generate clips from the video
            #clips = clips_from_video(images, c3d_model.FRAMES_PER_CLIP)
            #print("generated %s clips of size %s" % (len(clips), c3d_model.FRAMES_PER_CLIP))

            #for i, c in enumerate(clips):
            #    video_file_name = os.path.basename(v)
            #    clip_tfrecord_file_name = video_file_name + "_clip%02d.tfrecord" % i

            #    tfrecord_output_path = os.path.join(output, clip_tfrecord_file_name)
            #    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_output_path)
            #    assert tfrecord_writer is not None, "tfrecord_writer instantiation failed"

            #    features = dict()
            #    features['label'] = _int64_feature(label_int)

            #    # package up the frames from the clip
            #    for i in range(c.shape[0]):
            #        frame = c[i]
            #        # frame = tf.image.encode_jpeg(frame, quality=100)
            #        frame_raw = frame.tostring()
            #        features['frames/{:04d}'.format(i)] = _bytes_feature(frame_raw)

            example = tf.train.Example(features=tf.train.Features(feature=features))

            tfrecord_writer.write(example.SerializeToString())
            tfrecord_writer.close()
            print("images shape: %s, written to tfrecord file %s" % (c.shape, tfrecord_output_path))


def video_class(path):
    """returns the class of the video given the path"""
    filename = os.path.basename(path)
    parts = filename.split('_')
    classname = parts[1]
    return classname


def clips_from_video(frames, clip_size):
    '''takes the frames from a video and creates as many full-size
    clips as possible given the clip_size'''
    num_clips = int(math.floor(c3d_model.FRAMES_PER_VIDEO / clip_size))
    clips = []

    offset = 0
    for i in range(num_clips):
        clips.append(frames[offset:offset + clip_size])
        offset += clip_size

    return clips


def process_frame(frame, width, height, flip):
    '''performs pre-processing on a frame'''
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC) 
    if flip:
        image = cv2.flip(image, 1)
    image = image.astype(np.float32)

    return image


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
        fps = int(cap.get(cv2.cv.CAP_PROP_FPS))
    else:
        print("using cv2 for meta")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    print("video metadata:")
    print("frame_count = %d" % (frame_count))
    print("height = %d" % (height))
    print("width = %d" % (width))
    print("fps = %f" % (fps))
    print("length = %f" % (frame_count / fps))

    sample_size = int(fps * sample_length)
    frames_needed = c3d_model.FRAMES_PER_VIDEO
    buf = np.empty((frames_needed, resize_height, resize_width, 3), np.dtype('uint8'))

    # read as many frames as possible into a list
    frame_buffer = []
    success, image = cap.read()
    while success:
        frame_buffer.append(image)
        success, image = cap.read()

    # determine how many frames were captured
    frames_obtained = len(frame_buffer)

    frames_captured = 0
    if frames_obtained <= frames_needed:
        # turn the buffer into a circular list
        frame_buffer = cycle(frame_buffer)
        while frames_captured < frames_needed:
            image = process_frame(next(frame_buffer), resize_width, resize_height, flip_video)
            buf[frames_captured] = image
            frames_captured += 1

    else:
        indexes = list(range(frames_obtained))[::50]
        indexes = indexes[:-1]  # chop off the last index
        frame_index = 0
        capture = False
        while frames_captured < frames_needed:
            if frame_index in indexes:
                # start capturing
                capture = True
                sample_index = 0
            elif sample_index == 50:
                capture = False
                sample_index = 0

            if capture:
                image = process_frame(frame_buffer[frame_index], resize_width, resize_height, flip_video)
                buf[frames_captured] = image
                sample_index += 1
                frame_index += 1
                frames_captured += 1
            else:
                frame_index += 1

    assert frames_captured == frames_needed, "captured %s frames, needed %s" % (frames_captured, frames_needed)

    return video_class_name, width, height, buf


def print_help():
    """prints a help message"""
    print("Usage:\ntfrecord-gen.py <Dataset> <root directory> <tfrecord output directory>")


if __name__ == "__main__":
    main()
