

from __future__ import print_function

import cv2
import numpy as np
import random
import sys


np.set_printoptions(threshold='nan')

samples = 5
sample_length = 2

video_file = sys.argv[1]

cap = cv2.VideoCapture(video_file)

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

if len(video_seconds[::sample_length]) > samples:
    sample_times = random.sample(video_seconds[::sample_length], samples)
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
    while len(sample_times) < (samples * sample_length):
        sample_times.append(video_seconds[index])
        index += 1
        if index == len(video_seconds):
            index = 0

print("sample_times = %s" % (sample_times))
success, image = cap.read()

buf = np.empty((int(fps * sample_length * samples), height, width, 3), np.dtype('uint8'))

in_second = 0
count = 0
sequence = 0
while success:
    if in_second in sample_times:
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
                cv2.imwrite("./frames/frame%d-%d.jpg" % (sequence, count), image)

    success, image = cap.read()
    # print('read new frame: ', success)
    count += 1
    in_second = int(count / fps)

print("np buffer shape, sample data:")
print(buf.shape)
print(buf[0])
