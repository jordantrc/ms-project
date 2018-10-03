

from __future__ import print_function

import cv2
import random
import sys


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
start_times = []
oversample = False

if len(video_seconds[::sample_length]) > samples:
    start_times = random.sample(video_seconds[::sample_length], samples)
    start_times.sort()
else:
    # oversampling is needed
    oversample = True
    index = 0
    while len(start_times) < samples:
        start_times.append(video_seconds[index])
        index += 1
        if index == len(video_seconds):
            index = 0

print("start_times = %s" % (start_times))
success, image = cap.read()

in_second = 0
count = 0
while success:
    if in_second in start_times:
        if oversample:
            num_samples = start_times.count(in_second)
            indices = [i for i, x in enumerate(start_times) if x == in_second]
        else:
            num_samples = 1

        if num_samples == 1:
            cv2.imwrite("./frames/frame%d-%d.jpg" % (in_second, count), image)
        else:
            for s in range(num_samples):
                index = indices[s]
                frame_index = in_second + index + s
                # print("frame index = %s" % frame_index)
                cv2.imwrite("./frames/frame%d-%d.jpg" % (frame_index, count), image)

    success, image = cap.read()
    # print('read new frame: ', success)
    count += 1
    in_second = int(count / fps)
