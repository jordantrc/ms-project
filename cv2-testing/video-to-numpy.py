

from __future__ import print_function

import cv2


samples = 5
sample_length = 2

video_file = "delete.avi"

cap = cv2.VideoCapture(video_file)
success, image = cap.read()

count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)
    success, image = cap.read()
    print('read new frame: ', success)
    count += 1
