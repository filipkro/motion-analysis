import cv2
import numpy as np

file_name = '/home/filipkr/Downloads/vid(3).mp4'
# file_name = '03SLS1R_MUSSE.mp4'

cap = cv2.VideoCapture(file_name)
frame = 1
while cap.isOpened():
    flag, img = cap.read()
    if not flag:
        break

    print(np.shape(img))
    frame += 1
    print(frame)
