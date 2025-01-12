
path = "/gemini/data-3/Hierarchical-Localization/datasets/car/video.mp4"

output_path = "/gemini/data-3/Hierarchical-Localization/datasets/car/images"

# convert mp4 to images
import cv2
import os
import sys
import pycolmap


def video2img(path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(path)
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_path, str(count).zfill(6) + '.jpg'), frame)
            count += 1
        else:
            break
    cap.release()
    print('Total frames: ', count)

video2img(path, output_path)
