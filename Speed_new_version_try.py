import numpy as np
import cv2
from multiprocessing import Process, Queue
from Queue import Empty
from matplotlib import pyplot as plt
from decimal import Decimal
import math
import copy
import argparse
import glob
import multiprocessing



frameCount = 20

#GLOBAL FRAME ARRAY
frameQueue = [20]

def speed_Estimation(currentFrame_index, currentFrame):
    im_height= np.size(currentFrame, 0)
    im_width = np.size(currentFrame, 1)
    bestDiff_yet = 255*im_width*im_height

    for offset in range(4, 20):
        compare_index = np.mod(currentFrame_index - offset, 20)

        im_A = frameQueue[:, :, currentFrame_index]
        im_B = frameQueue[:, :, compare_index]
        this_difference = sum(abs(im_A - im_B))


        if this_difference < bestDiff_yet:
            bestDiff_yet = this_difference
            best_offset = offset


video_file = ('Road_Video.AVI')
cap = cv2.VideoCapture(video_file)



for frame in range(frameCount):
    ret, frameQueue[frame] = cap.read()

frameCount = 0

for frames in range(21, 100, 20):
    currentFrame = frame_mod(frameNumber, QueueSize)
    frames[currentFrame] = read_Frame(frameNumber)
