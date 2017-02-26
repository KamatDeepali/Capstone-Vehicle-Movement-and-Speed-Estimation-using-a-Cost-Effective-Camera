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




video_file = ('/Users/deepaliKamat/Desktop/Monroe_to_Hylan.AVI')
cap = cv2.VideoCapture(video_file)
#length_of_file = int(cap.get(20))
frame_counter = 0
outputFrameIndices=[]

queue = Queue()


while(True):

    frame_counter = frame_counter + 1
    ret, frame1 = cap.read()  # read current frame
    outputFrameIndices.append(frame_counter)

    startFrame = frame_counter
    endFrame = startFrame + 20
    sum_of_diff = 0
    list_diff =[]

    for frame in range(startFrame, endFrame):
        #set next frame to start frame
        cap.set(1, frame)
        frame_num = int(cap.get(0))
        queue.put((frame_num, frame1))

        rows = np.size(frame, 0)
        columns = np.size(frame, 1)


        frame = frame1
        # setting first  and last 300 rows to zero
        frame[0:300, :] = 0
        end_count = rows-300
        frame[end_count:rows, :] = 0

        # setting left and right width of 320 pixels to zero
        frame[:, 0:columns] = 0
        end_column = columns-320
        frame[:, end_column:columns] = 0

        sum_of_diff = sum_of_diff + (frame1-frame)
        print "sum of differences: ", sum_of_diff

        list_diff.append(sum_of_diff)


















