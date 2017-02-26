import numpy as np
import cv2
import argparse
import glob
from matplotlib import pyplot as plt
from decimal import Decimal
import math
import copy


###############################################################################
#
#    MAIN
#
###############################################################################

my_window_name = "Window"

frame1 = cv2.imread('frame1.png')

frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)

im_bgr = np.zeros(shape=frame.shape)
# changing image to double precision
cv2.normalize(frame.astype('float64'), im_bgr, 0, 1, cv2.NORM_MINMAX)

# Conversion to possible color spaces

im_r2gb = (im_bgr[:, :, 0] + 2.0 * im_bgr[:, :, 1] + im_bgr[:, :, 2]) / 4.0

q_fact = 2

#
#  Gaussian blur, using a Kernel that is 7 across and 3 high.
#  Blur for a distance of 1 pixel, nominally.
#
im_filter = cv2.GaussianBlur(im_r2gb, (7, 3), 1)
cv2.imshow(my_window_name, im_filter)

horiz = 500; # harcoded HACK


#
#  Make a deep copy and manually threshold the image:
#
im_segmented = im_r2gb.copy()

# Trim off the horizon values:
print "about to trim off hte horizontal values"
im_segmented[0:horiz, :] = 0

print "Get the size of the segmented image... "

height = np.size(im_segmented, 0)
width = np.size(im_segmented, 1)

print "width = ", width, "     height = ", height

im_r2gb[im_r2gb > 0.5] = 1
im_r2gb[im_r2gb <= 0.5] = 0

im_segmented = copy.deepcopy(im_r2gb)
# Blur the image before running canny on it:

im_filtered = cv2.GaussianBlur(im_segmented, (7,3), 1)
im_arr = im_filtered.astype(np.uint8)
im_edges = cv2.Canny(im_arr, 0.8, 0.5, 1)

minLineLength = 200
maxLineGap = 0.1
lines = cv2.HoughLinesP(im_edges, 1, np.pi/180, 50, minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(frame1, (x1, y1), (x2, y2), (0, 255, 0), thickness=7, lineType=8, shift=0)


# cv2.startWindowThread()
cv2.imshow("Main_Frame", frame1)
cv2.namedWindow("Figure", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Figure", 1920, 1024)
cv2.imshow("Figure", im_edges)
cv2.waitKey()

