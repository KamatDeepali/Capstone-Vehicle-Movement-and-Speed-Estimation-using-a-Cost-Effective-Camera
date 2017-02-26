import numpy as np
import cv2
import cv
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

cap = cv2.VideoCapture('/Volumes/CORP/ROAD_VIDS/08390005_road_stripes.AVI');

while(True):

    ret, frame1 = cap.read()
    #frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
    frame = frame1
    im_bgr = np.zeros(shape=frame.shape)
    # changing image to double precision
    # cv2.normalize(frame.astype('float64'), im_bgr, 0, 1, cv2.NORM_MINMAX)
    # Professor kinsman klutzing in...
    #print "frame  is of type: ", type(frame[0])
    im_bgr = frame
    #print "im_bgr is of type: ", type(im_bgr)


    # Conversion to possible color spaces

    # im_r2gb = (im_bgr[:, :, 0] + 2.0 * im_bgr[:, :, 1] + im_bgr[:, :, 2]) / 4.0
    im_r2gb = 1.0 * im_bgr[:,:,1]   # copy.deepcopy(im_bgr[:, :, 1])

    q_fact = 2

    #
    #  Gaussian blur, using a Kernel that is 7 across and 3 high.
    #  Blur for a distance of 1 pixel, nominally.
    #
    #im_filter = cv2.GaussianBlur(im_r2gb, (7, 3), 1)
    im_filter = im_r2gb
    #cv2.imshow(my_window_name, im_filter)

    horiz = 500; # harcoded HACK

    #
    #  Make a deep copy and manually threshold the image:
    #
    im_segmented = im_r2gb.copy()

    # Trim off the horizon values:
    print "about to trim off the horizontal values"
    # Temporarily not removing the clouds...
    im_segmented[0:horiz, :] = 0

    print "Get the size of the segmented image... "

    height = np.size(im_segmented, 0)
    width  = np.size(im_segmented, 1)

    print "width = ", width, "     height = ", height

    cv2.imshow("Before thresholding image", frame)
    cv2.namedWindow("Before thresholding image", cv2.WINDOW_NORMAL)
    cv2.waitKey(1)

    im_thresh = np.zeros(shape=im_r2gb.shape)
    #thresh_temp = cv2.threshold(im_r2gb.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    # cv2.adaptiveThreshold(im_r2gb, im_thresh, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, (7,7), 0)
    #thresh_temp = cv2.ADAPTIVE_THRESH_GAUSSIAN_C(im_r2gb, (7,3))


    # thresh = np.average(im_r2gb)*(2.0/3.0)
    # print "threshold value = ", thresh
    # Threshold the image to try to separate the white lines:
    thresh_temp = 95 # ; 95
    im_r2gb[im_r2gb > thresh_temp] = 255
    im_r2gb[im_r2gb <= thresh_temp] = 0

    cv2.imshow("After thresholding image", im_r2gb)
    cv2.namedWindow("After thresholding image", cv2.WINDOW_NORMAL)
    cv2.waitKey(1)

    #im_segmented = copy.deepcopy(im_r2gb)
    im_segmented = im_r2gb
    # Blur the image before running canny on it:

    # im_filtered = cv2.GaussianBlur(im_segmented, (7,3), 1)
    im_filtered = im_segmented  # Bypass...

    im_arr = im_filtered.astype(np.uint8)
    # Python: cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
    im_edges = cv2.Canny(im_arr, 0.6, 0.125, 1)

    minLineLength = 200
    maxLineGap = 25
    # cv2.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn]]])
    lines = cv2.HoughLinesP(im_edges, 25, (np.pi*15)/180, 1, 100) # minLineLength, maxLineGap)

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(frame1, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)


    #lines = cv2.HoughLines(im_edges, 1, np.pi / 180, 100)
    '''
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''
    # cv2.startWindowThread()
    cv2.imshow("Main_Frame", frame1)
    cv2.namedWindow("Figure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Figure", 1920, 1024)
    cv2.imshow("Figure", im_edges)
    cv2.waitKey(1)

    print("hit a key to continue\n")
    cv2.waitKey(0)


# After completion, release capture
#cap.release()
#cv2.destroyAllWindows()
#cv2.destroyAllWindows()

