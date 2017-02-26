import numpy as np
import cv2
from matplotlib import pyplot as plt
from decimal import Decimal
import math
import copy
import argparse
import glob
import multiprocessing


# Input image frame (image now)
#Using single frame to test functioning
image = cv2.imread('frame1.png')

height = np.size(image, 0)
width = np.size(image, 1)

angle = 45 # default angle tolerance

print "BEFORE RESIZING IMAGE: "
print "height: ", height, " width: ", width

# SETTING IMAGE, RESOLUTION: 1920 x 1024
image_1 = cv2.resize(image, (1920, 1024))

height_1 = np.size(image_1, 0)
width_1 = np.size(image_1, 1)

print " AFTER RESIZING IMAGE: "
print "height: ", height_1, "width: ", width_1

# INPUT IMAGE
cv2.imshow("Figure: Input Image", image)


# ELIMINATING TOP HALF, HARD CODED HACK (1024/2)
horizontal = 512
image_1[0:horizontal, :] = 0

# ELIMINATING LEFT HALF, FOCUSING ON RIGHT LANE (1920/2)
vertical = 960
image_1[:, 0:vertical] = 0

# TRIMMED IMAGE
#cv2.imshow("Figure: Trimmed Image", image_1)


# applying Gaussian Blur
img_array = image_1.astype(np.uint8)

im_filter = np.zeros(shape=img_array.shape)

im_filter = cv2.GaussianBlur(img_array, (5, 5), math.sqrt(2),math.sqrt(2))

#img_edges = cv2.Canny(im_filter, 0.5, 0.3, 1)

# CALCULATING EDGE GRADIENTS

scale = 1
delta = 0
ddepth = cv2.CV_64F
im_filter_1 = im_filter[:,:,1]

# x gradient
sobel_x = cv2.Sobel(im_filter_1,ddepth,1,0,ksize = 5, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)


cv2.imshow("Figure: X Sobel Edges", sobel_x)
# y gradient
sobel_y = cv2.Sobel(im_filter_1,ddepth,0,1,ksize = 5, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)


magnitude = np.zeros(shape=image_1.shape)



cv2.imshow("Figure: Y Sobel Edges", sobel_y)
for row_idx in range(0,1024):
    for col_idx in range(0,1920):
        magnitude[row_idx,col_idx] = math.sqrt((sobel_x[row_idx, col_idx]**2) + (sobel_y[row_idx, col_idx]**2))

print "magnitude", magnitude
cv2.imshow("Magnitude values ", magnitude )

abs_grad_x = cv2.convertScaleAbs(sobel_x)   # converting back to uint8
abs_grad_y = cv2.convertScaleAbs(sobel_y)

angles = np.zeros(shape=image_1.shape)

#angles.append(math.atan2(sobel_y, sobel_x))

for row_idx in range(0,1024):
    for col_idx in range(0,1920):
        angles[row_idx,col_idx] =(math.atan2(sobel_y[row_idx, col_idx], sobel_x[row_idx, col_idx]) + (math.pi/2))/ math.pi

print "angles", angles


cv2.imshow("Angles Computed -- first attempt ", angles )





total_gradient = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)

print "total_gradient: ", total_gradient



#Applying hough lines

minLineLength = 70
maxLineGap = 0.1

lines = cv2.HoughLinesP(total_gradient, 1, np.pi/180, 5, minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(image_1, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)


lines = cv2.HoughLines(total_gradient,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)


#for var in range(-135, 150, 45):






cv2.imshow("Figure: Sobel Edges", total_gradient)
cv2.imshow("Figure: Frame", image)
cv2.waitKey(0)








