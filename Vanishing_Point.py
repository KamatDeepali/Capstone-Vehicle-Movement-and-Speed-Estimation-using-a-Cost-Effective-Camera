import numpy as np
import cv2
import argparse
import glob
from matplotlib import pyplot as plt
from decimal import Decimal
import math
import copy

# Input image frame
# can perform on each frame in video
img = cv2.imread('frame1.png')
im_double = img.astype('float64')

im_edges = np.zeros(shape=im_double.shape)

# Applying a Sobel filter
im_horiz = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # Applying Sobel filter in x-axis direction
im_vert = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)   # Applying Sobel filter in y-axis direction

# Calculate magnitude and direction of edges
magnitude = math.sqrt(im_horiz**2 + im_vert**2)
direction = math.atan2(im_vert, im_horiz) * 180 / math.pi   # in degree

#the magnitude of the green channel
green_mag = magnitude[:, :, 1]

dims = green_mag.shape

# vanishing point votes
vote_data = np.zeros(shape=dims)

x = np.arrange(0, dims[1], 1)
y = np.arrange(0, dims[0], 1)
xv, yv = np.meshgrid(x, y)

# for each of the main angles
for angle in range(-135, 150, 45):

    # min and max value of angles with respect to the delta angle(45/2)
    min_angle = angle - 22.5
    max_angle = angle + 22.5


    angles_x = direction[:, :, 1]
    angles_x[angles_x <= min_angle] = 0
    angles_x[angles_x >= max_angle] = 0

    edge_points = angles_x if angles_x is not 0 else 1

    # magnitude for points of interest for selected angles
    image_mag = angles_x(edge_points)

    # assuminng relevant edges to be in top 5%
    # eliminate rest
    [all_values, sk] = np.sort(image_mag[:])
    [n1, n2] = np.size(all_values)
    n = math.floor((0.95*n1*n2)+0.5)

    top_thresh_id = n

    top_thresh_value = all_values(top_thresh_id)

    im_mag_grn_big = np.zeros(shape=image_mag.shape);

    # if  pixels are at correct angle, form boolean value
    # values > 'top_thresh_value' magnitude ~ edge strength
    selected_edge_pts = (im_mag_grn_big > 0) & image_mag;

    minLineLength = 200
    maxLineGap = 0.1
    lines = cv2.HoughLinesP(selected_edge_pts, 1, np.pi / 180, 10, minLineLength, maxLineGap)

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)

        x = np.arrange(0, dims[1], 1)
        y = np.arrange(0, dims[0], 1)
        xs, ys = np.meshgrid(x, y)

        denom = abs(math.sqrt((y2 - y1)** 2 + (x2 - x1)** 2));

        dy = (y2 - y1);
        dx = (x2 - x1);
        const = (y2 * y1) - (x2 * x1);

        dists = abs(dy * xs - dx * ys + (const)) / denom;

        b_vote_map = (dists <= 1.5);

        vote_data[b_vote_map] = vote_data[b_vote_map] + denom;

    cv2.imshow("Votes", vote_data)
    cv2.namedWindow("Figure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Figure", 1920, 1024)
    #cv2.imshow("Figure", im_edges)
    #cv2.waitKey(1)

    kernel = np.ones((5, 5), np.float64) / 25
    disk= np.zeros(shape=vote_data.shape)
    cv2.filter2D(vote_data, (-1/cv2.CV_64F), kernel, disk,(-1,-1), 0, cv2.BORDER_DEFAULT)

    mmax = np.amax(disk)
    [col, row] = disk == mmax

    cv2.imshow("Vanishing point", disk)
    cv2.namedWindow("Figure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Figure", 1920, 1024)
    cv2.imshow("Figure", im_edges)
    cv2.waitKey(1)






























