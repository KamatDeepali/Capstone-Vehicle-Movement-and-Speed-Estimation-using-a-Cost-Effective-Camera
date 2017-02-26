import numpy as np
import cv2
import argparse
import glob
from matplotlib import pyplot as plt
from decimal import Decimal
import math


def img_quant(im_dbl, ttl, quant) :

    '''
    plt.figure()
    plt.subplot(2, 2, 1)
    cv2.imshow("frame",im_dbl)
    plt.title(ttl)
    '''
    # Quantize image and show planes:
    # im_quant = np.divide(round(im_dbl * (quant - 1)), (quant - 1))

    im_quant = np.ndarray.round(im_dbl * (quant-1))/(quant - 1)

    '''
    plt.subplot(2, 2, 2)

    if(len(im_dbl)) == 3 :
        cv2.imshow("frame", im_quant[:,:, 1])
    else :
        cv2.imshow("frame", im_quant)


    ttl_b = 'Quantized by {0}'.format(quant)
    plt.title(ttl_b)

    if (len(im_dbl) == 3):
        plt.subplot(2, 2, 3)
        cv2.imshow(im_quant[:,:, 2])
        plt.title([ttl(2), ' '])
        plt.subplot(2, 2, 4)
        cv2.imshow(im_quant[:,:,3])
        plt.title([ttl(3), ' '])

    '''
# end of img_quant.




def show_img_in_colorspace(im_dbl) :

    xs = im_dbl[:,:, 0]
    ys = im_dbl[:,:, 1]
    zs = im_dbl[:,:, 2]

    #plt.figure('Position', [30, 4, 1024, 768])
    plt.plot(xs[1:5:-1], ys[1:5:-1], zs[1:5:-1], 'k.')
    plt.show()
# end of function.






###############################################################################
#
#    MAIN
#
###############################################################################

my_window_name = "Window"
# single frame test
frame = cv2.imread('frame1.png')


im_bgr = np.zeros(shape=frame.shape)
# changing image to double precision
cv2.normalize(frame.astype('float64'), im_bgr, 0, 1, cv2.NORM_MINMAX)

# Conversion to possible color spaces

im_r2gb = (im_bgr[:, :, 0] + 2.0 * im_bgr[:, :, 1] + im_bgr[:, :, 2]) / 4.0
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

# Normalizing LAB
lab_frame[:, :, 0] = np.divide(lab_frame[:, :, 0], 100.0)

a_star_min = -100
a_star_max = 100
lab_frame[:, :, 1] = (lab_frame[:, :, 1] - a_star_min) / (a_star_max - a_star_min)

b_star_min = -100
b_star_max = 100
lab_frame[:, :, 2] = (lab_frame[:, :, 2] - b_star_min ) / (b_star_max - b_star_min)

ycbcr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
im_ymb = im_bgr[:, :, 2] + im_bgr[:,:, 1] - 2 * im_bgr[:, :, 0]
im_mmg = im_bgr[:, :, 0] - 2 * im_bgr[:, :, 1] + im_bgr[:, :, 2]

q_fact = 2

#
#  Gaussian blur, using a Kernel that is 7 across and 3 high.
#  Blur for a distance of 1 pixel, nominally.
#
im_filter = cv2.GaussianBlur(im_r2gb, (7, 3), 1)
cv2.imshow(my_window_name, im_filter)

img_quant(im_filter, 'RGB', q_fact)


'''
#TRIAL : http://answers.opencv.org/question/13989/calchist-usage-for-rgb-histogram/

imgCount = 1
dims = 3
sizes = [256,256,256]
channels= [0,1,2]
rRange = {0,256}
gRange = {0,256}
bRange = {0,256}
ranges = [rRange, gRange, bRange];
#hist_new= cv2.calcHist(im_r2gb, channels, dims, sizes, ranges);


plt.figure(1)
#hist_new = cv2.calcHist(im_r2gb, 0, None, [255], [0, 255])
#plt.plot(hist_new)

thresh = cv2.THRESH_BINARY(im_r2gb)
print("thesh = ", thresh)
# best segmentation calculation along axis dimension
hist_proj = np.sum(im_r2gb, 1)

plt.subplot(2, 2, 4)
cv2.imshow(im_r2gb)
# xs = hist_proj
# ys = 1:size(im_r2gb.size, 1)
# plt.plot(xs(:), ys(:),'ro-')

dims = (im_r2gb.size)

horiz = 500; # harcoded HACK

plt.plot([1, dims(2)], [1, 1] * horiz, 'c-', 'LineWidth', 3)

im_segmented = cv2.cvtColor(im_r2gb,thresh)

#eliminating the sky
im_segmented[0:horiz, :] = 0
cv2.imshow(im_segmented)
'''

horiz = 500; # harcoded HACK

print "about to try thresholding the image... "

#
#  Make a deep copy and manually threshold the image:
#
im_segmented = im_r2gb.copy()

# Trim off the horizon values:
print "about to trim off hte horizontal values"
im_segmented[0:horiz, :] = 0

print "im_segmented\n", im_segmented
print "Get the size of the image... "

height = np.size(im_segmented, 0)
width = np.size(im_segmented, 1)

print "width = ", width, "     height = ", height

MY_THRESHOLD = 0.5;

for row_id in range(0, height ) :
    for col_id in range(0, width ) :
        if ( im_segmented[row_id,col_id] < MY_THRESHOLD) :
            im_segmented[row_id,col_id] = 0.0;
        else :
            im_segmented[row_id,col_id] = 1.0;

print "done looping over the image... "
print "Displaying the image..."
#cv2.imshow( "SEGMENTED IMAGE", im_segmented )



im_r2gb[im_r2gb > 0.5] = 1
im_r2gb[im_r2gb <= 0.5] = 0



#ret, thresh = cv2.threshold(im_segmented, 0.5, 1, cv2.THRESH_BINARY)
#print "ret =", ret
#print " thresh = ", thresh

im_segmented = im_r2gb.copy()
print "the deep copy worked..."

#cv2.imshow("frame1", im_segmented)

# Blur the image before running canny on it:

im_filtered = cv2.GaussianBlur(im_segmented, (7,7), 1)
im_arr =  im_filtered.astype(np.uint8)
im_edgs = cv2.Canny(im_arr, 0.5, 0.3, 1)

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(im_edgs,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(im_edgs,(x1,y1),(x2,y2),(0,255,0),2)


cv2.startWindowThread()
cv2.namedWindow("hello", cv2.WINDOW_NORMAL)
cv2.resizeWindow("hello", 620, 480)
cv2.imshow("hello", im_edgs)
#cv2.waitKey()

cv2.imshow("cannyFrame", im_edgs)
img_quant(frame,'RGB',q_fact)
show_img_in_colorspace(frame)

img_quant(hsv_frame, 'HSV', q_fact)
show_img_in_colorspace(hsv_frame)

img_quant(lab_frame, 'LAB ', q_fact)
show_img_in_colorspace(lab_frame)

#img_quant(ycbcr_frame, 'YCC ', n_quants)
#show_img_in_colorspace(ycbcr_frame)

