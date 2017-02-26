import sys  # For getsizeof
import numpy as np  # Numpy for math.
import cv2  # OpenCV2
import time  # For time.sleep( )

# cap = cv2.VideoCapture(0)  	#  Or, could open up a video here.
# cap = cv2.VideoCapture('/Users/tbk/TBK_PROF/CS_631/VIDEOS_of_ROAD/2016_06_28_10360630/08390006_road_stripes.AVI')
cap = cv2.VideoCapture('Road_Video.AVI')

# Params for ShiTomasi corner detection:
#
# REMEMBER: the image has been through motion compression, which loses information.
#
# maxCorners is 600, so at most 400 points are pulled out of the image.
#
# minDistance seems to be a limit that keeps points from being any closer together than this.
# So, if this is set to 9, then no blocks will be closer than 9 pixels together.
#
# Block size seems to be the size of the block to consider.
#
# A motion compression block is 16x16, so set the block size to twice that, plus one.
#                       blockSize    = (16*2+1) )
feature_params = dict(maxCorners=600,
                      qualityLevel = 0.05,
                      #qualityLevel=0.00625,
                      minDistance=(8 + 1),
                      blockSize=(16 * 2 + 1))
# qualityLevel = 0.075,

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(63, 63),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 21, 0.001))

# Warm up the camera and let auto-exposure stabilize.
# Take first frame and find corners in it.
for jj in range(0, 5):
    ret, frame_0_clr = cap.read()

frame_0_grn = frame_0_clr[:, :, 1]
p0 = cv2.goodFeaturesToTrack(frame_0_grn, mask=None, **feature_params)
cv2.imshow('MyWindow', frame_0_clr)

p0 = cv2.goodFeaturesToTrack(frame_0_grn, mask=None, **feature_params)
cv2.imshow('MyWindow', frame_0_clr)

n_frames_these_pts = 1;
while (1):
    # Create a mask image for drawing purposes.
    # We will draw all the lines onto the mask, and then add that
    # mask onto the green frame, before displaying the colored image.
    mask = np.zeros_like(frame_0_grn)

    ## print "\nFrame number ", n_frames_these_pts
    ret, frame_1_clr = cap.read()  # SKIPPING A FRAME
    ret, frame_1_clr = cap.read()  # SKIPPING 2 FRAMES
    ret, frame_1_clr = cap.read()  # SKIPPING 3 FRAMES
    ret, frame_1_clr = cap.read()  # SKIPPING 4 FRAMES
    ret, frame_1_clr = cap.read()  # Get a new frame.
    frame_1_grn = frame_1_clr[:, :, 1].copy();  # Copy the green plane,
    # because we are going to change it later.
    cv2.imshow('MyWindow', frame_1_clr)  # Display the color image.

    # calculate optical flow
    p1 = 1.0 * p0;  # Default is no motion.
    p1, status, err = cv2.calcOpticalFlowPyrLK(frame_0_grn, frame_1_grn, p0, p1, None, **lk_params)

    print("P1     contains  ", len(p1), "points.")
    ## print "P1     has shape ", p1.shape
    # print "Status contains  ", len(status), "points."
    # print "Status has shape ", status.shape


    #
    #  DRAW THE MOTION VECTORS:
    #
    last_valid_idx = min([len(p1), len(p0)]) - 1

    # print "last_valid_idx = ", last_valid_idx
    bad_pts = [];
    for idx in range(0, last_valid_idx):
        # print "status[", idx, "] = ", status[idx]

        pt1 = p0[idx]  # This is an array of arrays.
        pt1B = pt1[0]  # This gets the actual values themselves.
        x1 = pt1B[0]  # The associated X location for point 1.
        y1 = pt1B[1]

        pt2 = p1[idx]
        pt2B = pt2[0]
        x2 = pt2B[0]  # The associated X location for point 2.
        y2 = pt2B[1]  # The associated Y location for point 2.

        the_color = [255, 255, 255]  # BGR ...
        if mask is None:
            print("Mask is None")
        else:
            # The point can be not tracked correctly, so always check the status.
            # For an example of a bad status, cover the camera lens temporarily.  :-)
            if status[idx] != 0:
                cv2.line(mask, (x1, y1), (x2, y2), the_color)
            else:
                bad_pts.append(idx);

    if len(bad_pts) > 0:
        print(len(bad_pts), " bad points found: ", bad_pts)

    if mask is None:
        print("Mask is None")
        tmp_img = frame_1_clr
    else:
        # Add the mask of lines to the green channel:
        # Remember you are in PYTHON, 1 is the center channel (green).
        tmp_img = cv2.add(mask, frame_1_clr[:, :, 1])
        frame_1_clr[:, :, 1] = tmp_img;

    cv2.imshow('ColorWindow', frame_1_clr)

    # Check for the escape key:
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    n_frames_these_pts = n_frames_these_pts + 1;
    if n_frames_these_pts >= 1:
        p0 = cv2.goodFeaturesToTrack(frame_1_grn, mask=None, **feature_params)
    n_frames_these_pts = 1;
    frame_0_clr = frame_1_clr.copy();
    frame_0_grn = frame_1_grn.copy();
    # time.sleep( 1/30.0 )

cv2.destroyAllWindows()
cap.release()

