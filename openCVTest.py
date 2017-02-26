#
#  Display both temporal and spatial edges.  Uses Blue and Yellow instead of red/green for 
#  enhanced visibility.
#
import cv2 			# OpenCV2
import time			# For timing operations.
import string			# String manipulations.
import numpy as np		# For manipulating images.
from numpy import *
import scipy
from scipy import stats 	# For the mode of an array.
import sys

ESC_KEY 		= 27;		# Key to stop the program.

TIME_QUANT      	=  1; 		# How many frames to advance.
SPACE_QUANT     	= 16;		# 

BLU_PLANE 		= 0;		# Constant to keep from going crazy.
GRN_PLANE 		= 1;		# Constant to keep from going crazy.
RED_PLANE 		= 2;		# Constant to keep from going crazy.

OUTPUT_DIR 		= "OUTPUT_FRAMES_v78_blue_orange/"	# Place to put output videos.

DISPLAY_EACH_FRAME 	= True		# Want to watch as it runs?
SAVE_EACH_FRAME 	= False		# What to look at each individual frame in its own file?

GAUS_KERNEL_SIZE 	= (9,9)		# Spacial extent of the image.
GAUS_KERNEL_STD 	= 1.85		# Standard deviation for the Gaussian blurring.

#
#  MOUSE HANDLER:
#
#  If use clicks on the mouse, set the global variable named clicked to true,
#  so we know to stop the program later on.
#  
clicked 		= False

def onMouse(event, x, y, flags, param) :
    global clicked
    if event == cv2.cv.CV_EVENT_LBUTTONUP :
        clicked = True


##########################################################################################################
#
#                                                   MAIN
#
##########################################################################################################
"""if len(sys.argv[1]) > 0 :			# Checks for input parameters on the command line.
    input_filename 		= sys.argv[1]
else :
    input_filename 		= '/Volumes/SCANDISK/ONE_ROAD_VIDEO/11440006.AVI';
"""

input_filename      = '/Volumes/TimeWarp/ONE_ROAD_VIDEO/11440006.AVI';

dir_parts 			= input_filename.split('/');					# Get sub-directories
fn_entire			= dir_parts[-1];
file_parts 			= fn_entire.split('.');						# Get the basename.
sep 				= '_';
output_filename 		= "TBK_OT_BluOrng_" + dir_parts[-2] + "_" + sep.join( file_parts[ 0:-1 ]) + "_v78.mov";		# Build output as date + .mov
output_frame_prefix 		= "TBK_Fr_BluOrng_" + dir_parts[-2] + '_' + sep.join( file_parts[ 0:-1 ] ) + '_v78_'		# Used for each frame saved out

# print "output_filename          = ", output_filename 
# print "output_frame_prefix      = ", output_frame_prefix

#
#   OPEN THE INPUT VIDEO:
#
vidIn   	= cv2.VideoCapture( input_filename );

fps 		= vidIn.get(cv2.cv.CV_CAP_PROP_FPS)
print 'FPS = ', fps

ht    		= int(vidIn.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT));
wd    		= int(vidIn.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH));
size    	= ( ht, wd );

nFrames 	= int( vidIn.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) )
print 'nFrames = ', nFrames, 'Sized ( ', ht, ' x ', wd, ' )'

print "Pausing for a second to let that sink in... "
time.sleep(1)

#
#   OPEN THE OUTPUT VIDEO:
#
fourcc 		= cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
vidOut 		= cv2.VideoWriter( output_filename, fourcc, fps, (wd,ht) )

#
#  Get the first frame, display it, and set the mouse handler 
#  so that a click will stop playing the video:
#
success, frame 	= vidIn.read()
sTempFrame 	= np.float32( (2* np.float32(frame[:,:,GRN_PLANE]) + \
				  np.float32(frame[:,:,RED_PLANE]) + \
				  np.float32(frame[:,:,BLU_PLANE]) ) / 3.0 );
sThisFrame 	= cv2.GaussianBlur( sTempFrame, GAUS_KERNEL_SIZE, GAUS_KERNEL_STD )
sLastFrame 	= sThisFrame.copy();
sLastFrame2 	= sLastFrame.copy();
sDeltaChg2 	= 0 * sLastFrame;

cv2.imshow(output_frame_prefix, sThisFrame );
cv2.setMouseCallback(output_frame_prefix, onMouse); 		# All this was just to set this callback on the window.

dupCntr 	= 0;
frameCntr 	= 0;
kk 		= -1;
kk 		= cv2.waitKey(1);

# Create a BLANK IMAGE:
sDisplayChgs 	= np.zeros((ht,wd,3), np.uint8)
nTime_A 	= time.mktime( time.localtime() );

while   (success) and (kk == -1) and (not clicked) and ( frameCntr <= (nFrames-2) ): 

    print 'Frame ', frameCntr, ' / ', nFrames, '  ',

    # TIME DIFFERENCES W.R.T. Two Frames Ago --
    sDeltaTm2 	= np.float32(sThisFrame) - np.float32(sLastFrame2);
    sDeltaTm1 	= np.float32(sThisFrame) - np.float32(sLastFrame);

    ttl_change 		= np.sum( abs(sDeltaTm1) )
    print "ttl change = ", ttl_change
    if ttl_change > 485000 :

	# avg_change = ttl_change / wd / ht;
 	# fmt_ttl_chg  		= "%7d"   % ttl_change
 	# fmt_avg_chg 		= "%6.3f" % avg_change
        # print 'Total Change = ', fmt_ttl_chg, ' Average Change = ', fmt_avg_chg

	# sGrayImage 			= cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        sSpatialEdgsGrayMag 		= np.float64( cv2.Laplacian( sThisFrame, cv2.CV_64F, ksize=5 ) )

	# Ignore small temporal and spatial values:
	sTimeEdgesMagQuant 		= np.uint8( ( abs(sDeltaTm1/TIME_QUANT) + abs(sDeltaTm2/(2*TIME_QUANT) ) ) * (TIME_QUANT*2)  )
 	sSpatialEdgsGrayMagQuant 	= np.uint8( np.uint8(abs(sSpatialEdgsGrayMag / SPACE_QUANT )) * (SPACE_QUANT/3.0) ) 

	#
	#  VISUALIZATION:
	#
	sDisplayChgs[:,:,BLU_PLANE] 	= np.uint8( sSpatialEdgsGrayMagQuant );
	sDisplayChgs[:,:,GRN_PLANE] 	= np.uint8( (sTimeEdgesMagQuant + sSpatialEdgsGrayMagQuant) / 2.0 );
        sDisplayChgs[:,:,RED_PLANE] 	= np.uint8( sTimeEdgesMagQuant  );

	#
	# WRITE OUT A FRAME OF VIDEO:
	#
        vidOut.write( np.uint8(sDisplayChgs) )

	# Show the changes to the user, at least once every so-many frames:
	if ( DISPLAY_EACH_FRAME ) or ( frameCntr % 90 == 0 ) :
	    cv2.imshow(output_frame_prefix, sDisplayChgs )

	if SAVE_EACH_FRAME :
	    file_numb 	= str(frameCntr).zfill(5);
	    fn_out 	= OUTPUT_DIR + output_frame_prefix + file_numb + ".jpg"
	    cv2.imwrite( fn_out, sDisplayChgs )
	else :
	    print " ",

    else:
	dupCntr  += 1;
	# print 'Duplicate frame.  Skipped.'

    # Store the old frames for the next loop iteration:
    sLastFrame2 = sLastFrame.copy();
    sLastFrame 	= sThisFrame.copy();

    #
    # Get the next, current, frame:
    success, frame 	= vidIn.read();	
    # print "Skipping a frame |    ",
    if success :
        success, frame 	= vidIn.read();		# Skip one frame...
        frameCntr  += 1;

    # print "Skipping a frame |    ",
    if success :
        success, frame 	= vidIn.read();		# Skip one frame...
        frameCntr  += 1;

    sTempFrame 	= np.float32( (2* np.float32(frame[:,:,GRN_PLANE]) + \
				  np.float32(frame[:,:,RED_PLANE]) + \
				  np.float32(frame[:,:,BLU_PLANE]) ) / 3.0 );
    sThisFrame 	= cv2.GaussianBlur( sTempFrame, GAUS_KERNEL_SIZE, GAUS_KERNEL_STD )
    frameCntr  += 1;

    # Check for a key, for 1/1000th of a second:
    kk = cv2.waitKey(1);
    # print( 'kk = ', kk )

nTime_B 	= time.mktime( time.localtime() );
duration 	= nTime_B - nTime_A

print "Duration was ", duration
print "Processed ", frameCntr / duration, " frames per second "


#  Clean up memory:
cv2.destroyAllWindows();
vidIn.release() 
vidOut.release() 
vidIn = None


print "There were ", dupCntr, " duplicates in the ", frameCntr, " frames.  Or a ", dupCntr * 100.0 / frameCntr, " duplicate rate."


