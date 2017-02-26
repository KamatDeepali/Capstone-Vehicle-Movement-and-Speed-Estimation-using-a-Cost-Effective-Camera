import numpy as np
import cv2
import argparse
import glob
from matplotlib import pyplot as plt



cap = cv2.VideoCapture('Road_Video');

while(True):
    # Capturing one frame at a time
    ret, frame = cap.read();


    print (type(frame))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);

    # Defining bounds for the color white in HSV
    sensitivity = 15;
    lower_white = np.array([0, 0, 255-sensitivity], dtype=np.uint8);
    upper_white = np.array([180, sensitivity, 255], dtype=np.uint8);


    v= np.median(hsv)
    sigma =0.9
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))
    edges = cv2.Canny(frame, 100, 200)

    #plt.subplot(121), plt.imshow(frame, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(edges, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()


    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Thresholding HSV to access only white
    mask = cv2.inRange(hsv, lower_white, upper_white);

    #bitwise_and mask on original image
    result = cv2.bitwise_and(frame,frame, mask=mask);

    cv2.imshow('frame', frame);
    cv2.imshow('mask', mask);
    cv2.imshow('result', result);
    #cv2.imshow('edged', lines)


    # Displaying resulting frame
    #cv2.imshow('frame', hsv)
    k = cv2.waitKey(1) & 0xFF;
    if k==27:
        break;


# After completion, release capture
cap.release()
cv2.destroyAllWindows()




