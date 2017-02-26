import cv2
import numpy as np

#Dense Optical Flow Calculation
#version trial for speed estimation
VidInFile = 'Video_name.AVI'
inputVideo = cv2.VideoCapture(VidInFile)

#reading ech frame
ret, frame1 = inputVideo.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
   ret, frame2 = inputVideo.read()
   next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

   flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
   mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
   hsv[...,0] = ang*180/np.pi/2
   hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
   bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

   cv2.imshow('frame2',bgr)
   k = cv2.waitKey(30) & 0xff
   if k == 27:
      break
   elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

inputVideo.release()
cv2.destroyAllWindows()
