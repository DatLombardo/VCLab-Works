import cv2
import numpy as np

#Load mp4 file
capture = cv2.VideoCapture('../Testing-Data/leaves_jumping.mp4')
while (capture.isOpened()):
    ret, currFrame = capture.read()
    if ret:
        hsv = cv2.cvtColor(currFrame, cv2.COLOR_BGR2HSV)
        height, width, channels = currFrame.shape
        #Take dimensions for the center of the frame
        upper_left = (int(width / 3), int(height / 3))
        bottom_right = (int(width * 2 / 3), int(height * 2 / 3))
        centerFrame = hsv[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        imageMeanHSV = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
        centerMeanHSV = centerFrame[:,:,0].mean(), centerFrame[:,:,1].mean(), centerFrame[:,:,2].mean()
        #cv2.imshow('frame', currFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
