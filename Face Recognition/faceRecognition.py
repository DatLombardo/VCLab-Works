#import required libraries
import cv2
import numpy as np

#cap = cv2.VideoCapture("../Testing Data/test.mp4")
cap = cv2.VideoCapture("../Testing Data/river_crossing.mp4")
#load cascade classifier training file for haarcascade
harrCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

collection = []


while (cap.isOpened()):
    # Capture frame-by-frame
    ret, currFrame = cap.read()
    if ret == True:
        facesColl = []
        grayFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
        #np.append(frameCollection, [currFrame, label, values])

        #Face Recognition4
        faces = harrCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5);
        for (x, y, w, h) in faces:
            cv2.rectangle(currFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            np.append(facesColl, [x,y,x + w,y + h])

        #Person Detection
        np.append(collection, [currFrame, facesColl])
        cv2.imshow('frame', currFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break
