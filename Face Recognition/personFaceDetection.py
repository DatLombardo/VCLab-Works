#import required libraries
import cv2
import numpy as np
import time


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


#cap = cv2.VideoCapture("../Testing Data/test.mp4")
#cap = cv2.VideoCapture("../Testing Data/river_crossing.mp4")
cap = cv2.VideoCapture("../Testing Data/leaves_jumping.mp4")

#load cascade classifier training file for haarcascade
harrCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

collection = []
frameTimes = []

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, currFrame = cap.read()
    start = time.time()
    if ret == True:
        facesColl = []
        personsColl = []
        grayFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)

        #Face Recognition
        faces = harrCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5);
        for (x, y, w, h) in faces:
            cv2.rectangle(currFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            np.append(facesColl, [x,y,x + w,y + h])
        found, w = hog.detectMultiScale(currFrame, scale=1.1)
        if (len(found) > 0):
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and is_inside(r, q):
                        break
                    else:
                        found_filtered.append(r)
            for (x, y, w, h) in found_filtered:
                cv2.rectangle(currFrame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                np.append(personsColl, [x, y, x + w, y + h])

        #Person Detection
        np.append(collection, [currFrame, facesColl, personsColl])
        cv2.imshow('frame', currFrame)
        end = time.time()
        np.append(frameTimes, [end - start])
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break
