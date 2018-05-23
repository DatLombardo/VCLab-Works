#import required libraries
import cv2
import numpy as np

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

cap = cv2.VideoCapture("../Testing Data/river_crossing.mp4")
#cap = cv2.VideoCapture("../Testing Data/leaves_jumping.mp4")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

collection = []

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, currFrame = cap.read()
    if ret == True:
        personsColl = []
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
        np.append(collection, [currFrame, personsColl])
        cv2.imshow('frame', currFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
