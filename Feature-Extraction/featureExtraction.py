import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import csv

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.65,
    'gpu': 1.0,
}

def frameLabel(frame):
    #Convert frame to gray for blurry and uniform labels
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #label = darkLabel(frame, (blurryLabel(grayFrame, (uniformLabel(grayFrame, label, values)))))
    Y = darkLabel(frame)
    S = blurryLabel(grayFrame)
    U = uniformLabel(grayFrame)
    return [Y, S, U]

def darkLabel(frame):
    #Normalize to 0.0 - 1.0
    image = frame / 255.0

    #Determine Darkness coefficient Y = mean(RED * R + GREEN * G + BLUE * B)
    Y = ((image[:,:,0].mean() * 0.0722 ) +
                (image[:,:,1].mean() * 0.7152) +
                (image[:,:,2].mean() * 0.2126))

    #Compare Y value to empirically determined Sigma
    if (Y <= 0.097):
        print("Frame is classified as dark.")

    return round(Y, 5)

def blurryLabel(frame):
    #Horizontal Image Gradient
    gaussianX = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    #Vertical Image Gradient
    gaussianY = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    #Compute S Coefficient - mean(Gx^2 + Gy^2)
    S = np.mean((gaussianX * gaussianX) + (gaussianY * gaussianY))

    #Compare S value to empirically determined Beta
    if (S <= 502.32):
        print("Frame is classified as blurry.")

    return round(S, 5)

def uniformLabel(frame):
    #128-bin histogram -> flatten into 1D array -> normalize
    #To get the max range of values use: np.ptp(gray_image)
    hist = (cv2.calcHist([frame],[0],None,[127],[0.0,255.0]).flatten())/ 255.087

    #Sort in descending order
    hist[::-1].sort()

    #Create ratio of each value with respect to hist
    ratios = np.divide(hist, np.sum(hist))

    #Compute U coefficient - sum of top 5th percentile
    U = 1 - np.sum(ratios[0:int(np.floor(0.05 * 128))])

    #Compare U value to empirically determined Y
    if (U < 0.2):
        print("Frame is classified as uniform.")

    return round(U, 5)

def HSVMean(frame):
    #Convert to HSV colour space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, channels = frame.shape

    #Take dimensions for the center of the frame
    upper_left = (int(width / 3), int(height / 3))
    bottom_right = (int(width * 2 / 3), int(height * 2 / 3))
    centerFrame = hsv[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

    #Take the slice of each column of H,S,V and take the average
    imageMeanHSV = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
    centerMeanHSV = centerFrame[:,:,0].mean(), centerFrame[:,:,1].mean(), centerFrame[:,:,2].mean()
    return imageMeanHSV, centerMeanHSV


tfnet = TFNet(option)

#Load mp4 file
capture = cv2.VideoCapture('../Testing-Data/leaves_jumping.mp4')
#capture = cv2.VideoCapture("../Testing-Data/playing_ball.mp4")
colours = [tuple(255 * np.random.rand(3)) for i in range(5)]

#Definition of empty container for frame / label storage
frameCollection = []

#Initial position of text
x = 10
y = 50
font = cv2.FONT_HERSHEY_SIMPLEX

while (capture.isOpened()):
    ret, currFrame = capture.read()
    if ret:
        frameObjects = []

        imageHSV, centerHSV = HSVMean(currFrame)
        values = frameLabel(np.float32(currFrame))
        #Yolo Object Detection
        results = tfnet.return_predict(currFrame)
        for colour, result in zip(colours, results):
            #Extract X,Y Position of Top-Left Bounding Box, Bottom-Right Bounding Box
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            currFrame = cv2.rectangle(currFrame, tl, br, colour, 3)
            currFrame = cv2.putText(currFrame, result['label'], tl, font, 0.7, (0, 0, 0), 2)
            #Generate Object list, [label, confidence, top-left bb, bottom-right bb]
            frameObjects.append([result['label'], result['confidence'], tl, br])

        #Dark (Y) [0], Blurry (S) [1], Uniform (U) [2], Image Mean HSV, Center Image Mean HSV, #Detected Objects, Object list
        frameCollection.append([currFrame, values[0], values[1], values[2], imageHSV, centerHSV, len(frameObjects), frameObjects])

        #Add frame labelling values to display,
        cv2.putText(currFrame,"Dark(Y) =" + str(values[0]),(x,y),font,0.5,(255,246,0),1)
        cv2.putText(currFrame,"Blurry(S) =" + str(values[1]),(x,y+20),font,0.5,(255,246,0),1)
        cv2.putText(currFrame,"Uniform(U) =" + str(values[2]),(x,y+40),font,0.5,(255,246,0),1)

        #Display current frame
        cv2.imshow('frame', currFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break

#Open feature vector container, and empty.
featureFile = open('featureVector.csv', 'w')
featureFile.truncate()

#Write to csv
with featureFile:
    writer = csv.writer(featureFile)
    writer.writerows(frameCollection)

featureFile.close()
