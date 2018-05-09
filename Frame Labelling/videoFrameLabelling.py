import cv2
import numpy as np

def frameLabel(frame):
    label = []
    #Convert frame to gray for blurry and uniform labels
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    label = darkLabel(frame, (blurryLabel(grayFrame, (uniformLabel(grayFrame, label)))))
    return label

def darkLabel(frame, label):
    #Normalize to 0.0 - 1.0
    image = frame / 255.0

    #Determine Darkness coefficient Y = mean(RED * R + GREEN * G + BLUE * B)
    Y = ((image[:,:,0].mean() * 0.0722 ) +
                (image[:,:,1].mean() * 0.7152) +
                (image[:,:,2].mean() * 0.2126))

    #Compare Y value to empirically determined Sigma
    if (Y <= 0.097):
        label.append("dark")
        print("Frame is classified as dark.")
    #print("Computed Y Value: " + str(Y))

    return label

def blurryLabel(frame, label):
    #Horizontal Image Gradient
    gaussianX = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    #Vertical Image Gradient
    gaussianY = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    #Compute S Coefficient - mean(Gx^2 + Gy^2)
    S = np.mean((gaussianX * gaussianX) + (gaussianY * gaussianY))

    #Compare S value to empirically determined Beta
    if (S <= 502.32):
        label.append("blurry")
        print("Frame is classified as blurry.")
    #print("Computed S Value: " + str(S))

    return label

def uniformLabel(frame, label):
    #128-bin histogram -> flatten into 1D array -> normalize
    #To get the max range of values use: np.ptp(gray_image)
    hist = (cv2.calcHist([frame],[0],None,[127],[0.0,255.0]).flatten())/ 255.0

    #Sort in descending order
    hist[::-1].sort()

    #Create ratio of each value with respect to hist
    ratios = np.divide(hist, np.sum(hist))

    #Compute U coefficient - sum of top 5th percentile
    U = 1 - np.sum(ratios[0:int(np.floor(0.05 * 128))])

    #Compare U value to empirically determined Y
    if (U < 0.2):
        label.append("uniform")
        print("Frame is classified as uniform.")
    #print("Computed U Value: " + str(U))
    return label

#Load mp4 file
cap = cv2.VideoCapture("playing_ball.mp4")

#Definition of empty container for frame / label storage
frameCollection = []

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, currFrame = cap.read()
    if ret == True:
        label = frameLabel(np.float32(currFrame))
        np.append(frameCollection, [currFrame, label])
    else:
        break

cap.release()
