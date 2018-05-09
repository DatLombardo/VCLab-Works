import cv2
import numpy as np
from matplotlib import pyplot as plt

#Constant Definition
Y = 0.2

#Image import
img = cv2.imread('uniformLabel.jpg')

#Convert image to gray & Convert to float32
gray_image =  np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

#128-bin histogram -> flatten into 1D array -> normalize
#To get the max range of values use: np.ptp(gray_image)
hist = (cv2.calcHist([gray_image],[0],None,[127],[0.0,255.0]).flatten())/ 255.0

#Sort in descending order
hist[::-1].sort()

#Create ratio of each value with respect to hist
ratios = np.divide(hist, np.sum(hist))

#Compute U coefficient - sum of top 5th percentile
U = 1 - np.sum(ratios[0:int(np.floor(0.05 * 128))])

#Compare U value to empirically determined Y
if (U < Y):
    print("Frame is classified as uniform.")
print("Computed U Value: " + str(U))
