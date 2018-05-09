import cv2
import numpy as np

#Constants Declaration
RED = 0.2126
GREEN = 0.7152
BLUE = 0.0722
SIGMA = 0.097

#Image import & Convert image to float32
img = np.float32(cv2.imread('darkLabel.jpg'))

#Normalize to 0.0 - 1.0
image = m / 255.0

#Determine Darkness coefficient Y = mean(RED * R + GREEN * G + BLUE * B)
Y = ((image[:,:,0].mean() * BLUE ) +
            (image[:,:,1].mean() * GREEN) +
            (image[:,:,2].mean() * RED))

#Compare Y value to empirically determined Sigma
if (Y <= SIGMA):
    print("Frame is classified as dark.")
print("Computed Y Value: " + str(Y))

