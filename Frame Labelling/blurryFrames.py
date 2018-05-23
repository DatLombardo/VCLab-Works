import cv2
import numpy as np

#Constant Declaration
BETA = 502.32

#Image import
img = cv2.imread('../Testing Data/blurryLabel.jpg')

#Convert image to gray & Convert to float32
gray_image = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

#Horizontal Image Gradient
gaussianX = cv2.Sobel(gray_image,cv2.CV_32F,1,0)
#Vertical Image Gradient
gaussianY = cv2.Sobel(gray_image,cv2.CV_32F,0,1)

#Compute S Coefficient - mean(Gx^2 + Gy^2)
S = np.mean((gaussianX * gaussianX) + (gaussianY * gaussianY))

#Compare S value to empirically determined Beta
if (S <= BETA):
    print("Frame is classified as blurry.")
print("Computed S Value: " + str(S))
