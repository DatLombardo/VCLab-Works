import cv2
import numpy as np

#Constants Declaratin
RED = 0.2126
GREEN = 0.7152
BLUE = 0.0722

#Image import
img = cv2.imread('darkLabel.jpg')

#Convert image to float32
m = np.float32(img)

#Normalize to 0.0 - 1.0
image = m / 255.0

#Determine Darkness coefficient Y
Y = ((image[:,:,0].mean() * BLUE ) +
            (image[:,:,1].mean() * GREEN) +
            (image[:,:,2].mean() * RED))
print(Y)
