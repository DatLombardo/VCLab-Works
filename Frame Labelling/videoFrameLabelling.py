import cv2
import numpy as np

def frameLabel(frame):
    label = []
    values = []
    #Convert frame to gray for blurry and uniform labels
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #label = darkLabel(frame, (blurryLabel(grayFrame, (uniformLabel(grayFrame, label, values)))))
    label, values = darkLabel(frame, label, values)
    label, values = blurryLabel(grayFrame, label, values)
    label, values = uniformLabel(grayFrame, label, values)
    return label, values

def darkLabel(frame, label, values):
    #Normalize to 0.0 - 1.0
    image = frame / 255.0

    #Determine Darkness coefficient Y = mean(RED * R + GREEN * G + BLUE * B)
    Y = ((image[:,:,0].mean() * 0.0722 ) +
                (image[:,:,1].mean() * 0.7152) +
                (image[:,:,2].mean() * 0.2126))

    values.append(round(Y, 4))
    #Compare Y value to empirically determined Sigma
    if (Y <= 0.097):
        label.append("dark")
        print("Frame is classified as dark.")
    #print("Computed Y Value: " + str(Y))

    return label, values

def blurryLabel(frame, label, values):
    #Horizontal Image Gradient
    gaussianX = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    #Vertical Image Gradient
    gaussianY = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    #Compute S Coefficient - mean(Gx^2 + Gy^2)
    S = np.mean((gaussianX * gaussianX) + (gaussianY * gaussianY))

    values.append(round(S, 4))
    #Compare S value to empirically determined Beta
    if (S <= 502.32):
        label.append("blurry")
        print("Frame is classified as blurry.")
    #print("Computed S Value: " + str(S))

    return label, values

def uniformLabel(frame, label, values):
    #128-bin histogram -> flatten into 1D array -> normalize
    #To get the max range of values use: np.ptp(gray_image)
    hist = (cv2.calcHist([frame],[0],None,[127],[0.0,255.0]).flatten())/ 255.0

    #Sort in descending order
    hist[::-1].sort()

    #Create ratio of each value with respect to hist
    ratios = np.divide(hist, np.sum(hist))

    #Compute U coefficient - sum of top 5th percentile
    U = 1 - np.sum(ratios[0:int(np.floor(0.05 * 128))])

    values.append(round(U, 4))
    #Compare U value to empirically determined Y
    if (U < 0.2):
        label.append("uniform")
        print("Frame is classified as uniform.")
    #print("Computed U Value: " + str(U))
    return label, values

#Load mp4 file
cap = cv2.VideoCapture("../Testing Data/playing_ball.mp4")
cap = cv2.VideoCapture("../Testing Data/playing_ball.mp4")
#Definition of empty container for frame / label storage
frameCollection = []

#Initial position of text
x = 10
y = 50
font = cv2.FONT_HERSHEY_SIMPLEX
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, currFrame = cap.read()
    if ret == True:
        label, values = frameLabel(np.float32(currFrame))
        np.append(frameCollection, [currFrame, label, values])
        #Add frame labelling values to display,
        #Teal indicates a valid value, and Red indicates a value within threshold.
        if ('dark' in label):
            cv2.putText(currFrame,"Dark(Y) =" + str(values[0]),(x,y),font,0.5,(0,0,255),1)
        else:
            cv2.putText(currFrame,"Dark(Y) =" + str(values[0]),(x,y),font,0.5,(255,246,0),1)
        if ('blurry' in label):
            cv2.putText(currFrame,"Blurry(S) =" + str(values[1]),(x,y+20),font,0.5,(0,0,255),1)
        else:
            cv2.putText(currFrame,"Blurry(S) =" + str(values[1]),(x,y+20),font,0.5,(255,246,0),1)
        if ('uniform' in label):
            cv2.putText(currFrame,"Uniform(U) =" + str(values[2]),(x,y+40),font,0.5,(0,0,255),1)
        else:
            cv2.putText(currFrame,"Uniform(U) =" + str(values[2]),(x,y+40),font,0.5,(255,246,0),1)

        cv2.imshow('frame', currFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break


cap.release()
cv2.destroyAllWindows()
