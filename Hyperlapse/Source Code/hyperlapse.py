'''
python hyp.py -f filename [<args>]
Executable:
./hyp -f filename [<args>]
-f or --filename
Video input file.
-v or --velocity
Generated video velocity. Default: 10
-o or --output
Generated video file. Default: out_ <filename>
-h or --help
Print this help.
'''


import sys
import getopt
import cv2
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix

def alignmentCost(homography, srcPts, dstPts):
	totalError = 0
	for k in xrange(0, len(srcPts)  ):
		totalError += errorModule(homography, srcPts[k], dstPts[k])
	meanError = totalError / len(srcPts)
	return meanError

def errorModule(homography, srcPt,dstPt):
	distance = (dstPt - cv2.perspectiveTransform(np.array([srcPt])[None, :, :],homography))[0][0]
	return hypotenuse(distance[0], distance[1])

def overlapCost(homography, center):
	return errorModule(homography, center, center)

def matchingCost(cr, co, tauC, gam):
	M, N = cr.get_shape()
	cm = dok_matrix((M, N), dtype = np.float32)
	for i in range(M):
		for j in range(N):
			if cr[i,j] < tauC:
				cm[i,j] = co[i,j]
			else:
				cm[i,j] = gam
	#cm[i,j] = co[i,j] if cr[i,j] < tauC else gam
	return cm

def hypotenuse(width, height):
	return (width**2 + height**2)**(0.5)

def gamma(width, height):
	return 0.5*hypotenuse(width, height)

def speedupCost(i, j, velocity, tauS = 200):
	return min(abs(j - i - velocity)**2, tauS)

def accelerationCost(i, j, h, tauA = 200):
	return min(abs((j - i) - (i - h))**2, tauA)

def minimizeDynamicCost(dv, g, windowSize, frameCount):
    minDv = None
    minI = None
    minJ = None
    for i in xrange(frameCount - g, frameCount):
        for j in xrange(i + 1, min(frameCount, i + windowSize + 1)):
            if (minDv is None) or dv[i, j] < minDv:
                minDv = dv[i,j]
                minI = i
                minJ = j
    return minI, minJ

def costFunc(dv, ca, i, j, k, lambdaA):
    return dv[i - k, i] + lambdaA * (ca[k])[i, j]

def minimize(dv, ca, i, j, windowSize, lambdaA = 80):
    minValue = None
    minK = None
    for k in xrange(1, min(windowSize, i) + 1):
        value = costFunc(dv, ca, i, j, k, lambdaA)
        if ((minValue is None) or (value < minValue)) and (value is not None):
            minValue = value
            minK = k
    return minValue, minK


def detect(videoCapture):
	frameCount = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
	kps = []
	descriptors = []
	i = 0
	while True and i<=frameCount:
		showProgress(i, frameCount, "Detecting video:")

		i += 1

		videoCapture.grab()
		r,img = videoCapture.retrieve()
		if not r:
			videoCapture.release()
			break

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		featDetector = cv2.GFTTDetector_create(500, 0.01,1,3,True,0.04)
		features = featDetector.detect(gray)
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		kp, des = brief.compute(gray, features)

		kps.append(kp)
		descriptors.append(des)
	sys.stdout.write("\n")
	return kps, descriptors

def matchAndFindHomography(matcher, srcDescriptor, dstDescriptor, srcKps, dstKps):
	if srcDescriptor is None or dstDescriptor is None:
		return None

	matches = matcher.match(srcDescriptor, dstDescriptor)

	qis = [match.queryIdx for match in matches]
	currentPts = [kp.pt for kp in srcKps]
	srcPts = np.asarray([currentPts[k] for k in qis])

	tis = [match.trainIdx for match in matches]
	currentPts = [kp.pt for kp in dstKps]
	dstPts = np.asarray([currentPts[k] for k in tis])

	homography, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
	return homography

def saveVideo(filename, path, output = None):
	if output is None:
		output = "out_" + filename+'.avi'
	videoCapture = cv2.VideoCapture()
	videoCapture.open(filename)
	width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#fourcc = int(videoCapture.get(cv2.CAP_PROP_FOURCC))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
	videoWriter = cv2.VideoWriter(output, fourcc, fps, (width, height))
	for i in xrange(0, path[-1] + 1):
	    videoCapture.grab()
	    r,img = videoCapture.retrieve()
	    if i in path:
	        videoWriter.write(img)

def showProgress(i, total, label = "progress:"):
	perc = (100*i)/float(total)
	sys.stdout.write("\r")
	sys.stdout.write(label)
	sys.stdout.write(" ")
	sys.stdout.write("{0:5.1f}%".format(perc))
	sys.stdout.flush()

def usage():
	print ("usage: ./hyp -f filename [<args>]")

	print("-f, --filename")
	print("Videoinput file.")

	print ("-v, --velocity")
	print ("Generated video velocity. Default: 10")

	print ("-o, --output")
	print ("Generated video file. Default: out_<filename>")

	print ("-h, --help")
	print ("Print this help.")

def main(argv):
	velocity = 10
	lambdaS = 200
	filename = None
	output = None

	opts, args = getopt.getopt(argv, "v:f:o:h",["velocity=","filename=","output=","help"])
	for opt, arg in opts:
		if opt in ("-v", "--velocity"):
			velocity = int(arg)
		elif opt in ("-f", "--filename"):
			filename = arg
		elif opt in ("-o", "--output"):
			output = arg
		elif opt in ("-h", "--help"):
			usage()
			sys.exit()
		else:
			usage()
			sys.exit()

	if filename is None:
		usage()
		sys.exit()

	windowSize = velocity * 2
	matcher = cv2.BFMatcher()

	videoCapture = cv2.VideoCapture()
	videoCapture.open(filename)

	frameCount = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
	width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
	tauC = 0.1 * hypotenuse(width, height)
	gam = gamma(width, height)
	center = np.array([width/2, height/2])
	print ("Frame count: " + str(frameCount))

	kps, descriptors = detect(videoCapture)

	cr = dok_matrix((len(kps),len(kps)), dtype = np.float32)
	co = dok_matrix((len(kps),len(kps)), dtype = np.float32)
	cs = dok_matrix((len(kps),len(kps)), dtype = np.float32)
	ca = [None for x in range(windowSize + 1)]


	for h in xrange(1, windowSize + 1):
		ca[h] = dok_matrix((len(kps),len(kps)), dtype = np.float32)

	for i in xrange(0,len(kps)-1):
		showProgress(i, len(kps) - 1, "Calculating costs:")
		for j in xrange(i + 1,i + windowSize + 1):
			if j == len(kps):
				break

			cs[i, j] = speedupCost(i, j, velocity)
			for k in xrange(1, windowSize + 1):
				(ca[k])[i, j] = accelerationCost(i, j, i - k)

			if descriptors[i] is None or descriptors[j] is None:
				homography = None
			else:
				matches = matcher.match(descriptors[i], descriptors[j])
				qis = [match.queryIdx for match in matches]
				currentPts = [kp.pt for kp in kps[i]]
				srcPts = np.asarray([currentPts[k] for k in qis])

				tis = [match.trainIdx for match in matches]
				currentPts = [kp.pt for kp in kps[j]]
				dstPts = np.asarray([currentPts[k] for k in tis])

				homography, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)

			if homography is not None:
			    cr[i, j] = alignmentCost(homography, srcPts, dstPts)
			    co[i, j] = overlapCost(homography, center)
			else:
				cr[i, j] = tauC
				co[i, j] = gam

	sys.stdout.write("\n")

	cm = matchingCost(cr, co, tauC, gam)

	##### Stage 2 #######
	g = 4
	dv = dok_matrix((len(kps),len(kps)), dtype = np.float32)
	tv = dok_matrix((len(kps),len(kps)), dtype = np.int)

	###Initialization
	for i in xrange(1, g + 1):
	    for j in xrange(i + 1, i + windowSize + 1):
	        dv[i, j] = cm[i, j] + lambdaS * cs[i, j]

	###First pass
	for i in xrange(g, frameCount):
		showProgress(i, frameCount, "Path selection:")
		for j in xrange(i + 1, min(frameCount,i + windowSize) + 1):
		    c = cm[i, j] + lambdaS * cs[i, j]
		    minCost, argmin = minimize(dv, ca, i, j, windowSize)
		    dv[i, j] = c + minCost
		    tv[i, j] = argmin
	sys.stdout.write("\n")

	###Second pass
	s, d = minimizeDynamicCost(dv, g, windowSize, frameCount)
	p = [d]
	while s > g:
	    p[:0] = [s]
	    b = tv[s,d]
	    d = s
	    s = s - b

	print ("Optimal path:")
	print (p)

	saveVideo(filename, p, output)

if __name__ == '__main__':
    main(sys.argv[1:])
