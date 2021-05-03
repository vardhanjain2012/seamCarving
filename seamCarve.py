import cv2
import numpy as np

def energyMeasure(img):
	return (np.sum(img)/(img.shape[0]*img.shape[1]))

def energyE1(img):
	fx = np.matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	fy = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	Ix = cv2.filter2D(img, -1, fx)
	Iy = cv2.filter2D(img, -1, fy)
	e1 = np.add(Ix, Iy)
	return e1

def cumulativeMinEnergy(e1):
	minEnergy = np.zeros(e1.shape)

	nx = (e1.shape)[0]
	ny = (e1.shape)[1]

	for x in range(nx):
		minEnergy[x][0] = e1[x][0]

	for y in range(1, ny):
		for x in range(nx):
			minVal = 255
			if(x>0):
				minVal = min(minVal, minEnergy[x-1][y-1])
			if(x<(nx-1)):
				minVal = min(minVal, minEnergy[x+1][y-1])
			minVal = min(minVal, minEnergy[x][y-1])
			minVal+=e1[x][y]
			minEnergy[x][y]=minVal
	return minEnergy

def findMinSeam(minEnergy):
	seamPoints = []
	nx = (minEnergy.shape)[0]
	ny = (minEnergy.shape)[1]
	minX = 0
	minVal = 255
	for x in range(nx):
		if(minVal>minEnergy[x][ny-1]):
			minVal=minEnergy[x][ny-1]
			minX = x
	seamPoints.append((minX, ny-1))
	for y in range(ny-2, -1, -1):
		xD = minX
		minVal = minEnergy[xD][y]
		if(xD>0 and minEnergy[xD-1][y]<minVal):
			minVal = minEnergy[xD-1][y]
			minX = xD-1
		if(xD<(nx-1) and minEnergy[xD+1][y]<minVal):
			minVal = minEnergy[xD+1][y]
			minX = xD+1
		seamPoints.append((minX, y))
	return seamPoints

def updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index):
	for (x, y) in seamPoints:
		(nx, ny) = inverseImgMap[x, y]
		srcSeamMap[nx, ny] = index
	return

def deletePoints(img, seamPoints):
	newImg = np.delete(img, -1, 0)
	for (x, y) in seamPoints:
		for i in range(x+1, img.shape[0]):
			newImg[i-1, y] = img[i, y]
	return newImg

def detectSeam(img, inverseImgMap, srcSeamMap, index, output):
	energy = energyE1(img)
	minEnergy = cumulativeMinEnergy(energy)
	seamPoints = findMinSeam(minEnergy)
	updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index)
	newImg = deletePoints(img, seamPoints)
	newInverseMap = deletePoints(inverseImgMap, seamPoints)
	output = deletePoints(output, seamPoints)
	return (newImg, newInverseMap, output)

def detectSeams(numSeams, bw, inverseImgMap, srcSeamMap, output):
	for x in range(numSeams):
		(bw, inverseImgMap, output) = detectSeam(bw, inverseImgMap, srcSeamMap, x, output)
	return output

def initialization(shape):
	srcSeamMap = np.ones(shape, dtype='uint')*(shape[0] + shape[1])
	inverseImgMap = np.empty(shape, dtype=object)
	for i in range(shape[0]):
		for j in range(shape[1]):
			inverseImgMap[i, j] = (i, j)
	return (srcSeamMap, inverseImgMap)

def displaySeams(src, srcSeamMap, numSeams):
	for i in range(src.shape[0]):
		for j in range(src.shape[1]):
			if(srcSeamMap[i, j]<numSeams):
				src[i, j] = [0, 0, 255]
	return

if __name__== "__main__":
	src = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_COLOR)
	output = src.copy()
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	(srcSeamMap, inverseImgMap) = initialization(bw.shape)
	numSeams = 50
	output = detectSeams(numSeams, bw, inverseImgMap, srcSeamMap, output)
	srcSeam = src.copy()
	displaySeams(srcSeam, srcSeamMap, numSeams)
	cv2.imshow("src", src)
	cv2.imshow("output", output)
	cv2.imshow("srcSeams", srcSeam)
	cv2.waitKey(0)
	cv2.destroyAllWindows()