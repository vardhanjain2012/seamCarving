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

def cumulativeMinEnergy(e1, alongX):
	minEnergy = np.zeros(e1.shape)

	nx = (e1.shape)[0]
	ny = (e1.shape)[1]

	if(alongX):
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
	else:
		for y in range(ny):
			minEnergy[0][y] = e1[0][y]

		for x in range(1, nx):
			for y in range(ny):
				minVal = 255
				if(y>0):
					minVal = min(minVal, minEnergy[x-1][y-1])
				if(y<(ny-1)):
					minVal = min(minVal, minEnergy[x-1][y+1])
				minVal = min(minVal, minEnergy[x-1][y])
				minVal+=e1[x][y]
				minEnergy[x][y]=minVal
	return minEnergy

def findMinSeam(minEnergy, alongX):
	seamPoints = []
	nx = (minEnergy.shape)[0]
	ny = (minEnergy.shape)[1]

	if(alongX):
		minX = 0
		minVal = 255
		for x in range(nx):
			if(minVal>minEnergy[x][ny-1]):
				minVal=minEnergy[x][ny-1]
				minX = x
		seamPoints.append((minX, ny-1))
		minEnergyVal = minVal
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
	else:
		minY = 0
		minVal = 255
		for y in range(ny):
			if(minVal>minEnergy[nx-1][y]):
				minVal=minEnergy[nx-1][y]
				minY = y
		seamPoints.append((nx-1, minY))
		minEnergyVal = minVal
		for x in range(nx-2, -1, -1):
			yD = minY
			minVal = minEnergy[x][yD]
			if(yD>0 and minEnergy[x][yD-1]<minVal):
				minVal = minEnergy[x][yD-1]
				minY = yD-1
			if(yD<(ny-1) and minEnergy[x][yD+1]<minVal):
				minVal = minEnergy[x][yD+1]
				minY = yD+1
			seamPoints.append((x, minY))
	return (seamPoints, minEnergyVal)

def updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index, alongX):
	if(not alongX):
		index = -(index+1)
	for (x, y) in seamPoints:
		(nx, ny) = inverseImgMap[x, y]
		srcSeamMap[nx, ny] = index
	return

def deletePoints(img, seamPoints, alongX):
	if(alongX):
		newImg = np.delete(img, -1, 0)
		for (x, y) in seamPoints:
			for i in range(x+1, img.shape[0]):
				newImg[i-1, y] = img[i, y]
	else:
		newImg = np.delete(img, -1, 1)
		for (x, y) in seamPoints:
			for i in range(y+1, img.shape[1]):
				newImg[x, i-1] = img[x, i]
	return newImg

def addPoints(img, seamPoints, alongX):
	if(alongX):
		newImg = np.insert(img, -1, 1, axis=0)
		for (x, y) in seamPoints:
			if(x==0 or type(img[0, 0]) is tuple):
				newImg[x, y] = img[x, y]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[x, y] = (img[x, y]/2 + img[x-1, y]/2)
			for i in range(x, img.shape[0]):
				newImg[i+1, y] = img[i, y]
	else:
		newImg = np.insert(img, -1, 1, axis=1)
		for (x, y) in seamPoints:
			if(y==0 or type(img[0, 0]) is tuple):
				newImg[x, y] = img[x, y]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[x, y] = (img[x, y]/2 + img[x, y-1]/2)
			for i in range(y, img.shape[1]):
				newImg[x, i+1] = img[x, i]
	return newImg

def removeMinSeam(img, inverseImgMap, alongX, srcSeamMap, output, seamPoints, index):
	updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index, alongX)
	newImg = deletePoints(img, seamPoints, alongX)
	newInverseMap = deletePoints(inverseImgMap, seamPoints, alongX)
	output = deletePoints(output, seamPoints, alongX)
	return (newImg, newInverseMap, output)

def addMinSeam(img, inverseImgMap, alongX, srcSeamMap, output, seamPoints, index):
	updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index, alongX)
	newImg = addPoints(img, seamPoints, alongX)
	newInverseMap = addPoints(inverseImgMap, seamPoints, alongX)
	output = addPoints(output, seamPoints, alongX)
	return (newImg, newInverseMap, output)

def detectMinSeam(img, alongX):
	energy = energyE1(img)
	minEnergy = cumulativeMinEnergy(energy, alongX)
	(seamPoints, minEnergyVal) = findMinSeam(minEnergy, alongX)
	newImg = deletePoints(img, seamPoints, alongX)
	return (newImg, minEnergyVal, seamPoints)

def initialization(shape):
	srcSeamMap = np.ones(shape, dtype='int')*(shape[0] + shape[1])
	inverseImgMap = np.empty(shape, dtype=object)
	for i in range(shape[0]):
		for j in range(shape[1]):
			inverseImgMap[i, j] = (i, j)
	return (srcSeamMap, inverseImgMap)

def detectSeams(numSeamsx, numSeamsy, src, remove=True):
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	transportMap = np.ones((numSeamsx+1, numSeamsy+1))
	bitMap = np.ones((numSeamsx+1, numSeamsy+1), dtype=bool)
	seamPointsList = [[None for _ in range(numSeamsy+1)] for _ in range(numSeamsx+1)]
	bwList = [None]*(numSeamsx+1)
	transportMap[0, 0] = 0
	bwList[0] = bw
	for x in range(numSeamsx):
		(bwList[x+1], minEnergyVal, seamPointsList[x+1][0]) = detectMinSeam(bwList[x], alongX = True)
		transportMap[x+1, 0] = minEnergyVal
		bitMap[x+1, 0] = True
	for y in range(numSeamsy):
		(bwList[0], minEnergyVal, seamPointsList[0][y+1]) = detectMinSeam(bwList[0], alongX = False)
		transportMap[0, y+1] = minEnergyVal
		bitMap[0, y+1] = False
		for x in range(numSeamsx):
			(bwx, minEnergyValx, seamPointsx) = detectMinSeam(bwList[x], alongX = True)
			(bwy, minEnergyValy, seamPointsy) = detectMinSeam(bwList[x+1], alongX = False)
			if((minEnergyValx + transportMap[x, y+1]) < (minEnergyValy + transportMap[x+1, y])):
				bwList[x+1] = bwx
				transportMap[x+1, y+1] = (minEnergyValx + transportMap[x, y+1])
				bitMap[x+1, y+1] = True
				seamPointsList[x+1][y+1] = seamPointsx
			else:
				bwList[x+1] = bwy
				transportMap[x+1, y+1] = (minEnergyValy + transportMap[x+1, y])
				bitMap[x+1, y+1] = False
				seamPointsList[x+1][y+1] = seamPointsy
	(i, j, seamsOrder, seamsOptimalList) = (numSeamsx, numSeamsy, [], [])
	while(not (i==0 and j==0)):
		if(bitMap[i, j]):
			i-=1
			seamsOrder.append(True)
			seamsOptimalList.append(seamPointsList[i+1][j])
		else:
			j-=1
			seamsOrder.append(False)
			seamsOptimalList.append(seamPointsList[i][j+1])
	seamsOrder.reverse()
	seamsOptimalList.reverse()
	(srcSeamMap, inverseImgMap) = initialization(bw.shape) 
	(index, output, bw) = (0, src.copy(), cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
	for index in range(len(seamsOrder)):
		if(remove):
			(bw, inverseImgMap, output) = removeMinSeam(bw, inverseImgMap, seamsOrder[index], srcSeamMap, output, seamsOptimalList[index], index)
		else:
			(bw, inverseImgMap, output) = addMinSeam(bw, inverseImgMap, seamsOrder[index], srcSeamMap, output, seamsOptimalList[index], index)
	return (srcSeamMap, output)

def displaySeams(src, srcSeamMap, numSeamsx, numSeamsy):
	srcSeam = src.copy()
	for i in range(src.shape[0]):
		for j in range(src.shape[1]):
			if(srcSeamMap[i, j]<(numSeamsx+numSeamsy)):
				srcSeam[i, j] = [0, 0, 255]
				if(srcSeamMap[i, j]<0):
					srcSeam[i, j] = [0, 255, 0]
	return srcSeam

if __name__== "__main__":
	src = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_COLOR)
	(numSeamsx, numSeamsy) = (20, 10)
	(srcSeamMap, output) = detectSeams(numSeamsx, numSeamsy, src, remove=True)
	srcSeam = displaySeams(src, srcSeamMap, numSeamsx, numSeamsy)
	cv2.imshow("src", src)
	cv2.imshow("output", output)
	cv2.imshow("srcSeams", srcSeam)
	cv2.waitKey(0)
	cv2.destroyAllWindows()