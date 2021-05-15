import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def saliency(gray):
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(gray)
	return saliencyMap

def energyL1(gray):
	grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
	grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)
	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
	return grad

def energyFunction(gray, mask):
	energy = saliency(gray).astype('float64')
	# energy = energyL1(gray).astype('float64')	
	energy[np.where(mask>0)] = -1000000
	# np.set_printoptions(threshold=sys.maxsize)
	# print(energy[-1, :])
	return energy

# def energyMeasure(img):
# 	return (np.sum(energyFunction(img))/(img.shape[0]*img.shape[1]))

def cumulativeMinEnergy(e1, alongX):
	minEnergy = np.zeros(e1.shape, dtype='int')

	nx = (e1.shape)[0]
	ny = (e1.shape)[1]

	if(alongX):
		for x in range(nx):
			minEnergy[x][0] = e1[x][0]

		for y in range(1, ny):
			for x in range(nx):
				minVal = minEnergy[x][y-1]
				if(x>0):
					minVal = min(minVal, minEnergy[x-1][y-1])
				if(x<(nx-1)):
					minVal = min(minVal, minEnergy[x+1][y-1])
				minVal+=e1[x][y]
				minEnergy[x][y]=minVal
	else:
		for y in range(ny):
			minEnergy[0][y] = e1[0][y]

		for x in range(1, nx):
			for y in range(ny):
				minVal = minEnergy[x-1][y]
				if(y>0):
					minVal = min(minVal, minEnergy[x-1][y-1])
				if(y<(ny-1)):
					minVal = min(minVal, minEnergy[x-1][y+1])
				minVal+=e1[x][y]
				minEnergy[x][y]=minVal
	# np.set_printoptions(threshold=sys.maxsize)
	# print(minEnergy[:, -1])
	return minEnergy

def findMinSeam(minEnergy, alongX):
	seamPoints = []
	nx = (minEnergy.shape)[0]
	ny = (minEnergy.shape)[1]

	if(alongX):
		minX = -1
		minVal = 255*(nx+ny)
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
		minY = -1
		minVal = 255*(nx+ny)
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

def addSrcImgPoints(img, seamPoints, alongX):
	if(alongX):
		newImg = np.insert(img, -1, 1, axis=0)
		(ymax, xmax) = (0, 0)
		for (x, y) in seamPoints:
			if(x==0):
				newImg[x, y] = img[x, y]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[x, y] = cv2.addWeighted(img[x, y], 0.5, img[x-1, y], 0.5, 0).reshape((3))
			for i in range(x, img.shape[0]):
				newImg[i+1, y] = img[i, y]
			if(y>ymax):
				(xmax, ymax) = (x, y)
		for y in range(ymax, img.shape[1]):
			if(xmax==0):
				newImg[xmax, y] = img[xmax, y]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[xmax, y] = cv2.addWeighted(img[xmax, y], 0.5, img[xmax-1, y], 0.5, 0).reshape((3))
			for i in range(xmax, img.shape[0]):
				newImg[i+1, y] = img[i, y]
	else:
		newImg = np.insert(img, -1, 1, axis=1)
		(ymax, xmax) = (0, 0)
		for (x, y) in seamPoints:
			if(y==0):
				newImg[x, y] = img[x, y]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[x, y] = cv2.addWeighted(img[x, y], 0.5, img[x, y-1], 0.5, 0).reshape((3))
			for i in range(y, img.shape[1]):
				newImg[x, i+1] = img[x, i]
			if(x>xmax):
				(xmax, ymax) = (x, y)
		for x in range(xmax, img.shape[0]):
			if(ymax==0):
				newImg[x, ymax] = img[x, ymax]
			else:
				# TODO: ASK: IS AVERAGE CORRECT?
				newImg[x, ymax] = cv2.addWeighted(img[x, ymax], 0.5, img[x, ymax-1], 0.5, 0).reshape((3))
			for i in range(ymax, img.shape[1]):
				newImg[x, i+1] = img[x, i]
	return newImg

def removeMinSeam(inverseImgMap, alongX, srcSeamMap, output, seamPoints, index, mask):
	updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index, alongX)
	newInverseMap = deletePoints(inverseImgMap, seamPoints, alongX)
	output = deletePoints(output, seamPoints, alongX)
	newMask = deletePoints(mask, seamPoints, alongX)
	return (newInverseMap, output, newMask)

def addMinSeam(inverseImgMap, alongX, srcSeamMap, output, seamPoints, index):
	updateSeamMap(srcSeamMap, inverseImgMap, seamPoints, index, alongX)
	newInverseMap = deletePoints(inverseImgMap, seamPoints, alongX)
	output = addSrcImgPoints(output, seamPoints, alongX)
	return (newInverseMap, output)

def detectMinSeam(img, mask, alongX):
	energy = energyFunction(img, mask)
	minEnergy = cumulativeMinEnergy(energy, alongX)
	(seamPoints, minEnergyVal) = findMinSeam(minEnergy, alongX)
	newImg = deletePoints(img, seamPoints, alongX)
	newMask = deletePoints(mask, seamPoints, alongX)
	return (newImg, newMask, minEnergyVal, seamPoints)

def initialization(shape):
	srcSeamMap = np.ones(shape, dtype='int')*(shape[0] + shape[1])
	inverseImgMap = np.empty(shape, dtype=object)
	for i in range(shape[0]):
		for j in range(shape[1]):
			inverseImgMap[i, j] = (i, j)
	return (srcSeamMap, inverseImgMap)

def detectSeams(numSeamsx, numSeamsy, src, mask, remove=True):
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	transportMap = np.ones((numSeamsx+1, numSeamsy+1))
	bitMap = np.ones((numSeamsx+1, numSeamsy+1), dtype=bool)
	seamPointsList = [[None for _ in range(numSeamsy+1)] for _ in range(numSeamsx+1)]
	bwList = [None]*(numSeamsx+1)
	maskList = [None]*(numSeamsx+1)
	transportMap[0, 0] = 0
	bwList[0] = bw
	maskList[0] = mask
	for x in range(numSeamsx):
		(bwList[x+1], maskList[x+1], transportMap[x+1, 0], seamPointsList[x+1][0]) = detectMinSeam(bwList[x], maskList[x], alongX = True)
		bitMap[x+1, 0] = True
	for y in range(numSeamsy):
		(bwList[0], maskList[0], transportMap[0, y+1], seamPointsList[0][y+1]) = detectMinSeam(bwList[0], maskList[0], alongX = False)
		bitMap[0, y+1] = False
		for x in range(numSeamsx):
			(bwx, maskx, minEnergyValx, seamPointsx) = detectMinSeam(bwList[x], maskList[x], alongX = True)
			(bwy, masky, minEnergyValy, seamPointsy) = detectMinSeam(bwList[x+1], maskList[x+1], alongX = False)
			if((minEnergyValx + transportMap[x, y+1]) < (minEnergyValy + transportMap[x+1, y])):
				bwList[x+1] = bwx
				transportMap[x+1, y+1] = (minEnergyValx + transportMap[x, y+1])
				bitMap[x+1, y+1] = True
				seamPointsList[x+1][y+1] = seamPointsx
				maskList[x+1] = maskx
			else:
				bwList[x+1] = bwy
				transportMap[x+1, y+1] = (minEnergyValy + transportMap[x+1, y])
				bitMap[x+1, y+1] = False
				seamPointsList[x+1][y+1] = seamPointsy
				maskList[x+1] = masky
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
	newMask = mask.copy()
	seamsOrder.reverse()
	seamsOptimalList.reverse()
	(srcSeamMap, inverseImgMap) = initialization(bw.shape) 
	(index, output) = (0, src.copy())
	# (numSeams, imageEnergy) = ([], [])
	for index in range(len(seamsOrder)):
		if(remove):
			(inverseImgMap, output, newMask) = removeMinSeam(inverseImgMap, seamsOrder[index], srcSeamMap, output, seamsOptimalList[index], index, newMask)
		else:
			(inverseImgMap, output) = addMinSeam(inverseImgMap, seamsOrder[index], srcSeamMap, output, seamsOptimalList[index], index)
		# numSeams.append(index)
		# imageEnergy.append(energyMeasure(output))
	# plt.plot(numSeams, imageEnergy)
	# plt.xlabel('reduction in img size')
	# plt.ylabel('image energy')
	# plt.title('Image energy function vs number of seams')
	# plt.show()
	return (srcSeamMap, seamsOrder, output, newMask)

def displaySeams(src, srcSeamMap, seamsOrder, numSeamsx, numSeamsy):
	srcSeam = src.copy()
	for x in range(len(seamsOrder)):
		for i in range(src.shape[0]):
			for j in range(src.shape[1]):
				if(srcSeamMap[i, j]==x):
					srcSeam[i, j] = [0, 0, 255] 
					if(not seamsOrder[x]):
						srcSeam[i, j] = [0, 255, 0]
		cv2.imshow("srcSeams", srcSeam)
		cv2.waitKey(500)
	return srcSeam

drawing=False
def readMask(src):
	def draw(event, former_x, former_y, flags, param):
		global current_former_x,current_former_y,drawing
		if event==cv2.EVENT_LBUTTONDOWN:
			drawing=True
			current_former_x,current_former_y=former_x,former_y

		elif event==cv2.EVENT_MOUSEMOVE:
			if drawing==True:
				cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
				cv2.line(mask,(current_former_x,current_former_y),(former_x,former_y),255,5)
				current_former_x = former_x
				current_former_y = former_y
				#print former_x,former_y
		elif event==cv2.EVENT_LBUTTONUP:
			drawing=False
			cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
			cv2.line(mask,(current_former_x,current_former_y),(former_x,former_y),255,5)
			current_former_x = former_x
			current_former_y = former_y
		return former_x,former_y 

	im = src.copy()
	mask = np.zeros((src.shape[0], src.shape[1]))
	cv2.namedWindow("Shade object")
	cv2.setMouseCallback('Shade object',draw)
	while(1):
		cv2.imshow('Shade object',im)
		k=cv2.waitKey(1)&0xFF
		if k==27:
			break
	cv2.destroyAllWindows()
	return mask

if __name__== "__main__":
	src = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_COLOR)
	orimask = readMask(src)
	mask = orimask.copy()
	output = src.copy()
	(numSeamsx, numSeamsy) = (30, 0)
	p = 0
	while (len(np.where(mask > 0)[0]) > 0):
		if(p%2 == 0):
			(srcSeamMap, seamsOrder, output, mask) = detectSeams(1, 0, output, mask, remove=True)
		else:
			(srcSeamMap, seamsOrder, output, mask) = detectSeams(0, 1, output, mask, remove=True)
		p += 1	
		
	# (srcSeamMap, seamsOrder, output) = detectSeams(numSeamsx, numSeamsy, src, mask, remove=True)
	# srcSeam = displaySeams(src, srcSeamMap, seamsOrder, numSeamsx, numSeamsy)
	print(output.shape)
	print(src.shape)
	(diff_x, diff_y) = (src.shape[0] - output.shape[0], src.shape[1] - output.shape[1])
	print(diff_x, diff_y)
	# cv2.imshow("Before Insertion", output)
	cv2.imwrite("Before_Insertion.jpg", output)
	(srcSeamMap, seamsOrder, output, mask) = detectSeams(diff_x, diff_y, output, mask, remove=False)
	# cv2.imshow("src", src)
	# cv2.imshow("mask", orimask)
	cv2.imwrite("mask.jpg", orimask)
	# cv2.imshow("After Insertion", output)
	cv2.imwrite("After_Insertion.jpg", output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()