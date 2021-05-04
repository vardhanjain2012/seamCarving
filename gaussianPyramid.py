import cv2
import numpy as np


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
		minVal = 255*(nx+ny)
		for x in range(nx):
			if(minVal>minEnergy[x][ny-1]):
				minVal=minEnergy[x][ny-1]
				minX = x
				print(minVal, x)
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

def removeMinSeam(img, inverseImgMap, alongX, srcSeamMap, output, index):
	energy = energyE1(img)
	minEnergy = cumulativeMinEnergy(energy, alongX)
	(seamPoints, minEnergyVal) = findMinSeam(minEnergy, alongX)
	newImg = deletePoints(img, seamPoints, alongX)
	newInverseMap = deletePoints(inverseImgMap, seamPoints, alongX)
	output = deletePoints(output, seamPoints, alongX)
	return (newImg, newInverseMap, output)

myMat = np.array([[1, 2, 7], [9, 4, 5], [6, 2, 4]])
print(myMat)
alongX = False
gg = cumulativeMinEnergy(myMat, alongX)
print(gg)

(seamPoints, minEnergyVal) = findMinSeam(gg, alongX)

print(deletePoints(myMat, seamPoints, alongX))