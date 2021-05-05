import cv2
import numpy as np
import time

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
image = cv2.imread("./sampleImages/s1.jpg", cv2.IMREAD_COLOR)
bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(bw)
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# show the images
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)
