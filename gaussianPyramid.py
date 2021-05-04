import cv2
import numpy as np




if __name__== "__main__":
	src = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_COLOR)
	
	cv2.imshow("src", src)
	cv2.imshow("output", output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()