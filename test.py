import cv2
import numpy as np
 
# Picture path
img = cv2.imread('s2.jpg')
a = []
b = []
 
drawing = False

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        drawing = True
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        # cv2.imshow("image", img)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            xy = "%d, %d" %  (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (0,0,255), thickness = -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        xy = "%d, %d" %  (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x,y), 1, (0,0,255), thickness = -1)
    cv2.imshow("image", img)

 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('a'):
        break

# print(a, b)

l, w, h = img.shape

rmask = np.ones((l,w))*0

for i in range(len(a)):
    rmask[b[i]][a[i]] = 255

cv2.imshow("rmask", rmask)
cv2.waitKey()
rmask.reshape((l, w, 1))
cv2.imwrite("rmask.jpg", rmask)
cv2.destroyAllWindows()

