import cv2
import numpy as np
img = cv2.imread("picture.png")

h, w = img.shape[:2]  # height, width
ctr = (w // 2, h // 2)
ang = 45
scl = 1.0

mat = cv2.getRotationMatrix2D(ctr, ang, scl)
rot = cv2.warpAffine(img, mat, (w, h))

cv2.imshow("Rotated Image", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("rotated_picture.png", rot)