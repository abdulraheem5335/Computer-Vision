import cv2


image = cv2.imread('logo.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Logo BGR', image)
cv2.imshow('Logo Gray', gray_image)


cv2.waitKey(0)
cv2.destroyAllWindows()