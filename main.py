import cv2
from matplotlib import pyplot as plt

img = cv2.imread('resource/testImages/digitrecognition1.jpg', 0)

blur = cv2.GaussianBlur(img,(5,5),0)
canny = cv2.Canny(img, 100, 200)
img_c, contours, hierarchy = cv2.findContours(canny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_c, contours, -1, (0,255,0), 1)

rects = [cv2.boundingRect(contour) for contour in contours];
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit

    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = img[pt1:pt1 + leng, pt2:pt2 + leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_LINEAR)
    #roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features

cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.waitKey()