import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from network import Network

img = cv2.imread('testImages/ex10.jpg', 0)
img = cv2.GaussianBlur(img,(5,5),0)
ret, im_th = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
_, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
#canny = cv2.Canny(img, 100, 200)
_, contours, hierarchy = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img_c, contours, -1, (0,255,0), 1)
net = Network([784,100,10])
#net.train(60,100)
rects = [cv2.boundingRect(contour) for contour in contours]
output = []
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit

    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    #out[pt1:pt1 + leng, pt2:pt2 + leng] = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi = roi.reshape([1, 784])
    tensorInput = tf.convert_to_tensor(roi, dtype=tf.float32)
    val = net.predict(tensorInput)[0]
    output.append(val)
    print val
    cv2.putText(out, str(val), (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

#plotting image
plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(out, cmap='gray'),plt.title('Processed')
plt.xticks([]), plt.yticks([])
plt.show()