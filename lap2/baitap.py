import cv2
import numpy as np

filename = 'D:\download/tn.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result được mở rộng để đánh dấu các góc, không quan trọng
dst = cv2.dilate(dst,None)

# Ngưỡng cho một giá trị tối ưu, nó có thể khác nhau tùy thuộc vào hình ảnh.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()