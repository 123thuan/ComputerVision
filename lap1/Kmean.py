import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# đọc file ảnh
image = cv2.imread("D:\download/tn.png", 1)

# chuyển đổi ảnh về ảnh xám
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# chuyển đổi ảnh 3d về 2d có 3 màu RGB
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
#Xác định tiêu chí kết thúc thuật toán: số lần lặp tối đa được đặt thành 100 và epsilon = 0,2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# lam phẳng mảng nhãn
labels = labels.flatten()

# chuyển đổi tất cả các pixel thành màu của trung tâm
segmented_image = centers[labels]

# định hình lại kích thước hình ảnh ban đầu
segmented_image = segmented_image.reshape(image.shape)

# show the image
plt.imshow(segmented_image)
plt.show()

# chỉ vô hiệu hóa cụm số 2 (chuyển pixel thành màu đen)
masked_image = np.copy(image)
# chuyển đổi sang hình dạng của một vectơ có giá trị pixel
masked_image = masked_image.reshape((-1, 3))
#hien ra 2 cụm
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]

# chuyển đổi trở lại hình dạng ban đầu
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()