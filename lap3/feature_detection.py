import cv2
import matplotlib.pyplot as plt
import numpy as np


# SIFT (Scale-Invariant Feature Transform)
def siftFeatureDetector(image):
    img1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # call function to create object
    sift = cv2.SIFT_create(2400)

    points = sift.detect(img1_gray, None)#tìm ddiemr quan trọng trong ảnh xám

    points_img = np.copy(image)

    cv2.drawKeypoints(image, points, points_img, color=(255, 0, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#vẽ lên diem  tim thấy

    return points_img


img = cv2.imread("D:\download/tn.png")
img_keypoint = siftFeatureDetector(img)

fig = plt.figure(figsize=(8, 5))
ax1, ax2 = fig.subplots(1, 2)

ax1.imshow(img)
ax1.set_title('Ảnh gốc')
ax1.axis('off')

ax2.imshow(img_keypoint)
ax2.set_title('Ảnh chứa keypoint')
ax2.axis('off')
plt.show()