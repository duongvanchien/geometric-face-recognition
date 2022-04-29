import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread("face_age/test.png")
# img = cv2.imread("face_age/056/67.png")
# img = cv2.imread("face_age/001/16.png")
img = cv2.imread("face_age/092/202.png")

# img = cv2.imread('test.jpg')
print(img.shape) # Print image shape
# cv2.imshow("original", img)

# # Cropping an image
cropped_image = img[0:30, 41:170]

# # # Display cropped image
# cv2.imshow("cropped", cropped_image)

# # # Save the cropped image
# cv2.imwrite("Cropped Image.jpg", cropped_image)

img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 50, 100)


titles = ['image', 'canny']
images = [img, canny]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
# 29 32 / 97 59

# cv2.waitKey(0)
# cv2.destroyAllWindows()