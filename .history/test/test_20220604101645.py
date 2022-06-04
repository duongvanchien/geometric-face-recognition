import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib

# read the image
# img = cv2.imread("images/<20/3.jpg")
# img = cv2.imread("images/<20/3.jpg")
img = cv2.imread("./abc/01267.png")
# img = cv2.imread("images/test2.webp")

# convert image from RGB -> GRAY 
convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

img0 = cv2.GaussianBlur(convert_img,(3,3),0)
# load face detector
face_detector = dlib.get_frontal_face_detector()

# load face recognition
faces = face_detector(convert_img)

# load the predictor:
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

face_features = predictor(image=convert_img, box=faces[0])

for n in range(0, 81):
        # if n == 30 or n == 27 or n == 51: 
            x = face_features.part(n).x
            y = face_features.part(n).y
        #     print(x, y, n)
        #     # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=2, color=(0,255,0), thickness=1)



# forehead = img0[face_features.part(71).y:face_features.part(19).y, face_features.part(19).x:face_features.part(24).x]

# edge_forehead = cv2.Canny(forehead,50,50)
# forehead_pixel_percentage = np.sum(edge_forehead>0)/(edge_forehead.shape[0]*edge_forehead.shape[1])


# right_eye_edge = img0[face_features.part(17).y:face_features.part(41).y, face_features.part(75).x:face_features.part(36).x]
# edge_eye_right = edge_forehead = cv2.Canny(right_eye_edge,50,50)
# right_percentage = np.sum(edge_eye_right>0)/(edge_eye_right.shape[0]*edge_eye_right.shape[1])


# d_en = face_features.part(30).y - face_features.part(27).y
# d_nm = face_features.part(51).y - face_features.part(30).y
# print(d_en/d_nm)

# print("percentage of forehead wrinkle: ", forehead_pixel_percentage )
# print("percentage of right edge wrinkle: ", right_percentage )
# # show image
# cv2.imshow("Face Image", forehead)

# cv2.waitKey(delay=0)

# # cv2.imshow("forehead", edge_forehead)

# cv2.imshow("right eye edge", edge_eye_right)
# # wait for a key press to exit
# cv2.waitKey(delay=0)


under_left_eye = img0[(face_features.part(40).y):face_features.part(29).y, face_features.part(18).x:face_features.part(21).x]
under_right_eye = img0[(face_features.part(47).y):face_features.part(29).y, face_features.part(22).x:face_features.part(25).x]

cv2.rectangle(img, (face_features.part(18).x, face_features.part(40).y), (face_features.part(21).x, face_features.part(29).y), (255,0,0), 2)
cv2.rectangle(img, (face_features.part(22).x, face_features.part(47).y), (face_features.part(25).x, face_features.part(29).y), (255,0,0), 2)

cv2.rectangle(img, (face_features.part(4).x, face_features.part(29).y), (face_features.part(48).x, face_features.part(4).y), (255,0,0), 2)
cv2.rectangle(img, (face_features.part(54).x, face_features.part(29).y), (face_features.part(13).x, face_features.part(12).y), (255,0,0), 2)


cv2.imshow('img', img)

# plt.subplot(1,5,1),plt.imshow(img0,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,2),plt.imshow(under_left_eye,cmap = 'gray')
# plt.title('Left eye'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,3),plt.imshow(under_right_eye,cmap = 'gray')
# plt.title('Right eye'), plt.xticks([]), plt.yticks([])

# plt.show()
# cv2.imshow("right eye edge", under_left_eye)
# # wait for a key press to exit
cv2.waitKey(delay=0)





# # close all window
# cv2.destroyAllWindows()




