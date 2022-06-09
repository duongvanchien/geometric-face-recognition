import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib

# read the image
# img = cv2.imread("images/<20/3.jpg")
# img = cv2.imread("images/<20/3.jpg")
img = cv2.imread("./test/01267.png")
# img = cv2.imread("images/test2.webp")

# convert image from RGB -> GRAY 
convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)


# load face detector
face_detector = dlib.get_frontal_face_detector()

# load face recognition
faces = face_detector(convert_img)

# load the predictor:
predictor = dlib.shape_predictor("shape_predictor_81_face_landmark.dat")

face_features = predictor(image=convert_img, box=faces[0])

# for n in range(0, 81):
    
#             x = face_features.part(n).x
#             y = face_features.part(n).y

#         #     # Draw a circle
#             cv2.circle(img=img, center=(x, y), radius=2, color=(0,255,0), thickness=1)
#             cv2.putText(img, str(n), (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)



img = cv2.Canny(img, 100, 120)


under_left_eye = img[(face_features.part(40).y):face_features.part(29).y, face_features.part(18).x:face_features.part(21).x]
under_right_eye = img[(face_features.part(47).y):face_features.part(29).y, face_features.part(22).x:face_features.part(25).x]

#under eye
cv2.rectangle(img, (face_features.part(18).x, face_features.part(40).y), (face_features.part(21).x, face_features.part(29).y), (255,0,0), 2)
cv2.rectangle(img, (face_features.part(22).x, face_features.part(47).y), (face_features.part(25).x, face_features.part(29).y), (255,0,0), 2)

#cheek
cv2.rectangle(img, (face_features.part(4).x, face_features.part(29).y), (face_features.part(48).x, face_features.part(4).y), (255,0,0), 2)
cv2.rectangle(img, (face_features.part(54).x, face_features.part(29).y), (face_features.part(13).x, face_features.part(12).y), (255,0,0), 2)

#forehead
cv2.rectangle(img, (face_features.part(19).x, face_features.part(71).y), (face_features.part(24).x, face_features.part(19).y), (255,0,0), 2)

#left-eye-edge
cv2.rectangle(img, (face_features.part(75).x, face_features.part(17).y), (face_features.part(36).x, face_features.part(29).y), (255,0,0), 2)

#right-eye-edge
cv2.rectangle(img, (face_features.part(45).x, face_features.part(26).y), (face_features.part(74).x, face_features.part(29).y), (255,0,0), 2)



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




