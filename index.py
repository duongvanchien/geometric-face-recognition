import cv2
import dlib

# read the image
# img = cv2.imread("images/<20/3.jpg")
# img = cv2.imread("images/<20/3.jpg")
# img = cv2.imread("face_age/092/202.png")
# img = cv2.imread("images/test2.webp")

# convert image from RGB -> GRAY 
img = cv2.imread("face_age/001/16.png")
# img = cv2.imread("images/test2.webp")

# convert image from 3D -> 2D
convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# load face detector
face_detector = dlib.get_frontal_face_detector()

# load face recognition
faces = face_detector(convert_img)

# load the predictor:
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_features = predictor(image=convert_img, box=faces[0])

# Loop through all 68 points
for n in range(0, 68):
    x = face_features.part(n).x
    y = face_features.part(n).y
    print(x, y, n)
    # Draw a circle
    cv2.circle(img=img, center=(x, y), radius=2, color=(0,255,0), thickness=1)

# show image
cv2.imshow(winname="Face Image", mat=img)

# wait for a key press to exit
cv2.waitKey(delay=0)


# close all window
cv2.destroyAllWindows()
