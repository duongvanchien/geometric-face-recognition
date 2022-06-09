import cv2
import dlib
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

imgs = load_images_from_folder("./face_age/037")

# convert image from RGB -> GRAY 
# img = cv2.imread("face_age/095/6403.png")
for img in imgs:
    # convert image from 3D -> 2D
    convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # load face detector
    face_detector = dlib.get_frontal_face_detector()

    # load face recognition
    faces = face_detector(convert_img)

    # load the predictor:
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmark.dat")

    face_features = predictor(image=convert_img, box=faces[0])

    d_en = face_features.part(30).y - face_features.part(27).y
    d_nm = face_features.part(51).y - face_features.part(30).y
    print(d_en/d_nm)

    # # Loop through all 81 points
    # for n in range(0, 81):
    #     if n == 30 or n == 27 or n == 51: 
    #         x = face_features.part(n).x
    #         y = face_features.part(n).y
    #         print(x, y, n)
    #         # Draw a circle
    #         cv2.circle(img=img, center=(x, y), radius=2, color=(0,255,0), thickness=1)

    # # show image
    # cv2.imshow(winname="Face Image", mat=img)

    # # wait for a key press to exit
    # cv2.waitKey(delay=0)


    # # close all window
    # cv2.destroyAllWindows()
