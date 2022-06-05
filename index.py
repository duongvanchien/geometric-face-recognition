from lib2to3.pytree import convert
import cv2
import dlib
import os
from matplotlib import pyplot as plt

#part1: build metadata
#step 1: load folder images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
imgs = load_images_from_folder("./human/duoi 20")
#step 2: loop each folder 'duoi 20', '20 - 45', '> 45'

def img_detector(img):
    convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(convert_img)
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmark.dat")
    face_features = predictor(image=convert_img, box=faces[0])
    return convert_img,face_features

def get_value_points_face(point):
    for key,value in points_area_face:
        if(value.__contains__(point)):
            return key,point
    return None


points_area_face = {
    "left_cheek": [1,3,48],
    "right_cheek": [15,13,54]
}

def handle_imgs(imgs):
    coordinates_face = dict()

    for img in imgs:
        convert_image,face_features = img_detector(img)
        dic = dict()
        for point in range(0, 81):
            #step 3: get coordinates need to crop areas
            if get_value_points_face(point): 
                dic["left_cheek"] = [face_features.part(n).x,face_features.part(n).y]
        #step 4: crop area and using canny algorithm detect edge
        print(dic)
        lcheek_img = img[dic[3][0]:dic[48][0], dic[1][1]:dic[3][1]]
        cv2.imshow("cropped",lcheek_img)
        cv2.waitKey(0)


handle_imgs(imgs=imgs)
#step 5: determined ratia area: ratio = count(pixel_edge)/count(pixel_area)
#step 6: save data: [area: ratio, label = ('duoi 20' or '20 -45' or '> 45')]

#part2: estimate age (later)

