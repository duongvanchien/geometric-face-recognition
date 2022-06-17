import cv2
import numpy as np
from matplotlib import image, pyplot as plt
import dlib
import csv
import os


# load the predictor:
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

result=[]

def load_images_from_folder(folder):
    files = []
    for filename in os.listdir(folder):
        files.append(os.path.join(folder,filename))   
    return files

def feature_extraction(folder, label):
    images = load_images_from_folder(folder)
    for item in images:
        img = cv2.imread(item)
        print(item)
        # convert image from RGB -> GRAY 
        convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        
        # load face recognition
        faces = face_detector(convert_img)
        
        #get face landmarks
        face_features = predictor(image=convert_img, box=faces[0])

        edge_img = cv2.Canny(convert_img, 50, 50)

        #geomatric feature
        d_en = face_features.part(30).y - face_features.part(27).y
        d_nm = face_features.part(62).y - face_features.part(30).y
        geomatric = d_en/d_nm

        #forehead feature
        forehead = edge_img[face_features.part(71).y:face_features.part(19).y, face_features.part(19).x:face_features.part(24).x]
        forehead_wrinkle_percentage = np.sum(forehead>0)*100/(forehead.shape[0]*forehead.shape[1])
        # forehead_wrinkle = cv2.Canny(forehead,50,50)
        # forehead_wrinkle_percentage = np.sum(forehead_wrinkle>0)*100/(forehead_wrinkle.shape[0]*forehead_wrinkle.shape[1])

        #left under eye feature
        left_under_eye = edge_img[face_features.part(40).y:face_features.part(29).y, face_features.part(18).x:face_features.part(21).x]
        left_under_eye_wrinkle_percentage = np.sum(left_under_eye>0)*100/(left_under_eye.shape[0]*left_under_eye.shape[1])
        # left_under_eye_wrinkle = cv2.Canny(left_under_eye,50,50)
        # left_under_eye_wrinkle_percentage = np.sum(left_under_eye_wrinkle>0)*100/(left_under_eye_wrinkle.shape[0]*left_under_eye_wrinkle.shape[1])

        #right under eye feature
        right_under_eye = edge_img[face_features.part(47).y:face_features.part(29).y, face_features.part(22).x:face_features.part(25).x]
        right_under_eye_wrinkle_percentage = np.sum(right_under_eye>0)*100/(right_under_eye.shape[0]*right_under_eye.shape[1])
        # right_under_eye_wrinkle = cv2.Canny(right_under_eye,50,50)
        # right_under_eye_wrinkle_percentage = np.sum(right_under_eye_wrinkle>0)*100/(right_under_eye_wrinkle.shape[0]*right_under_eye_wrinkle.shape[1])
        
        #left cheek 
        left_cheek = edge_img[face_features.part(29).y:face_features.part(4).y, face_features.part(4).x:face_features.part(48).x]
        left_cheek_wrinkle_percentage = np.sum(left_cheek>0)*100/(left_cheek.shape[0]*left_cheek.shape[1])
        # left_cheek_wrinkle = cv2.Canny(left_cheek,50,50)
        # left_cheek_wrinkle_percentage = np.sum(left_cheek_wrinkle>0)*100/(left_cheek_wrinkle.shape[0]*left_cheek_wrinkle.shape[1])

        #right cheek
        right_cheek = edge_img[face_features.part(29).y:face_features.part(12).y, face_features.part(54).x:face_features.part(13).x]
        right_cheek_wrinkle_percentage = np.sum(right_cheek>0)*100/(right_cheek.shape[0]*right_cheek.shape[1])
        # right_cheek_wrinkle = cv2.Canny(right_cheek,50,50)
        # right_cheek_wrinkle_percentage = np.sum(right_cheek_wrinkle>0)*100/(right_cheek_wrinkle.shape[0]*right_cheek_wrinkle.shape[1])  

        #left-eye-edge
        left_eye_edge = edge_img[face_features.part(17).y:face_features.part(29).y, face_features.part(75).x:face_features.part(36).x]
        left_eye_edge_wrinkle_percentage = np.sum(left_eye_edge>0)*100/(left_eye_edge.shape[0]*left_eye_edge.shape[1])
        # left_eye_edge_wrinkle = cv2.Canny(left_eye_edge,50,50)
        # left_eye_edge_wrinkle_percentage = np.sum(left_eye_edge_wrinkle>0)*100/(left_eye_edge_wrinkle.shape[0]*left_eye_edge_wrinkle.shape[1])  

        #right-eye-edge
        right_eye_edge = edge_img[face_features.part(26).y:face_features.part(29).y, face_features.part(45).x:face_features.part(74).x]
        right_eye_edge_wrinkle_percentage = np.sum(right_eye_edge>0)*100/(right_eye_edge.shape[0]*right_eye_edge.shape[1])
        # right_eye_edge_wrinkle = cv2.Canny(right_eye_edge,50,50)
        # right_eye_edge_wrinkle_percentage = np.sum(right_eye_edge_wrinkle>0)*100/(right_eye_edge_wrinkle.shape[0]*right_eye_edge_wrinkle.shape[1])  

        result.append({
            "path": item,
            "geomatric": geomatric,
            "forehead": forehead_wrinkle_percentage,
            "l_under_eye": left_under_eye_wrinkle_percentage,
            "r_under_eye": right_under_eye_wrinkle_percentage,
            "l_cheek": left_cheek_wrinkle_percentage,
            "r_cheek": right_cheek_wrinkle_percentage,
            "l_eye_edge": left_eye_edge_wrinkle_percentage,
            "r_eye_edge": right_eye_edge_wrinkle_percentage,
            "class": label
        })


def write_file():
    header=["path", "geomatric", "forehead", "left-under-eye", "right-under-eye", "left-cheek", "right-cheek", "left-eye-edge", "right-eye-edge", "class"]
    with open('data-set.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        #write rows
        for row in result:
            writer.writerow([row["path"],row["geomatric"],row["forehead"],row["l_under_eye"],                       \
            row["r_under_eye"],row["l_cheek"],row["r_cheek"],row["l_eye_edge"],row["r_eye_edge"],row["class"]])     \


feature_extraction("img\\under-20","under-20")
feature_extraction("img\\20-45","20-45")
feature_extraction("img\\over-45","over-45")
write_file()