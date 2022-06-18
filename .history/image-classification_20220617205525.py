import cv2
import numpy as np
from matplotlib import image, pyplot as plt
import dlib
import csv
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")




def initModel():
    data_path = "data.csv"
    ds = pd.read_csv(data_path,usecols = ["geomatric", "forehead", "left-under-eye", "right-under-eye", "left-cheek", "right-cheek", "left-eye-edge", "right-eye-edge", "class"])
    X = ds.iloc[:,:8]
    y = ds.iloc[:,-1]
    knnmodel=KNeighborsClassifier(n_neighbors=10)
    knnmodel.fit(X.values,y)
    return knnmodel


def feature_extraction(file):   
    img = cv2.imread(file)
    # convert image from RGB -> GRAY 
    convert_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    
    # load face recognition
    faces = face_detector(convert_img)
    
    #get face landmarks
    face_features = predictor(image=convert_img, box=faces[0])

    #canny edge detection
    edge_img = cv2.Canny(convert_img, 45, 55)

    #geomatric feature
    d_en = face_features.part(30).y - face_features.part(27).y
    d_nm = face_features.part(62).y - face_features.part(30).y
    geomatric = d_en/d_nm

    #forehead feature
    forehead = edge_img[face_features.part(71).y:face_features.part(19).y, face_features.part(19).x:face_features.part(24).x]
    forehead_wrinkle_percentage = np.sum(forehead>0)*100/(forehead.shape[0]*forehead.shape[1])

    #left under eye feature
    left_under_eye = edge_img[face_features.part(40).y:face_features.part(29).y, face_features.part(18).x:face_features.part(21).x]
    left_under_eye_wrinkle_percentage = np.sum(left_under_eye>0)*100/(left_under_eye.shape[0]*left_under_eye.shape[1])

    #right under eye feature
    right_under_eye = edge_img[face_features.part(47).y:face_features.part(29).y, face_features.part(22).x:face_features.part(25).x]
    right_under_eye_wrinkle_percentage = np.sum(right_under_eye>0)*100/(right_under_eye.shape[0]*right_under_eye.shape[1])

    #left cheek 
    left_cheek = edge_img[face_features.part(29).y:face_features.part(4).y, face_features.part(4).x:face_features.part(48).x]
    left_cheek_wrinkle_percentage = np.sum(left_cheek>0)*100/(left_cheek.shape[0]*left_cheek.shape[1])

    #right cheek
    right_cheek = edge_img[face_features.part(29).y:face_features.part(12).y, face_features.part(54).x:face_features.part(13).x]
    right_cheek_wrinkle_percentage = np.sum(right_cheek>0)*100/(right_cheek.shape[0]*right_cheek.shape[1])

    #left-eye-edge
    left_eye_edge = edge_img[face_features.part(17).y:face_features.part(29).y, face_features.part(75).x:face_features.part(36).x]
    left_eye_edge_wrinkle_percentage = np.sum(left_eye_edge>0)*100/(left_eye_edge.shape[0]*left_eye_edge.shape[1])

    #right-eye-edge
    right_eye_edge = edge_img[face_features.part(26).y:face_features.part(29).y, face_features.part(45).x:face_features.part(74).x]
    right_eye_edge_wrinkle_percentage = np.sum(right_eye_edge>0)*100/(right_eye_edge.shape[0]*right_eye_edge.shape[1])

    #draw feature zones
    cv2.rectangle(edge_img, (face_features.part(18).x, face_features.part(40).y), (face_features.part(21).x, face_features.part(29).y), (255,0,0), 2)
    cv2.rectangle(edge_img, (face_features.part(22).x, face_features.part(47).y), (face_features.part(25).x, face_features.part(29).y), (255,0,0), 2)
    #cheek
    cv2.rectangle(edge_img, (face_features.part(4).x, face_features.part(29).y), (face_features.part(48).x, face_features.part(4).y), (255,0,0), 2)
    cv2.rectangle(edge_img, (face_features.part(54).x, face_features.part(29).y), (face_features.part(13).x, face_features.part(12).y), (255,0,0), 2)
    #forehead
    cv2.rectangle(edge_img, (face_features.part(19).x, face_features.part(71).y), (face_features.part(24).x, face_features.part(19).y), (255,0,0), 2)
    #left-eye-edge
    cv2.rectangle(edge_img, (face_features.part(75).x, face_features.part(17).y), (face_features.part(36).x, face_features.part(29).y), (255,0,0), 2)
    #right-eye-edge
    cv2.rectangle(edge_img, (face_features.part(45).x, face_features.part(26).y), (face_features.part(74).x, face_features.part(29).y), (255,0,0), 2)
    
    
    return {
        "original_img": img,
        "edge_img": edge_img,
        "features": [geomatric, forehead_wrinkle_percentage, left_under_eye_wrinkle_percentage, right_under_eye_wrinkle_percentage, left_cheek_wrinkle_percentage, right_cheek_wrinkle_percentage, left_eye_edge_wrinkle_percentage, right_eye_edge_wrinkle_percentage]
    }


result=feature_extraction('abc\\07577.png')
cv2.imshow('img', result["edge_img"])
cv2.waitKey(delay=0)

model = initModel()
print(model.predict([result["features"]]))




