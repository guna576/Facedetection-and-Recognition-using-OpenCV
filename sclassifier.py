# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:03:12 2020

@author: Gunashekar Chenna
"""
#importing packages
import numpy as np
from PIL import Image
import os
import cv2

#we call the function inorder to train the classifier 

def train_classifier(data_dir):
    
    #we created a list of images with paths
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L') #open and convert image in black and white
        #we appended the each image in numpy array
        imageNp = np.array(img , 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        
        faces.append(imageNp)
        ids.append(id)
        
    ids = np.array(ids)
    #the below function in face library helps to recognize the faces
    clf = cv2.face.LBPHFaceRecognizer_create()
    
    
    clf.train(faces,ids)
    
    #we write the data in the xml classifier
    
    clf.write("classifier.xml")
    
 #we give the directory to the function to train data   
train_classifier(r"C:\Users\Gunashekar Chenna\Desktop\python\data")
    