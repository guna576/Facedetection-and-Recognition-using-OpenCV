# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:45:19 2020

@author: Gunashekar Chenna
"""

import cv2

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color,text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    
    # drawing rectangle around the face
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "")
    
    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Capturing real time
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    #key q will close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()

cv2.destroyAllWindows()