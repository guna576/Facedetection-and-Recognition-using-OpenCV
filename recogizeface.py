# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:35:09 2020

@author: Gunashekar Chenna


"""

  
import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    #Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    #drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        # Check for id of user and label the rectangle accordingly
        if id==1:
            cv2.putText(img, "Shivani", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        if id==2:
            cv2.putText(img, "srikala", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        if id==3:
            cv2.putText(img, "guna", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
      
        coords = [x, y, w, h]
        

    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255), "black": (0,0,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")


video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = recognize(img, clf, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("recognizing...", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()

cv2.destroyAllWindows()