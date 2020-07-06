"""
Created on Sat May 29 11:10:45 2020

@author: Gunashekar Chenna
"""

import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("data/boy."+str(id)+"."+str(img_id)+".jpg",img)
    
def draw_boundary(img, classifier, scaleFactor, minNeighbors,color,text):
    #Converting image to gray-scale
    gray_img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY )
    #detecting features in gray-scale image, returns coordinates, width and height of features
    feautres = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    coords=[]
    
    for (x,y,w,h) in feautres:
         #drawing rectangle around the feature and labeling it
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0,0), 2)
        cv2.putText(img, text,(x,y-4), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords = [x,y,w,h]
        
    return coords, img

def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0) ,"red":(0,0,255),"green":(0,255,0)}
    
    coords, img = draw_boundary(img, faceCascade,1.1,10,color['green'],"")
    
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        #giving id to each person
        user_id = 3
        generate_dataset(roi_img, user_id, img_id)
    
    return img

#loadinf the classsifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  #using our pc cam      
video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
     # Reading image from video stream
    _, img = video_capture.read()
    img = detect(img,faceCascade,img_id)
    # Writing  image in a new window
    cv2.imshow("face detetcion", img)
    img_id +=1 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
    