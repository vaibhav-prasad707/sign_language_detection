import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp

cap = cv2.VideoCapture(0) #Device Camera 0, Web Camera 1
detector = HandDetector(maxHands=1)
offset = 20 
image_size = 300
counter = 0

#To collect the images for the ML
#Folder Path
folder = '/Users/vidushini/Desktop/SignLanguage/sign_language_detection/Data/Hello'


#Data Collection
while(True):
    success , img = cap.read() #Capturing the images
    hands , img = detector.findHands(img) #Finding the hands in the image
    if hands:
        hand = hands[0]
        #x axis, y axis, width, height
        x,y,w,h = hand['bbox']
        #To get the white background of the hand images to make the ML easier
        imgWhite = np.ones((image_size,image_size,3),np.uint8)*255
        imgCrop = img[y - offset : y + h + offset,x - offset : x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = image_size/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,image_size))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((image_size - wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
        else:
            k = image_size/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(hCal,image_size))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((image_size - hCal)/2)
            imgWhite[hGap : hCal + hGap,:] = imgResize

        cv2.imshow('ImageCrop',imgCrop)
        cv2.imshow('ImageWhite',imgWhite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)