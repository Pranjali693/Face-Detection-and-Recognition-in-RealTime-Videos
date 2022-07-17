import cv2

import keras
import numpy as np
from gaze_tracking import GazeTracking
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.4  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
gaze=GazeTracking()
cap = cv2.VideoCapture(0 )
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model = keras.models.load_model(r"F:\Nu\semester 6\R & D\my_facemodel1_show" ,compile=True)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Abhishek'
    elif classNo == 1:
        return 'Anurag'
    elif classNo == 2:
        return 'Aryan raj'
    elif classNo == 3:
        return 'Divyanshi'
    elif classNo == 4:
        return 'Gayatri'
    elif classNo == 5:
        return 'Harsh'
    elif classNo == 6:
        return 'Karthikeya'
    elif classNo ==7:
        return 'Karunesh'
    elif classNo == 8:
        return 'Mehak'
    elif classNo == 9:
        return 'Palak'
    elif classNo == 10:
        return 'Pranjali Dumbre'
    elif classNo == 11:
        return 'Shalom'


while True:

    _, frame = cap.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    # READ IMAGE
    success, imgOrignal =cap.read()

    # PROCESS IMAGE
    # img = np.asarray(imgOrignal)
    
    img = cv2.resize(imgOrignal,(32,32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex =  np.argmax(model.predict(img), axis=-1)                       # model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:

        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        print(getCalssName(classIndex) , classIndex)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()