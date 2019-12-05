"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import pickle

import cv2
from gaze_tracking import GazeTracking
import csv
import xgboost
import pandas as pd
from pymouse import PyMouse
import numpy as np

# with open('dataMouse.csv', 'a') as f:
#     writer = csv.writer(f)
#     fields = ['left_x', 'left_y', 'right_x', 'right_y', 'horizontalRatio', 'verticalRatio', 'cursor_x', 'cursor_y']
#     writer.writerow(fields)


moveMouseFlag = True
debugFlag = False
cursorModelFlag = False

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
centre_model = pickle.load(open("dhruvTrainedModels/centre.pickle.dat", "rb"))
left_model = pickle.load(open("dhruvTrainedModels/left.pickle.dat", "rb"))
right_model = pickle.load(open("dhruvTrainedModels/right.pickle.dat", "rb"))
up_model = pickle.load(open("dhruvTrainedModels/up.pickle.dat", "rb"))
down_model = pickle.load(open("dhruvTrainedModels/down.pickle.dat", "rb"))
cur_x_model = pickle.load(open("cursor_x.pickle.dat","rb"))
cur_y_model = pickle.load(open("cursor_y.pickle.dat","rb"))

#mouse setup
m = PyMouse()
x_dim, y_dim = m.screen_size()
smooth_x, smooth_y= 0.5, 0.5

# img = cv2.imread('Landscape.jpg', 0) # Read in image
img1 = cv2.imread('LandscapeGrey.jpg', 0)  # Read in image
img2 = cv2.imread('LandscapeGrey2.jpg', 0)  # Read in image
dst2 = cv2.resize(img2, None, fx=2, fy=2)

img4 = cv2.imread('LandscapeGrey4.jpg', 0)  # Read in image
dst4 = cv2.resize(img4, None, fx=4, fy=4)

img8 = cv2.imread('LandscapeGrey8.jpg', 0)  # Read in image
dst8 = cv2.resize(img8, None, fx=8, fy=8)

img16 = cv2.imread('LandscapeGrey16.jpg', 0)  # Read in image

# img16 = cv2.resize(img, None, fx=0.0625, fy=0.0625)
#
# cv2.imwrite("LandscapeGrey16.jpg",img16)

height = img1.shape[0]  # Get the dimensions
width = img1.shape[1]


while True:



    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    if(moveMouseFlag):
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

    horizontalRatio = gaze.horizontal_ratio()
    verticalRatio = gaze.vertical_ratio()


    # fields = ['left_x', 'left_y', 'right_x', 'right_y', 'horizontalRatio', 'verticalRatio', 'Centre', 'Left','Right', 'Up', 'Down']

    outputStr = ""
    raw_x, raw_y = 0.5, 0.5

    if(left_pupil!=None):
        left_x = left_pupil[0]
        left_y = left_pupil[1]
        right_x = right_pupil[0]
        right_y = right_pupil[1]

        df = [left_x,left_y,right_x,right_y,horizontalRatio,verticalRatio]
        input = pd.DataFrame([df], columns=['left_x', 'left_y', 'right_x', 'right_y', 'horizontalRatio', 'verticalRatio'])

        centre_pred = centre_model.predict(input)
        left_pred = left_model.predict(input)
        right_pred = right_model.predict(input)
        up_pred = up_model.predict(input)
        down_pred = down_model.predict(input)

        cur_x_pred = cur_x_model.predict(input)
        cur_y_pred = cur_y_model.predict(input)


        if(cursorModelFlag==False):
            if(centre_pred[0]==1):
                outputStr+="Centre "
                raw_x = 0.5
                raw_y = 0.5
            if (left_pred[0] == 1):
                outputStr += "Left "
                raw_x = raw_x - 0.5
            if (right_pred[0] == 1):
                outputStr += "Right "
                raw_x = raw_x + 0.5

            if (up_pred[0] == 1):
                outputStr += "Up "
                raw_y = raw_y + 0.5
            if (down_pred[0] == 1):
                outputStr += "Down "
                raw_y = raw_y - 0.5
        else:
            raw_x = cur_x_pred
            raw_y = cur_y_pred

        # smoothing out the gaze so the mouse has smoother movement
        smooth_x += 0.5 * (raw_x - smooth_x)
        smooth_y += 0.5 * (raw_y - smooth_y)

        x = smooth_x
        y = smooth_y

        y = 1 - y  # inverting y so it shows up correctly on screen
        x *= x_dim
        y *= y_dim
        # PyMouse or MacOS bugfix - can not go to extreme corners because of hot corners?
        x = min(x_dim - 10, max(10, x))
        y = min(y_dim - 10, max(10, y))

        if(moveMouseFlag):
            m.move(x, y)

        cur_m_x,cur_m_y = m.position()

        # with open('dataMouse.csv.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     fields = [left_x,left_y, right_x, right_y, horizontalRatio, verticalRatio,cur_m_x,cur_m_y]
        #     writer.writerow(fields)



    if(debugFlag):
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Horizontal: " + str(horizontalRatio), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Vertical: " + str(verticalRatio), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Pred: C:" + str(outputStr) , (90, 270), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break


        # Define mask

    mask = np.ones(img1.shape, dtype='uint8')
    dst = cv2.resize(img16, None, fx=16, fy=16)

    mouse_x, mouse_y = m.position()
    # Draw circle at x = 100, y = 70 of radius 25 and fill this in with 0
    cv2.circle(dst, (int(mouse_x * width / x_dim), int(mouse_y * height / y_dim)), 1000, 8, -1)
    cv2.circle(dst, (int(mouse_x * width / x_dim), int(mouse_y * height / y_dim)), 500, 4, -1)
    cv2.circle(dst, (int(mouse_x * width / x_dim), int(mouse_y * height / y_dim)), 200, 2, -1)
    cv2.circle(dst, (int(mouse_x * width / x_dim), int(mouse_y * height / y_dim)), 50, 1, -1)

    dst[dst == 8] = dst8[dst == 8]
    dst[dst == 4] = dst4[dst == 4]
    dst[dst == 2] = dst2[dst == 2]
    dst[dst == 1] = img1[dst == 1]

    cv2.imshow('image', dst)
    cv2.waitKey(1)

