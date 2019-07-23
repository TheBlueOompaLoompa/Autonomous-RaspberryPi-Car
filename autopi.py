#!/usr/bin/env python
import cv2 as cv
import datetime

# Read image from your local file system
cap = cv.VideoCapture(0)

# Load the classifier and create a cascade object for item detection
item_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

leftx = 0
rightx = 0

while True:
    ret, frame = cap.read()
    
    grayscale_image = cv.UMat(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    detected_items = item_cascade.detectMultiScale(grayscale_image)
    for (column, row, width, height) in detected_items:
        cv.rectangle(
            frame,
            (column, row),
            
            (column + width, row + height),
            (255, 0, 255),
            2
        )
        
        print(str(datetime.datetime.now().strftime('%I:%M:%S %p')) + "       x:" + str(column + (width/2)), "   y:" + str(row + (height/2)))
        if(column + (width/2) <= 320):
            leftx = column + (width/2)
            cv.putText(frame, "Left", (column, row - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            rightx = column + (width/2)
            cv.putText(frame, "Right", (column, row - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(str(leftx) + "   " + str(rightx))
    cv.imshow('Tracking',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
