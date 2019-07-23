#!/usr/bin/env python
import cv2 as cv
import datetime

# Read image from your local file system
cap = cv.VideoCapture(0)

# Load the classifier and create a cascade object for item detection
item_cascade = cv.CascadeClassifier('cascade.xml')

# Store the x position of the curb on the left and right side of the road
leftx = 0
rightx = 0

while True:
    # Store the video capture in a varible called frame
    ret, frame = cap.read()

    # convert frame to grayscale and use UMat to force the pi to use its gpu
    grayscale_image = cv.UMat(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    # Actually detect the curb defined in the cascade classifier
    detected_items = item_cascade.detectMultiScale(grayscale_image)

    # Generate a purple rectangle around the curb
    for (column, row, width, height) in detected_items:
        cv.rectangle(
            frame,
            (column, row),
            
            (column + width, row + height),
            (255, 0, 255),
            2
        )

        # Print the detected curbs location and shows timestamp
        print(str(datetime.datetime.now().strftime('%I:%M:%S %p')) + "       x:" + str(column + (width/2)), "   y:" + str(row + (height/2)))
        if(column + (width/2) <= 320): # Checks if the curb is on the left and displays the corrisponding text
            leftx = column + (width/2)
            cv.putText(frame, "Left", (column, row - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: # If its not on the left it must be on the right and displays the corrisponding text
            rightx = column + (width/2)
            cv.putText(frame, "Right", (column, row - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(str(leftx) + "   " + str(rightx))

    # Shows the final image
    cv.imshow('Tracking',frame)

    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
