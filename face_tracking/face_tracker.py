import cv2 
import pathlib

# Load Video
cap = cv2.VideoCapture(0)

# Load cascade from opencv library file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Read the frame
    _, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 4,
        minSize = (10,10),
        flags = cv2.CASCADE_SCALE_IMAGE)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display
    cv2.imshow('face tracker', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()
