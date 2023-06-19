import cv2
import numpy as np
from matplotlib import pyplot as plt
# Define our imshow function
def imshow(title = "Image", image = None, size = 8):
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = h/w
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 represents the default camera

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Unable to open the camera")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame")
        break
    # Display the frame
    cv2.imshow("Camera", frame)

    # Wait for the Enter key to be pressed
    if cv2.waitKey(1) == 13:  # ASCII code for Enter key is 13
        cv2.imwrite("input.jpg", frame)
        print("Photo captured successfully!")
        break

camera.release()
cv2.destroyAllWindows()

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

cv2.imwrite("output.jpg", img)
imshow('Eye & Face Detection',img)

