import cv2 #pip install opencv-python

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)# if you have more than one cam change to 1

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

print("Press q If you Want to Exit")

while True:

    # Read the frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display
    cv2.imshow('Face Detection', frame)

    # Stop if q key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Destroys all the windows we created
cv2.destroyAllWindows()
