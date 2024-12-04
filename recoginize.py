import cv2
import numpy as np
import os

# Configurations
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  # Ensure this folder exists and contains subfolders for each person
print('Recognizing Face. Please be in sufficient light...')

# Initialize variables
(images, labels, names, id) = ([], [], {}, 0)

# Load images and labels from the dataset
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))  # Load grayscale images
            labels.append(int(label))
        id += 1

# Convert lists to numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Create the LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Train the recognizer
model.train(images, labels)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start video capture
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve image contrast
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))

        # Predict the face
        prediction = model.predict(face_resize)

        if prediction[1] < 500:
            name = f'{names[prediction[0]]} - {prediction[1]:.0f}'
            cv2.putText(im, name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            # If face is not recognized, display "Invalid"
            cv2.putText(im, 'Invalid', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    # Display the image with recognized faces or invalid
    cv2.imshow('Face Recognition', im)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
