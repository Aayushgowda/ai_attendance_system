import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# -----------------------------
# Load Haar Cascade
# -----------------------------
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Haar cascade file not found!")
    exit()

# -----------------------------
# Load Dataset
# -----------------------------
dataset_path = "dataset"
faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize all images to same size
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

if len(faces) == 0:
    print("Error: No training images found!")
    exit()

# -----------------------------
# Train Face Recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# -----------------------------
# Attendance File Setup
# -----------------------------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

marked_names = []

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("AI Attendance System Started")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in detected_faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        if confidence < 80:
            name = label_map[label]

            if name not in marked_names:
                time_now = datetime.now().strftime("%H:%M:%S")
                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [name, time_now]
                df.to_csv(attendance_file, index=False)
                marked_names.append(name)

            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
