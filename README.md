Offline AI Attendance System

This project is an AI-powered offline attendance system that automatically marks attendance using a web camera and face recognition.
It eliminates the need for manual attendance and works without any internet connection.

The system detects faces in real time through a webcam, recognizes registered students, and records their attendance automatically in a CSV file.

Key Features

 Real-time face detection using a webcam

 AI-based face recognition (LBPH algorithm)

 Works completely offline

 Automatic attendance marking

 Attendance stored in CSV format

 Reduces human effort and errors

 Technologies Used

Python

OpenCV (opencv-contrib-python)

NumPy

Pandas

Haar Cascade Classifier

Project Structure
AI_Attendance_System/
│
├── face_attendance.py
├── haarcascade_frontalface_default.xml
├── attendance.csv
│
└── dataset/
    └── Student_Name/
        ├── 1.jpg
        ├── 2.jpg
        └── 3.jpg
How to Run the Project
Install Dependencies
pip install opencv-contrib-python numpy pandas
Run the Program
python face_attendance.py
