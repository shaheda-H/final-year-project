import cv2
import numpy as np
import face_recognition
import os
import mysql.connector
from datetime import datetime
import pygame

# Initialize the pygame mixer
pygame.mixer.init()

# Load the sound file (replace 'alert_sound.wav' with your sound file)
alert_sound = pygame.mixer.Sound('thank-you.mp3')

# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance in MySQL and CSV
def markAttendance(name):
    # Establish a connection to the MySQL database
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='attdb'
    )

    cursor = connection.cursor()

    # Check if the name and date are not already in the MySQL database
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString = now.strftime('%Y-%m-%d')

    sql_check = "SELECT * FROM attendance WHERE Name = %s AND Date = %s"
    values_check = (name, dateString)
    cursor.execute(sql_check, values_check)
    result = cursor.fetchone()

    if result is None:
        # Insert data into MySQL database
        sql_insert = "INSERT INTO attendance (Name, Time, Date) VALUES (%s, %s, %s)"
        values_insert = (name, dtString, dateString)

        cursor.execute(sql_insert, values_insert)
        connection.commit()

        # Append data to the CSV file
        with open('Attendance.csv', 'a') as f:
            f.writelines(f'\n{name},{dtString},{dateString}')

        # Play the alert sound
        alert_sound.play()

        print(f"{name} marked attendance successfully.")
    else:
        print(f"{name} already marked attendance today.")

    # Close the MySQL connection
    cursor.close()
    connection.close()

# Load training images and their respective class names
path = 'Training_images'
images = [cv2.imread(f'{path}/{cl}') for cl in os.listdir(path)]
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Capture video from the webcam
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Initialize name as None
    name = None

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display "Unknown" message if no face is recognized
    if name is None:
        cv2.putText(img, "Unknown", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    
    # Call markAttendance function only if a face is recognized
    if name:
        markAttendance(name)
