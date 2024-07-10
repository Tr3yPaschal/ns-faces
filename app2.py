import face_recognition
import os
import numpy as np
import cv2

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_faces = []
known_names = []
greeted_faces = []

def save_new_face(frame, face_encoding):
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    new_name = input("Enter the name for the new face: ")
    known_names.append(new_name)
    known_faces.append(face_encoding)
    greeted_faces.append(new_name)
    img_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"Saved new face as {new_name}")

def load_known_faces():
    for filename in os.listdir(KNOWN_FACES_DIR):
        img_path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        
        if len(face_encodings) == 0:
            print(f"No face found in image: {filename}")
            continue
        
        encoding = face_encodings[0]  # Assuming only one face per image
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])
        print(f"Loaded face from {filename}")

def recognize_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    recognized_faces = []

    for face_encoding, loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if known_faces:
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        recognized_faces.append((name, loc))

    return recognized_faces
