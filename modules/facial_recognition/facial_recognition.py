import os
import cv2
import numpy as np
import face_recognition

KNOWN_FACES_DIR = "modules/facial_recognition/known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_faces = []
known_names = []
greeted_faces = []

class FacialRecognitionError(Exception):
    pass

def load_known_faces():
    global known_faces, known_names
    from .facial_recognition import recognize_faces  # Import inside function to break circular dependency
    try:
        for filename in os.listdir(KNOWN_FACES_DIR):
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
            if img is None:
                raise FacialRecognitionError(f"Error loading image: {filename}")
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb_img)
            if len(encoding) == 0:
                raise FacialRecognitionError(f"No face found in image: {filename}")
            
            known_faces.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"Loaded face from {filename}")
    except Exception as e:
        raise FacialRecognitionError(f"Error loading known faces: {str(e)}")

def save_new_face(frame):
    global known_faces, known_names, greeted_faces
    from .facial_recognition import recognize_faces  # Import inside function to break circular dependency
    try:
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
        new_name = input("Enter the name for the new face: ")
        known_names.append(new_name)
        img_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved new face as {new_name}")
        # Reload known faces after saving
        load_known_faces()
        return new_name
    except Exception as e:
        raise FacialRecognitionError(f"Error saving new face: {str(e)}")

def recognize_faces(frame):
    global known_faces, known_names
    try:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        recognized_faces = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
    
            if known_faces:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
    
            recognized_faces.append((name, face_locations[0]))
    
        return recognized_faces
    except Exception as e:
        raise FacialRecognitionError(f"Error recognizing faces: {str(e)}")
