import os
import cv2
import numpy as np
import face_recognition
import queue
import time

KNOWN_FACES_DIR = "modules/facial_recognition/known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_faces = []
known_names = []
greeted_faces = []
message_queue = queue.Queue()

class FacialRecognitionError(Exception):
    pass

def load_known_faces():
    global known_faces, known_names
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

def save_new_face(frame, face_encoding, new_name):
    try:
        known_names.append(new_name)
        known_faces.append(face_encoding)
        greeted_faces.append(new_name)
        img_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved new face as {new_name}")
        # Reload known faces after saving the new face
        reload_known_faces()

    except Exception as e:
        raise FacialRecognitionError(f"Error saving new face: {str(e)}")

def reload_known_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []
    greeted_faces.clear()
    load_known_faces()


def recognize_faces(frame):
    try:
        global known_faces, known_names
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        recognized_faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
    
            if known_faces:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
    
            recognized_faces.append((name, face_location))
    
        return recognized_faces
    except Exception as e:
        raise FacialRecognitionError(f"Error recognizing faces: {str(e)}")

def facial_recognition_loop(message_queue, response_queue):
    load_known_faces()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        message_queue.put("Error: Unable to access the camera. Please check camera permissions.")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            message_queue.put("Failed to capture image")
            break

        recognized_faces = recognize_faces(frame)
        if recognized_faces:
            for name, face_location in recognized_faces:
                if name == "Unknown":
                    if not message_queue.full():
                        message_queue.put("REQUEST_NAME")
                        print(f"Message sent: REQUEST_NAME")
                        while True:
                            msg = response_queue.get()
                            if msg.startswith("NAME:"):
                                new_name = msg.split(":")[1]
                                save_new_face(frame, face_location, new_name)
                                break
                else:
                    if name not in greeted_faces:
                        greeted_faces.append(name)
                        if not message_queue.full():
                            message_queue.put(f"GREET {name}")  # Notify to greet the recognized face
                        time.sleep(1)  # Wait briefly to avoid rapid greeting

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

