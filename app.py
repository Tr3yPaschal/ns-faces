import face_recognition  # Importing the face_recognition library for face detection and recognition
import os  # Importing the os module for operating system operations
import numpy as np  # Importing numpy for numerical operations
import cv2  # Importing OpenCV library for computer vision tasks

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"  # Setting the directory name where known faces will be stored
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)  # Creating the directory if it doesn't exist

# Initialize video capture from the default camera (0)
video_capture = cv2.VideoCapture(0)  # Opening a video capture object for the default camera (0)

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Unable to access the camera. Please check camera permissions.")
    exit()  # Exiting the program if the camera is not accessible

# Empty arrays to store known faces and names
known_faces = []  # Initializing an empty list to store face encodings of known faces
known_names = []  # Initializing an empty list to store names corresponding to known faces
greeted_faces = []  # Initializing an empty list to track faces that have been greeted

# Function to save a new face with its name
def save_new_face(frame, face_encoding):
    cv2.imshow('Video', frame)  # Displaying the video frame
    cv2.waitKey(1)  # Waiting briefly to display the frame
    new_name = input("Enter the name for the new face: ")  # Prompting user to input the name for the new face
    known_names.append(new_name)  # Adding the new name to the list of known names
    known_faces.append(face_encoding)  # Adding the face encoding to the list of known faces
    greeted_faces.append(new_name)  # Marking the new face as greeted
    img_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")  # Constructing the image path with the new name
    cv2.imwrite(img_path, frame)  # Saving the frame as an image file with the new name
    print(f"Saved new face as {new_name}")  # Printing a message confirming the saved face

# Main loop for capturing video frames and processing faces
while True:
    ret, frame = video_capture.read()  # Capturing a frame from the video stream
    if not ret:
        print("Failed to capture image")  # Printing an error message if capturing fails
        break  # Breaking out of the loop if capturing fails

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(frame)  # Locating faces in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)  # Encoding faces found in the frame


    print(len(known_faces))
    
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(known_faces, face_encoding)  # Comparing detected faces with known faces

        print(len(matches))

        name = "Unknown"  # Default name for unrecognized faces

        if known_faces:  # Checking if there are known faces stored
            face_distances = face_recognition.face_distance(known_faces, face_encoding)  # Calculating face distances
            best_match_index = np.argmin(face_distances)  # Finding the index of the closest match
            if matches[best_match_index]:  # Checking if the closest match is a match
                name = known_names[best_match_index]  # Assigning the name of the closest match to 'name'

        if name == "Unknown":  # If the face is not recognized as known
            save_new_face(frame, face_encoding)  # Calling the function to save a new face
        elif name not in greeted_faces:  # If the recognized face has not been greeted before
            print(f"Hello {name}, good to see you!")  # Greeting the recognized face by name
            greeted_faces.append(name)  # Marking the recognized face as greeted

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Checking for 'q' key press to exit the loop
        break  # Exiting the main loop

video_capture.release()  # Releasing the video capture object
cv2.destroyAllWindows()  # Closing all OpenCV windows
