# main.py

from modules.facial_recognition.facial_recognition import load_known_faces, recognize_faces, save_new_face, FacialRecognitionError
import cv2
import threading

def handle_recognized_faces(recognized_faces):
    """
    Function to handle recognized faces from facial recognition loop.
    Prints greeting or prompts user to save new faces.
    """
    for name, _ in recognized_faces:
        if name == "Unknown":
            print("Unknown face detected. Please provide a name.")
            new_name = save_new_face()  # Attempt to save new face
            print(f"Thank you, it's very nice to meet you {new_name}. How can I assist you today?")
        else:
            print(f"Hello {name}, good to see you!")
            print(f"Hi {name}, How can I help you today?")

def facial_recognition_loop():
    """
    Facial recognition loop running in a separate thread.
    """
    try:
        load_known_faces()

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Unable to access the camera. Please check camera permissions.")
            return
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture image")
                break

            recognized_faces = recognize_faces(frame)
            handle_recognized_faces(recognized_faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    except FacialRecognitionError as e:
        print(f"Facial Recognition Error: {str(e)}")

def main():
    """
    Main function to start the facial recognition loop and interact with the user.
    """
    facial_thread = threading.Thread(target=facial_recognition_loop)
    facial_thread.start()

    # Main thread continues here
    print("Main app is ready to interact with the user.")

    # Main thread can perform other tasks or wait for user input
    while True:
        user_input = input("Enter a command or prompt: ")
        # Handle user input or send to other modules like LLM

        if user_input.lower() == 'quit':
            break

    facial_thread.join()  # Wait for facial recognition thread to complete

if __name__ == "__main__":
    main()
