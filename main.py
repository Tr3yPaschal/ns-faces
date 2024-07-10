import threading
from queue import Queue
from modules.facial_recognition.facial_recognition import facial_recognition_loop, FacialRecognitionError

def main():
    message_queue = Queue()
    response_queue = Queue()

    facial_recognition_thread = threading.Thread(target=facial_recognition_loop, args=(message_queue, response_queue))
    facial_recognition_thread.start()

    greeted_faces = set()

    while True:
        try:
            if not message_queue.empty():
                message = message_queue.get()
                
                if message.startswith("Unknown face detected"):
                    print(message)
                    name = input("Enter the name for the new face: ")
                    response_queue.put(name)
                
                elif message.startswith("Hello"):
                    name = message.split(" ")[1].strip(",")
                    if name not in greeted_faces:
                        greeted_faces.add(name)
                        print(message)
                        print(f"Hi {name}, how can I help you today?")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
