import threading
from queue import Queue
from modules.facial_recognition.facial_recognition import facial_recognition_loop  # Import facial recognition loop function
from modules.llm.llm import chat  # Import LLM chat function
from modules.txt_to_speech.txt_to_speech import talk  # Import text-to-speech function
from modules.speech_to_txt.speech_to_txt import listen  # Import speech-to-text function

def main():
    # Initialize communication queues between threads
    message_queue = Queue()    # Queue for messages from facial recognition module
    response_queue = Queue()   # Queue for responses to other modules

    # Start facial recognition module in a separate thread
    facial_recognition_thread = threading.Thread(
        target=facial_recognition_loop,  # Function to run in the thread
        args=(message_queue, response_queue)  # Arguments to pass to the function
    )
    facial_recognition_thread.start()  # Start the facial recognition thread

    # Set to track recognized faces and current user
    greeted_faces = set()  # Set to store names of recognized faces
    current_user = None    # Variable to track current user being interacted with

    while True:  # Main application loop
        try:
            #print("Before while loop, message queue empty:", message_queue.empty())
            # Process messages from facial recognition module
            while not message_queue.empty():  # Check if there are messages in queue

                
                action = message_queue.get()  # Get next message from the queue
                
                
                if action.startswith("REQUEST_NAME"):  # If new face detected
            
                    talk("Hello, It's nice to meet you. I'm Cassandra, What is your name ?")  # Prompt for new face's name
                    name = listen()  # Prompt for new face's name
                    response_queue.put(f"NAME:{name}")  # Send name back to facial recognition module

                elif action.startswith("GREET"):  # If recognized face greeted
                    
                    parts = action.split(" ")
                    name = parts[1].strip(" ")  # Extract name from greeting
                    if name not in greeted_faces:  # If face not greeted before
                        greeted_faces.add(name)  # Add name to greeted faces set
                        current_user = name  # Set current user to this name
                        talk(f"Hello {name}, how can I help you today?")  # Greet the user
                        prompt = listen()  # Prompt user for input
                        response_queue.put(f"PROMPT:{prompt}")  # Send user's prompt to LLM module
                    else:
                        print("Unexpected format for action:", action)  # Handle unexpected format


            # Handle LLM responses and user prompts
            if current_user and not response_queue.empty():  # If there's a current user and responses in queue
                response = response_queue.get()  # Get next response from the queue

                if response.startswith("PROMPT:"):  # If response is a user prompt
                    prompt = response.split(":", 1)[1].strip()  # Extract the prompt text
                    llm_response = chat(prompt)  # Generate response from LLM
                    talk(llm_response)  # Print LLM's response
                    prompt = listen()  # Prompt user for next input
                    response_queue.put(f"PROMPT:{prompt}")  # Send new prompt to LLM module

        except KeyboardInterrupt:  # Handle keyboard interrupt (Ctrl+C)
            break  # Exit the main loop if interrupted

if __name__ == "__main__":
    main()  # Execute the main function if script is run directly
