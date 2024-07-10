# main.py

import threading                # Import threading module to allow concurrent execution
from queue import Queue         # Import Queue class for thread-safe communication between threads
from modules.facial_recognition.facial_recognition import facial_recognition_loop  # Import facial recognition loop function
from modules.llm.llm import simulate_llm  # Import the simulate_llm function from the LLM module

def main():
    """
    Main function to initialize and manage the application's threads and communication.
    """

    # Create queues for inter-thread communication
    message_queue = Queue()    # Queue for messages from facial recognition module
    response_queue = Queue()   # Queue for responses from main thread to other modules

    # Start the facial recognition module in a separate thread
    facial_recognition_thread = threading.Thread(
        target=facial_recognition_loop, 
        args=(message_queue, response_queue)
    )
    facial_recognition_thread.start()  # Start the facial recognition thread

    # Set to track faces that have been greeted and variable for the current user
    greeted_faces = set()  # Set to keep track of recognized and greeted faces
    current_user = None    # Variable to keep track of the current user being interacted with

    while True:  # Main application loop
        try:
            # Process messages from the facial recognition module
            while not message_queue.empty():  # Check if there are any messages in the queue
                action = message_queue.get()  # Retrieve the next message from the queue

                if action.startswith("Unknown face detected"):  # If the message indicates an unknown face
                    print(action)  # Print the message to the console
                    name = input("Enter the name for the new face: ")  # Prompt the user for a name
                    response_queue.put(f"NAME:{name}")  # Send the name back to the facial recognition module

                elif action.startswith("Hello"):  # If the message is a greeting for a recognized face
                    name = action.split(" ")[1].strip(",")  # Extract the name from the greeting message
                    if name not in greeted_faces:  # If this face has not been greeted before
                        greeted_faces.add(name)  # Add the name to the set of greeted faces
                        current_user = name  # Set this name as the current user
                        print(action)  # Print the greeting message
                        prompt = input(f"Hi {name}, how can I help you today? ")  # Prompt the user for input
                        response_queue.put(f"PROMPT:{prompt}")  # Send the user's prompt to the LLM module

            # Continuously wait for user prompts and process LLM responses
            if current_user and not response_queue.empty():  # Check if there is a current user and if there are responses in the queue
                response = response_queue.get()  # Retrieve the next response from the queue

                if response.startswith("PROMPT:"):  # If the response is a prompt from the user
                    prompt = response.split(":", 1)[1].strip()  # Extract the prompt from the response
                    print(f"User Input: {prompt}")  # Print the user's input
                    llm_response = simulate_llm(prompt)  # Generate a response from the LLM
                    print(f"LLM: {llm_response}")  # Print the LLM's response
                    # After responding, prompt the user for the next input
                    prompt = input(f"Hi {current_user}, how can I help you today? ")  # Prompt the user for input
                    response_queue.put(f"PROMPT:{prompt}")  # Send the user's prompt to the LLM module

        except KeyboardInterrupt:  # Handle keyboard interrupt (Ctrl+C)
            break  # Exit the main loop

if __name__ == "__main__":
    main()  # Execute the main function if this script is run directly
