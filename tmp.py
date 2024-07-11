# repeat_speech.py

from modules.txt_to_speech.txt_to_speech import talk
from modules.speech_to_txt.speech_to_txt import listen

if __name__ == "__main__":
    print("Please say something...")
    spoken_text = listen()
    
    if spoken_text:
        talk(spoken_text)
