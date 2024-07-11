# speech.py

from gtts import gTTS
from playsound import playsound
import os

def talk(text, lang='en', filename='output.mp3'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save the speech to a file
    tts.save(filename)
    
    # Play the converted file
    playsound(filename)
    
    # Optionally remove the file after playing
    os.remove(filename)
