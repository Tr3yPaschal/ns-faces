# speech_to_text.py

import os
import speech_recognition as sr
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play as play_mp3

def play_beep_mp3(filename):
    # Determine the absolute path to beep.mp3
    beep_path = os.path.join(os.path.dirname(__file__), filename)
    
    # Load and play beep sound in mp3 format
    beep_sound = AudioSegment.from_mp3(beep_path)
    play_mp3(beep_sound)

def listen(beep_sound='beep.mp3', timeout=5, phrase_time_limit=10):
    # Play beep when starting to listen
    play_beep_mp3(beep_sound)
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        try:
            # Listen with a timeout and phrase time limit
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            #print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return None
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return None
        finally:
            # Play beep sound when done listening
            play_beep_mp3(beep_sound)
