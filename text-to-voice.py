# import pyttsx3

# text_speech= pyttsx3.init()
# answer= input("write something: ")
# text_speech.say(answer)
# text_speech.runAndWait()

# ----------------------------------------------------

# import pyttsx3

# text_speech = pyttsx3.init()

# # Adjust the speech rate (words per minute)
# rate = text_speech.getProperty('rate')  # Get the current rate
# text_speech.setProperty('rate', rate - 50)  # Reduce the rate by 50 wpm

# answer = input("Write something: ")
# text_speech.say(answer)
# text_speech.runAndWait()

# -----------------------------------------------------------

# Python Text-to-Speech version 3
import pyttsx3

text_speech = pyttsx3.init()  #initializes a text-to-speech engine

# Adjust the speech rate (words per minute)
rate = text_speech.getProperty('rate')  # Gets the current speech rate
text_speech.setProperty('rate', rate - 50)  # Reduce the rate by 50 wpm

while True:
    answer = input("Write something (or 'exit' to quit): ")
    
    if answer.lower() == 'exit':
        break
    
    text_speech.say(answer) #  to convert the user's input text (answer) into speech.
    text_speech.runAndWait()

text_speech.stop()  # Stop the text-to-speech engine when the loop exits

# -------------------------------------------------------

