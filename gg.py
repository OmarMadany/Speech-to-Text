import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Say something...")
    audio = recognizer.listen(source)

try:
    text = recognizer.recognize_google(audio, language="en-US")
    print("📝 Recognized text:", text)
except sr.UnknownValueError:
    print("Could not understand the audio.")
except sr.RequestError:
    print("Error connecting to the recognition service.")
