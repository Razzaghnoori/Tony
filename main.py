import core
import os
import initialize
import speech_recognition as sr
import resource
from gtts import gTTS
from playsound import playsound

while True:
    if not (os.path.exists('word2vec/400') and os.path.exists('Phrases') and os.path.exists('tfidf.pkl')):
        print("Intilizing")
        initialize.start()

    print("How can I help you?")

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source)
            question = recognizer.recognize_google(audio)
        except:
            print('.')
            continue
    
    tony = core.Tony('word2vec/400', 'tfidf.pkl', 'Phrases', tfidf_factor=0.5)
    print question
    answer = tony.answer(question)
    if len(answer) > 0:
        tts = gTTS(answer)
        tts.save('answer.mp3')
        os.system('vlc answer.mp3')
    else:
        print("I am not sure. I will inform Mr. Razzaghnoori.")

