# -*- coding: utf-8 -*-
"""reading from microphone.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11gLy7AtUqSH0TQMbb-SWDHClVsq3mHKw
"""

pip3 install pyaudio #windows

sudo apt-get install python-pyaudio python3-pyaudio #linux
pip3 install pyaudio #linux

brew install portaudio #mac os
pip3 install pyaudio #mac os

with sr.Microphone() as source:
    # read the audio data from the default microphone
    audio_data = r.record(source, duration=5)
    print("Recognizing...")
    # convert speech to text
    text = r.recognize_google(audio_data)
    print(text)

text = r.recognize_google(audio_data, language="ko-KR")