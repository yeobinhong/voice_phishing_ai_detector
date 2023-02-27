#this module includes various functions that converts speech to text
#음성파일 동시처리 버전
import speech_recognition as sr #main에서도 import해줘야 합니다
#import librosa
r = sr.Recognizer()

#speakers = [sr.AudioFile("./1.wav"), sr.AudioFile("./2.wav"), sr.AudioFile("./3.wav")]
#main에서 이런식으로 음성파일 리스트를 만들어서 입력하면 됩니다
result = []

def convert_wav_to_text(speakers):
    for i, speaker in enumerate(speakers):
        with speaker as source:
            speaker_audio = r.record(source)
        output = (f"Text from speaker {i}: {r.recognize_google(audio_data=speaker_audio, language='ko-KR')}")
        result.append(output)
    return result



