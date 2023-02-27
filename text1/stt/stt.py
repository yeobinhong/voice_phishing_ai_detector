#this module includes various functions that converts speech to text
import speech_recognition as sr
import librosa
r = sr.Recognizer()


def convert_wav_to_text(audio_file_name):
    audiofile = f'./{audio_file_name}.wav' #상대위치
#    audiofile = f'C:\Users\SAMSUNG\voicephishing/{audio_num}.wav' #절대위치 사용할 경우 이렇게 변경
    korean_audio = sr.AudioFile(audiofile)

    with korean_audio as source:
        audio = r.record(source)
    output = r.recognize_google(audio_data = audio, language = 'ko-KR')
    return output



