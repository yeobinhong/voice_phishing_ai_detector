#this module includes functions that convert video to audio

import moviepy.editor as mp
#video_file_name = input() #이거를 main에 작성하면 됩니다

#mp4 to wav
def mp4_to_wav(video_file_name):
    video_file = f'{video_file_name}.mp4'
    converted_audio_name = f'{video_file_name}.wav'
    clip = mp.VideoFileClip(video_file).subclip(0, )  # 0초부터 끝까지로 설정했는데 원하는 시간대(초)로 입력하시면 됩니다
    return clip.audio.write_audiofile(converted_audio_name)

#avi to wav
def avi_to_wav(video_file_name):
    video_file = f'{video_file_name}.avi'
    converted_audio_name = f'{video_file_name}.wav'
    clip = mp.VideoFileClip(video_file).subclip(0, )  # 0초부터 끝까지로 설정했는데 원하는 시간대(초)로 입력하시면 됩니다
    return clip.audio.write_audiofile(converted_audio_name)

#mp4 to mp3
def mp4_to_mp3(video_file_name):
    video_file = f'{video_file_name}.mp4'
    converted_audio_name = f'{video_file_name}.mp3'
    clip = mp.VideoFileClip(video_file).subclip(0, )  # 0초부터 끝까지로 설정했는데 원하는 시간대(초)로 입력하시면 됩니다
    return clip.audio.write_audiofile(converted_audio_name)

#avi to mp3
def avi_to_wav(video_file_name):
    video_file = f'{video_file_name}.avi'
    converted_audio_name = f'{video_file_name}.mp3'
    clip = mp.VideoFileClip(video_file).subclip(0, )  # 0초부터 끝까지로 설정했는데 원하는 시간대(초)로 입력하시면 됩니다
    return clip.audio.write_audiofile(converted_audio_name)