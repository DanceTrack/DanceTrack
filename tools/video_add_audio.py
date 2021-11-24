import sys 
import os
from moviepy.editor import *

new_video_folder = 'vis_annotations_withaudio'
if not os.path.exists(new_video_folder):
    os.makedirs(new_video_folder)

for i in list(range(16, 20+1)):
    video_path = 'video_image_1stage/trimed_{}.mp4'.format(i)
    video = VideoFileClip(video_path)
    audio_clip = video.audio
    print("video.duration", video.duration)

    old_video_path = 'vis_annotations/image_{}.avi'.format(i)
    old_video = VideoFileClip(old_video_path)
    print("old_video.duration", old_video.duration)

    # lambda t: val * t, here val * desired_time = original_time
    val = old_video.duration / video.duration
    print("val", val)
    old_video = old_video.fl_time(lambda t: val * t, apply_to=['mask', 'video', 'audio'])
    new_video = old_video.set_duration(video.duration)
    new_video = new_video.set_audio(audio_clip)

    new_video_path = new_video_folder +'/image_{}_withaudio.mp4'.format(i)
    new_video.write_videofile(new_video_path, audio_codec="aac")
