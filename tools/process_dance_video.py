import sys 
import os

trim_command = "ffmpeg -ss {} -t {} -i  first20videos/{}.mp4 -vcodec copy -acodec copy dance_video_trimed/trimed_{}.mp4"
extract_frame_command = "ffmpeg -i dance_video_trimed/trimed_{}.mp4 -r 20 -q:v 2 -f image2 dance_video_trimed/image_{}/%08d.jpg"

args = [
            ["00:00:10", "00:01:00", 1],
            ["00:00:12", "00:01:00", 2],
            ["00:00:05", "00:01:00", 3],
            ["00:00:20", "00:01:00", 4],
            ["00:00:06", "00:01:00", 5],
            ["00:00:35", "00:01:00", 6],
            ["00:00:30", "00:01:00", 7],
            ["00:00:08", "00:01:02", 8],
            ["00:00:05", "00:01:00", 9],
            ["00:00:05", "00:01:00", 10],
            ["00:00:10", "00:01:00", 11],
            ["00:00:21", "00:01:00", 12],
            ["00:00:33", "00:01:00", 13],
            ["00:00:28", "00:01:00", 14],
            ["00:00:10", "00:01:00", 15],
            ["00:00:34", "00:01:48", 16],
            ["00:00:50", "00:02:40", 17],
            ["00:00:29", "00:00:25", 18],
            ["00:01:00", "00:02:00", 19],
            ["00:01:02", "00:00:29", 20]
]

os.makedirs("dance_video_trimed", exist_ok=True)
for arg in args:
    print("================ {} =================".format(int(arg[2])))
    cmd = trim_command.format(arg[0], arg[1], arg[2], arg[2])
    os.system(cmd)

    cmd = extract_frame_command.format(arg[2], arg[2])
    os.makedirs("dance_video_trimed/image_{}".format(arg[2]), exist_ok=True)
    os.system(cmd)
