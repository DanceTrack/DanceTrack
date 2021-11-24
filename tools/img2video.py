import cv2
import glob as gb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_file', default='results', type=str)
parser.add_argument('--video_name', default='dancetrack.avi', type=str)
parser.add_argument('--suffix', default='png', type=str)
parser.add_argument('--show_height', default=540, type=int)
parser.add_argument('--show_width', default=960, type=int)
parser.add_argument('--show_fps', default=20, type=int)

args = parser.parse_args()

saved_img_paths = gb.glob(args.img_file + "/*." + args.suffix) 

fps = args.show_fps
size = (args.show_width, args.show_height)

video_path = args.video_name


videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

print('Images is loading...')
for saved_img_path in sorted(saved_img_paths):
    img = cv2.imread(saved_img_path)
    img = cv2.resize(img, size)
    videowriter.write(img)

videowriter.release()
print('Video is finished.')
