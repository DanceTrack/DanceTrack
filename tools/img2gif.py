import cv2
import glob as gb
import imageio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_file', default='results', type=str)
parser.add_argument('--gif_name', default='dancetrack.gif', type=str)
parser.add_argument('--suffix', default='png', type=str)
parser.add_argument('--show_height', default=270, type=int)
parser.add_argument('--show_width', default=480, type=int)
parser.add_argument('--show_fps', default=20, type=int)
parser.add_argument('--start_frame', default=0, type=int)
parser.add_argument('--end_frame', default=-1, type=int)

args = parser.parse_args()

saved_img_paths = gb.glob(args.img_file + "/*." + args.suffix) 
fps = args.show_fps
size = (args.show_width, args.show_height)
gif_name = args.gif_name
start_frame = args.start_frame
end_frame = args.end_frame if args.end_frame > start_frame else len(saved_img_paths)

frames = []
print('Images is loading...')
for img_path in sorted(saved_img_paths)[start_frame:end_frame]:
    img = imageio.imread(img_path)
    img = cv2.resize(img, size)
    frames.append(img)

#     new_height = size[1]
#     new_width = size[0] - size[1]
    
#     img_left = img[:, :new_width]
#     img_right = img[:, new_width:]
#     new_img_left = img_left[int(new_height/6): -int(new_height/6), int(new_width/8): -int(new_width/8)]
#     new_img_right = cv2.resize(img_right, (new_img_left.shape[0], new_img_left.shape[0]))
#     frames.append(cv2.hconcat([new_img_left, new_img_right]))

imageio.mimsave(gif_name, frames, 'GIF', duration=1/fps)
print('GIF is finished.')
