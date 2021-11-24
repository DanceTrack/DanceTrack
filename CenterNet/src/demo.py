from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track', 'display']


# video(1080, 1920) cannot well fit to CenterNet(512, 512) in visualization, especially for segmentation
# we need to resize the original size to (540, 960), instead of pre-process to (544, 960)
def custom_resize(image, size=540, max_size=960):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    new_height, new_width = get_size_with_aspect_ratio(image.shape[:2], size, max_size)    
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            #         cv2.imshow('input', img)
#             ret = detector.run(img)
            ret = detector.run(custom_resize(img))
            time_str = ''
            for stat in time_stats:
                if stat in ret:
                    time_str = time_str + '{} {:07.3f}ms |'.format(stat, ret[stat] * 1000)
            print(time_str)
    #         if cv2.waitKey(1) == 27:
    #             return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for (image_name) in image_names:
            img = cv2.imread(image_name)
#             ret = detector.run(img)
            ret = detector.run(custom_resize(img))
            time_str = ''
            for stat in time_stats:
                if stat in ret:
                    time_str = time_str + '{} {:07.3f}ms |'.format(stat, ret[stat] * 1000)
            print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
