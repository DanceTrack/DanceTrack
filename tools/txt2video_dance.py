import os
import sys
import json
import cv2
import glob as gb
from colormap import colormap

import argparse
from multiprocessing import Process



def txt2img(visual_path="vis_gt", 
            show_video_name='dancetrack0001',
            img_path='dancetrack/train',
            txt_path='dancetrack/train/dancetrack0001/gt/gt.txt',
            args=None):

    # txt2img
    print(show_video_name, args.tracker, "txt2img is starting")
    if not os.path.exists(visual_path + "/" + show_video_name):
        os.makedirs(visual_path + "/" + show_video_name)

    txt_dict = dict()    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')

            img_id = linelist[0]
            obj_id = linelist[1]
            bbox = [float(linelist[2]), float(linelist[3]), 
                    float(linelist[2]) + float(linelist[4]), 
                    float(linelist[3]) + float(linelist[5]), int(obj_id)]
            if int(img_id) in txt_dict:
                txt_dict[int(img_id)].append(bbox)
            else:
                txt_dict[int(img_id)] = list()
                txt_dict[int(img_id)].append(bbox)
    
    color_list = colormap()

    ori_img_paths = gb.glob(img_path  + "/" + show_video_name + "/img1/*.jpg") 
    for ori_img_path in sorted(ori_img_paths):
        img = cv2.imread(ori_img_path)
        img_id = int(ori_img_path[-8:-4])
        if img_id in txt_dict:
            for bbox in txt_dict[img_id]:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[bbox[4]%79].tolist(), thickness=6)
                cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0])+5, int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_list[bbox[4]%79].tolist(), 4)        
        cv2.imwrite(visual_path + "/" + show_video_name + "/{:0>8d}.jpg".format(img_id), img)
    print(show_video_name, args.tracker, "txt2img is done")

    
    # img2video
    print(show_video_name, args.tracker, "img2video is starting")
    saved_img_paths = gb.glob(visual_path  + "/" + show_video_name + "/*.jpg") 
    fps = 20
    size = (960, 540)
    if args.gt:
        video_path = visual_path + "/" + show_video_name + ".avi"
    else:
        video_path = visual_path + "/" + show_video_name + "_"+ args.tracker +".avi"
        
    videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

    for saved_img_path in sorted(saved_img_paths):
        img = cv2.imread(saved_img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print(show_video_name, args.tracker, "img2video is done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='dancetrack', type=str, help='path to dancetrack dataset')
    parser.add_argument('--split', default='test', type=str, help='data split: [train, val, test]')
    parser.add_argument('--gt', default=False, action='store_true', help='False is to show prediction')
    parser.add_argument('--tracker', default='transtrack', type=str, help='name of tracker')   
    args = parser.parse_args()
        
    if args.gt:
        visual_path = "vis_gt_" + args.split 
    else:
        visual_path = "vis_" + args.split + '/' + args.tracker    
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    img_path = args.img_path + '/' + args.split
    assert os.path.exists(img_path), f'{img_path} does not exist, please prepare dataset first'
    show_video_names = sorted(os.listdir(img_path))
    
    process_list = []
    for show_video_name in show_video_names:
        if '.ipy' in show_video_name or 'seq' in show_video_name or 'DS_' in show_video_name:
            continue
        if args.gt:
            txt_path = 'dancetrack/' + args.split + '/' + show_video_name + '/gt/gt.txt'
        else:
            txt_path = args.split + '/' + args.tracker + '/' + show_video_name + '.txt'
            assert os.path.exists(txt_path), f'{txt_path} does not exist, please run prediction first'
        
        p = Process(target=txt2img, args=(visual_path, show_video_name, img_path, txt_path, args))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
