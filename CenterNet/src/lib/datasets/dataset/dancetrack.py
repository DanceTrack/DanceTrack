from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
from collections import defaultdict
import numpy as np
import torch
import json
import cv2
import os
import math

import torch.utils.data as data


class DanceTrack(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]
    
    def __init__(self, opt, split):
        super(DanceTrack, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'dancetrack')
        self.img_dir = os.path.join(self.data_dir, split)
        self.annot_path = os.path.join(self.data_dir, 'annotations', '{}.json').format(split)
        self.max_objs = 128
        self.class_name = [
            '__background__', 'dancer']
        self.cat_ids = {1: 0}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt
        
        print('==> initializing dancetrack {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))


    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        pass

    def save_results(self, results, save_dir):
        video_to_images = defaultdict(list)
        video_names = defaultdict()       
        for _, info in self.coco.imgs.items():
            video_to_images[info["video_id"]].append(
                {"image_id": info["id"], "frame_id": info["frame_id"]
                })
            
            video_name = info["file_name"].split("/")[0]
            if video_name not in video_names:
                video_names[info["video_id"]] = video_name
        
        assert len(video_to_images) == len(video_names)
        
        results_path = os.path.join(save_dir, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        print("Saving results to {}.".format(results_path))
        
        for video_id in video_to_images.keys():
            video_infos = video_to_images[video_id]
            video_name = video_names[video_id]
            file_path = os.path.join(save_dir, "results/{}.txt".format(video_name))
            f = open(file_path, "w")
            
            tracks = defaultdict(list)
            for video_info in video_infos:
                image_id, frame_id = video_info["image_id"], video_info["frame_id"]
                if image_id in results:
                    result = results[image_id]
                    for item in result:
                        if not ("tracking_id" in item):
                            raise NotImplementedError
                        tracking_id = item["tracking_id"]
                        bbox = item["bbox"]
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3], item['score'], item['active']]
                        tracks[tracking_id].append([frame_id] + bbox)

            rename_track_id = 0
            for track_id in sorted(tracks):
                rename_track_id += 1
                for t in tracks[track_id]:
                    if t[6] > 0:
                        f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n".format(
                            t[0], rename_track_id, t[1], t[2], t[3] - t[1], t[4] - t[2]))
            f.close()                

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        # Please run evaluation code in DanceTrack github repo 
