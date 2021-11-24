from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob as gb
import numpy as np
import cv2
import argparse
from deepsort_tracker.reid_model import Extractor
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

class AppearanceFeature(object):
    def __init__(self, model_path, use_cuda=True):

        self.extractor = Extractor(model_path, use_cuda=use_cuda)
    
    def update(self, output_results, img_file_name):
        ori_img = cv2.imread(img_file_name)
        self.height, self.width = ori_img.shape[:2]
        
        bboxes = output_results[:, :4]  # x1y1x2y2
        bbox_xyxy = bboxes
        bbox_tlwh = self._xyxy_to_tlwh_array(bbox_xyxy)

        # generate detections
        features = self._get_features(bbox_tlwh, ori_img)    
        
        return features
    
    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh    
    
    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2
    
    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
            features = np.asarray(features) / np.linalg.norm(features, axis=1, keepdims=True)
        else:
            features = np.array([])
        return features    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=25, type=int)
args = parser.parse_args()

# dataset = 'mot/val'
dataset = 'dancetrack/val'

val_pred = 'oracle_analysis/val_appearance'
if not os.path.exists(val_pred):
    os.makedirs(val_pred)
    
video_cosine_dist_ret = []
val_seqs = sorted(os.listdir(dataset))[args.start:args.end+1]
for video_name in val_seqs:
    print(video_name)
    det_results = {}
    with open(os.path.join(dataset, video_name, 'gt/gt.txt'), 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            img_id = linelist[0]
            bbox = [float(linelist[2]), 
                    float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]),
                    float(linelist[3]) + float(linelist[5]), 
                    float(linelist[1])]
            if int(linelist[7]) == 1:
                if int(img_id) in det_results:
                    det_results[int(img_id)].append(bbox)
                else:
                    det_results[int(img_id)] = list()
                    det_results[int(img_id)].append(bbox)
    f.close()
    
    cosine_dist_ret = []
    star_idx = len(gb.glob(os.path.join(dataset, video_name, 'img1') + "/*.jpg")) // 2 + 1
    tracker = AppearanceFeature(model_path='ckpt.t7')
    for frame_id in sorted(det_results.keys()):
        dets = det_results[frame_id]
        dets = np.array(dets)
#         image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id + star_idx))
        image_path = os.path.join(dataset, video_name, 'img1', '{:0>8d}.jpg'.format(frame_id))
        
        appearance_feat = tracker.update(dets, image_path)
        
        cosine_dist_mat = 1. - np.dot(appearance_feat, appearance_feat.T)
        cosine_dist  = cosine_dist_mat.sum() / len(appearance_feat) / len(appearance_feat)
        cosine_dist_ret.append(cosine_dist)
    
    video_cosine_dist_ret.append(sum(cosine_dist_ret) / len(cosine_dist_ret))
    print(video_cosine_dist_ret)


    
import matplotlib.pyplot as plt
mot  = [0.289, 0.327, 0.240, 0.224, 0.301, 0.262, 0.269]
dancetrack = [0.173, 0.150, 0.181, 0.181, 0.216, 0.176, 0.186, 0.215, 0.227, 0.181, 
              0.214, 0.172, 0.206, 0.204, 0.200, 0.236, 0.176, 0.172, 0.221, 0.170,
              0.212, 0.233, 0.207, 0.229, 0.140]

mot_x = range(len(mot))
dancetrack_x = range(len(mot), len(mot) + len(dancetrack))

fig, ax = plt.subplots(figsize=(15, 5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.bar(x=mot_x, height=mot, alpha=0.3, color='blue', label='MOT17')
plt.bar(x=dancetrack_x, height=dancetrack, alpha=0.3, color='red', label='DanceTrack')

plt.legend(fontsize=16)
plt.xticks([])
plt.ylim((0.10, 0.35))
plt.title("Cosine distance of re-ID feature", fontsize=16)
plt.savefig('bar.pdf', bbox_inches='tight', dpi=100)
plt.close()
