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
    
    feat_results = {}
    star_idx = len(gb.glob(os.path.join(dataset, video_name, 'img1') + "/*.jpg")) // 2 + 1
    tracker = AppearanceFeature(model_path='ckpt.t7')
    for frame_id in sorted(det_results.keys())[:200]:
        dets = det_results[frame_id]
        dets = np.array(dets)
#         image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id + star_idx))
        image_path = os.path.join(dataset, video_name, 'img1', '{:0>8d}.jpg'.format(frame_id))
        
        appearance_feat = tracker.update(dets, image_path)
        for idx, det in enumerate(dets):
            track_id = int(det[-1])
            if track_id in feat_results:
                feat_results[track_id].append(appearance_feat[idx])
            else:
                feat_results[track_id] = []
                feat_results[track_id].append(appearance_feat[idx])
    
    show_num_objects = 8
    tsne = TSNE(n_components=2, init='pca', random_state=1)

    embedding_collection = list()
    for track_id in sorted(feat_results.keys())[:show_num_objects]:
        embedding_collection.extend(feat_results[track_id])

    embedding_collection = np.stack(embedding_collection)
    tsne_points = tsne.fit_transform(embedding_collection)
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    start_idx = 0
    for idx, track_id in enumerate(sorted(feat_results.keys())[:show_num_objects]):
        num_points = len(feat_results[track_id])
        plt.scatter(
            tsne_points[start_idx:start_idx + num_points, 0],
            tsne_points[start_idx:start_idx + num_points, 1],
            c=colors[idx],
            s=10,
        )
        start_idx += num_points
    
#     plt.title("MOT17-02")
#     plt.title("DanceTrack"), 19 26 34, 6 8 10
#     plt.legend(['0', '1', '2', '3', '4', '5', '6', '7'], loc='upper right')
    
#     plt.savefig('tsne_mot17_02.png', bbox_inches='tight') 
    plt.savefig('tsne_dancetrack0019.png', bbox_inches='tight')

    plt.close()
    
    exit()
