from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob as gb
import numpy as np
from deepsort_tracker.appearance_tracker import ATracker
import argparse


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
                    1.0]
            if int(linelist[7]) == 1:
                if int(img_id) in det_results:
                    det_results[int(img_id)].append(bbox)
                else:
                    det_results[int(img_id)] = list()
                    det_results[int(img_id)].append(bbox)
    f.close()
    
    results = []
#     star_idx = len(gb.glob(os.path.join(dataset, video_name, 'img1') + "/*.jpg")) // 2 + 1
    tracker = ATracker(model_path='ckpt.t7', min_confidence=0.4, n_init=0)
    for frame_id in sorted(det_results.keys()):
        det = det_results[frame_id]
        det = np.array(det)
#         image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id + star_idx))
        image_path = os.path.join(dataset, video_name, 'img1', '{:0>8d}.jpg'.format(frame_id))
        online_targets = tracker.update(det, image_path)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = t[4]
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
        results.append((frame_id, online_tlwhs, online_ids))
    
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    filename = os.path.join(val_pred, video_name) + '.txt'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=int(track_id), x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    f.close()
