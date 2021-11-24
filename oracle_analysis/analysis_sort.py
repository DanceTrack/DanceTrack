import os
import glob
from sort import Sort
import numpy as np

# dataset = 'mot/val'
dataset = 'dancetrack/val'

val_pred = 'oracle_analysis/val_sort'
if not os.path.exists(val_pred):
    os.makedirs(val_pred)

val_seqs = sorted(os.listdir(dataset))
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
    tracker = Sort(det_thresh=0.4)
    for frame_id in sorted(det_results.keys()):
        det = det_results[frame_id]
        det = np.array(det)
        online_targets = tracker.update(det)
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
