import os
import glob

# dataset = 'mot/val'
dataset = 'dancetrack/val'

val_pred = 'oracle_analysis/val_pred'
if not os.path.exists(val_pred):
    os.makedirs(val_pred)

val_seqs = os.listdir(dataset)
for video_name in val_seqs:
    print(video_name)
    results = []
    with open(os.path.join(dataset, video_name, 'gt/gt.txt'), 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            results.append(linelist)
    f.close()
    
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    filename = os.path.join(val_pred, video_name) + '.txt'
    with open(filename, 'w') as f:
        for i in range(len(results)):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            category_id = int(frame_data[7])
            if category_id == 1:
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)
    f.close()
