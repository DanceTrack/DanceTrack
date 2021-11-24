import os
import json

vis_path = 'annotations_txt'
if not os.path.exists(vis_path):
    os.makedirs(vis_path)

for i in list(range(1, 21)):
    json_file = json.load(open('DanceTrack_json/{}.json'.format(i), 'r'))
    file_path = os.path.join(vis_path, "{}.txt".format(i))
    f = open(file_path, "w")
    for dancer_ind, dancer_ann in enumerate(json_file['instances']):
        for dancer_img in dancer_ann['frames']:
            t = list()
            t.append(dancer_img['frameIndex'] + 1)
            t.append(dancer_ind)
            inst = dancer_img['shape']
            x = inst['x']
            y = inst['y']
            w = inst['width']
            h = inst['height']
            t.extend([x, y, w, h])
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f}, 1, 1, 1\n".format(
                t[0], t[1], t[2], t[3], t[4], t[5]))
    f.close()
