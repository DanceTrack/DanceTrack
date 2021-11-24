import json
import os


"""
cd {DanceTrack ROOT}/CenterNet
cd data
mkdir -p dancetrack_kitti/annotations

cd dancetrack_kitti
ln -s ../kitti/training/image_2 kitti_train
ln -s ../dancetrack/train dancetrack_train
cd ../..
"""

print('kitti is loading...')
kitti_json = json.load(open('data/kitti/annotations/kitti_train.json','r'))

max_img_id = 0
img_list = list()
for img in kitti_json['images']:
    img['file_name'] = 'kitti_train/' + img['file_name']
    img_list.append(img)
    max_img_id = max(max_img_id, int(img['id']))

max_ann_id = 0    
ann_list = list()
for ann in kitti_json['annotations']:
    ann_list.append(ann)
    max_ann_id = max(max_ann_id, int(ann['id']))

category_list = kitti_json['categories']




print('dancetrack is loading...')
dancetrack_json = json.load(open('data/dancetrack/annotations/train.json','r'))

img_id_count = 0
for img in dancetrack_json['images']:
    img_id_count += 1
    img['file_name'] = 'dancetrack_train/' + img['file_name']
    img['id'] = img['id'] + max_img_id
    img_list.append(img)
    
for ann in dancetrack_json['annotations']:
    ann['id'] = ann['id'] + max_ann_id
    ann['image_id'] = ann['image_id'] + max_img_id
    ann['category_id'] = 1
    ann_list.append(ann)


    
print('mix is saving...')
mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['categories'] = category_list
json.dump(mix_json, open('data/dancetrack_kitti/annotations/train.json','w'))