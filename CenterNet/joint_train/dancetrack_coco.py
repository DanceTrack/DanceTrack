import json
import os


"""
cd {DanceTrack ROOT}/CenterNet
cd data
mkdir -p dancetrack_coco/annotations

cd dancetrack_coco
ln -s ../coco/train2017 coco_train
ln -s ../dancetrack/train dancetrack_train
cd ../..
"""

print('coco is loading...')
coco_json = json.load(open('data/coco/annotations/instances_train2017.json','r'))

max_img_id = 0
img_list = list()
for img in coco_json['images']:
    img['file_name'] = 'coco_train/' + img['file_name']
    img_list.append(img)
    max_img_id = max(max_img_id, int(img['id']))

max_ann_id = 0    
ann_list = list()
for ann in coco_json['annotations']:
    ann_list.append(ann)
    max_ann_id = max(max_ann_id, int(ann['id']))

category_list = coco_json['categories']




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
json.dump(mix_json, open('data/dancetrack_coco/annotations/train.json','w'))