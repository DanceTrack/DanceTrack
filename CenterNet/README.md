# CenterNet
An example of joint-training DanceTrack with mask, pose, depth datasets. 

This code is based on [CenterNet](https://github.com/xingyizhou/CenterNet) and [CenterNet-CondInst](https://github.com/CaoWGG/CenterNet-CondInst). We provide the trained models in [Google Drive](https://drive.google.com/drive/folders/1IHDFBwyVMJew78H6a-gGtXMOb-fGZRlh?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA)(code:awew).


| Training Data           |   Association |  HOTA    |   DetA   |  AssA    |   MOTA  |  IDF1   |
|-------------------------|---------------|----------|----------|----------|---------|---------|
|DanceTrack               |  bbox         | 36.9     |  63.6    |  21.6    |   78.8  |  39.2   |
|DanceTrack + COCO bbox   |  bbox         | 38.1     |  64.3    |  22.7    |   80.8  |  40.4   |
|DanceTrack + COCO mask   |  bbox         | 38.1     |  64.5    |  22.6    |   80.6  |  40.3   |
|DanceTrack + COCO mask   |  bbox + mask  | 39.2     |  64.9    |  23.9    |   80.7  |  41.6   |
|DanceTrack + COCO pose   |  bbox         | 40.6     |  65.5    |  25.3    |   82.9  |  42.9   |
|DanceTrack + COCO pose   |  bbox + pose  | 41.0     |  65.9    |  25.6    |   83.1  |  43.9   |
|DanceTrack + KITTI depth |  bbox         | 34.4     |  57.8    |  20.7    |   72.9  |  38.5   |
|DanceTrack + KITTI depth |  bbox + depth | 35.1     |  57.3    |  21.6    |   72.8  |  40.2   |

## Installation

* Our environment is linux, python3.7, pytorch1.7 and cu110. 
~~~
cd {DanceTrack ROOT}
cd CenterNet
pip install -r requirements.txt
cd src/lib/models/networks
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
python3 setup.py build develop
~~~
We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in the backbone. Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).

* Prepare DanceTrack dataset as in [Dataset](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Convert annotations to coco format:
~~~
cd {DanceTrack ROOT}
python3 tools/convert_dance_to_coco.py
cd CenterNet/data
ln -s ../../dancetrack dancetrack
~~~

* Download COCO 2017 images and annotations from [coco website](http://cocodataset.org/#download).
~~~
cd {DanceTrack ROOT}
cd CenterNet/data
ln -s {COCO PATH} coco
~~~

* Download KITTI [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), [annotations](http://www.cvlibs.net/download.php?file=data_object_label_2.zip), and [calibrations](http://www.cvlibs.net/download.php?file=data_object_calib.zip) from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Convert annotations to coco format:
~~~
cd {DanceTrack ROOT}
cd CenterNet/data
ln -s {KITTI PATH} kitti
cd ../src
python3 tools/convert_kitti_all_to_coco.py
~~~


## DanceTrack, bbox(Baseline)

* Train on DanceTrack with COCO pre-trained model [ctdet_coco_dla_1x.pth](https://drive.google.com/open?id=1r89_KNXyDyvUp8NggduG9uKQTMU2DsK_):
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py ctdet --dataset dancetrack --exp_id dancetrack_ctdet_dla --load_model ../models/ctdet_coco_dla_1x.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 2.5e-4  --save_all --num_epochs 12 --lr_step 10 --val_intervals 2
~~~
The trained model will be saved in CenterNet/exp/ctdet/dancetrack_ctdet_dla folder.

* Test on DanceTrack val:
~~~
python3 test.py ctdet --dataset dancetrack --exp_id dancetrack_ctdet_dla --load_model ../exp/ctdet/dancetrack_ctdet_dla/model_last.pth --tracking
~~~
The output txt will be saved in ../exp/ctdet/dancetrack_ctdet_dla/results folder.

* Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Demo on one of DanceTrack val videos:
~~~
python3 demo.py ctdet --num_classes 1 --demo ../../dancetrack/val/dancetrack000x/img1 --load_model ../exp/ctdet/dancetrack_ctdet_dla/model_last.pth --debug 4 --tracking
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ctdet/default/debug --video_name dancetrack000x_demo.avi
~~~
The output images will be saved in CenterNet/exp/ctdet/default/debug folder.


## DanceTrack+COCO bbox, bbox
* Prepare joint-dataset of DanceTrack and COCO as in [dancetrack_coco](https://github.com/DanceTrack/DanceTrack/blob/main/CenterNet/joint_train/dancetrack_coco.py).
~~~
cd {DanceTrack ROOT}
cd CenterNet
python3 joint_train/dancetrack_coco.py
~~~

* Train on joint-dataset with COCO pre-trained model [ctdet_coco_dla_1x.pth](https://drive.google.com/open?id=1r89_KNXyDyvUp8NggduG9uKQTMU2DsK_):
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py ctdet --dataset dancetrack_coco --exp_id dancetrack_coco_bbox_ctdet_dla --load_model ../models/ctdet_coco_dla_1x.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 2.5e-4  --save_all --num_epochs 12 --lr_step 10 --val_intervals 2 
~~~
The trained model will be saved in ../exp/ctdet/dancetrack_coco_bbox_ctdet_dla folder.

* Test on DanceTrack val:
~~~
python3 test.py ctdet --dataset dancetrack --num_classes 80 --exp_id dancetrack_coco_bbox_ctdet_dla --load_model ../exp/ctdet/dancetrack_coco_bbox_ctdet_dla/model_last.pth --tracking
~~~
The output txt will be saved in CenterNet/exp/ctdet/dancetrack_coco_bbox_ctdet_dla/results folder.

* Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Demo on one of DanceTrack val videos:
~~~
python3 demo.py ctdet --demo ../../dancetrack/val/dancetrack000x/img1 --load_model ../exp/ctdet/dancetrack_coco_bbox_ctdet_dla/model_last.pth --debug 4 --tracking
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ctdet/default/debug --video_name dancetrack000x_demo2.avi
~~~
The output images will be saved in CenterNet/exp/ctdet/default/debug folder.


## DanceTrack+COCO mask, bbox / mask
* Pre-train on COCO mask:
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py ctseg --exp_id ctseg_coco_dla --load_model ../models/ctdet_coco_dla_1x.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 5e-4  --save_all
python3 test.py ctseg --exp_id ctseg_coco_dla --load_model ../exp/ctseg/ctseg_coco_dla/model_last.pth
~~~
| type| AP |  AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> | 
|-----|------|-----|-----|-----|-----|-----|
|box|36.8|54.4|39.4|16.8|40.2|53.8|
|mask|31.0|49.8|32.0|11.0|34.4|49.4|

* Prepare joint-dataset of DanceTrack and COCO as in [dancetrack_coco](https://github.com/DanceTrack/DanceTrack/blob/main/CenterNet/joint_train/dancetrack_coco.py).
~~~
cd {DanceTrack ROOT}
cd CenterNet
python3 joint_train/dancetrack_coco.py
~~~

* Train on joint-dataset with COCO mask pre-trained model [ctseg_coco_dla_1x.pth](https://drive.google.com/file/d/1XTOuCtePKzrBQC2oiSY0x4P6lUL5yUK2/view?usp=sharing):
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py ctseg --dataset dancetrack_coco --exp_id dancetrack_coco_mask_ctseg_dla --load_model ../models/ctdet_coco_dla_1x.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 2.5e-4  --save_all --num_epochs 12 --lr_step 10 --val_intervals 2 
~~~
The trained model will be saved in CenterNet/exp/ctseg/dancetrack_coco_mask_ctseg_dla folder.

* Test on DanceTrack val:
~~~
python3 test.py ctseg --dataset dancetrack --num_classes 80 --exp_id dancetrack_coco_mask_ctseg_dla --load_model ../exp/ctseg/dancetrack_coco_mask_ctseg_dla/model_last.pth --tracking --fuse_mask --fast_mask
~~~
**--fuse_mask is associating with bbox + mask**, --fast_mask is not post-processing mask to speed up.

The output txt will be saved in CenterNet/exp/ctseg/dancetrack_coco_mask_ctseg_dla/results folder.

* Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Demo on one of DanceTrack val videos:
~~~
python3 demo.py ctseg --demo ../../dancetrack/val/dancetrack000x/img1 --load_model ../exp/ctseg/dancetrack_coco_mask_ctseg_dla/model_last.pth --debug 4 --tracking --fuse_mask
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ctseg/default/debug --video_name dancetrack000x_demo3.avi
~~~
The output images will be saved in CenterNet/exp/ctseg/default/debug folder.


## DanceTrack+COCO pose, bbox / pose
* Prepare joint-dataset of DanceTrack and COCO as in [dancetrack_coco_hp](https://github.com/DanceTrack/DanceTrack/blob/main/CenterNet/joint_train/dancetrack_coco_hp.py).
~~~
cd {DanceTrack ROOT}
cd CenterNet
python3 joint_train/dancetrack_coco_hp.py
~~~

* Train on joint-dataset with COCO pose pre-trained model [multi_pose_dla_3x.pth](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI):
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py multi_pose --dataset dancetrack_coco_hp --exp_id dancetrack_coco_hp_dla --load_model ../models/multi_pose_dla_3x.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 2.5e-4  --save_all --num_epochs 12 --lr_step 10 --val_intervals 2 
~~~
The trained model will be saved in CenterNet/exp/multi_pose/dancetrack_coco_hp_dla folder.

* Test on DanceTrack val:
~~~
python3 test.py multi_pose --dataset dancetrack --num_classes 1 --exp_id dancetrack_coco_hp_dla --load_model ../exp/multi_pose/dancetrack_coco_hp_dla/model_last.pth --tracking --fuse_pose
~~~
**--fuse_pose is associating with bbox + pose**.

The output txt will be saved in CenterNet/exp/multi_pose/dancetrack_coco_hp_dla/results folder.

* Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Demo on one of DanceTrack val videos:
~~~
python3 demo.py multi_pose --demo ../../dancetrack/val/dancetrack000x/img1 --load_model ../exp/multi_pose/dancetrack_coco_hp_dla/model_last.pth --debug 4 --tracking --fuse_pose
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/multi_pose/default/debug --video_name dancetrack000x_demo4.avi
~~~
The output images will be saved in CenterNet/exp/multi_pose/default/debug folder.


## DanceTrack+KITTI depth, bbox / depth
* Prepare joint-dataset of DanceTrack and KITTI as in [dancetrack_kitti](https://github.com/DanceTrack/DanceTrack/blob/main/CenterNet/joint_train/dancetrack_kitti.py).
~~~
cd {DanceTrack ROOT}
cd CenterNet
python3 joint_train/dancetrack_kitti.py
~~~

* Train on joint-dataset with KITTI pre-trained model [ddd_3dop.pth](https://drive.google.com/open?id=1znsM6E-aVTkATreDuUVxoU0ajL1az8rz):
~~~
cd {DanceTrack ROOT}
cd CenterNet/src
python3 main.py ddd --dataset dancetrack_kitti --exp_id dancetrack_kitti --load_model ../models/ddd_3dop.pth --batch_size 128 --master_batch 15 --gpus 0,1,2,3,4,5,6,7 --lr 2.5e-4  --save_all --num_epochs 12 --lr_step 10 --val_intervals 2 --remove_black
~~~
--remove_black will crop the KITTI image to avoid big padding when joint-training of DanceTrack and KITTI.

The trained model will be saved in CenterNet/exp/ddd/dancetrack_kitti folder.

* Test on DanceTrack val:
~~~
python3 test.py ddd --dataset dancetrack --num_classes 3 --exp_id dancetrack_kitti --load_model ../exp/ddd/dancetrack_kitti/model_last.pth --tracking --fuse_depth
~~~
**--fuse_depth is associating with bbox + depth**.

The output txt will be saved in CenterNet/exp/ddd/dancetrack_kitti/results folder.

* Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Demo on one of DanceTrack val videos:
~~~
python3 demo.py ddd --demo ../../dancetrack/val/dancetrack000x/img1 --load_model ../exp/ddd/dancetrack_kitti/model_last.pth --debug 4 --tracking --fuse_depth --world_size 16
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ddd/default/debug --video_name dancetrack000x_demo5.avi --show_width 1500
~~~
Adjust --world_size to get better bird-view visualization.

The output images will be saved in CenterNet/exp/ddd/default/debug folder.

## Citation

If you find this project useful for your research, please use DanceTrack and CenterNet BibTeX entry.

```BibTeX
@article{peize2021dance,
  title   =  {DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion},
  author  =  {Peize Sun and Jinkun Cao and Yi Jiang and Zehuan Yuan and Song Bai and Kris Kitani and Ping Luo},
  journal =  {arXiv preprint arXiv:2111.14690},
  year    =  {2021}
}

@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```
