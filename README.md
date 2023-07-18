# DanceTrack

DanceTrack is a benchmark for tracking multiple objects in uniform appearance and diverse motion.

DanceTrack provides box and identity annotations. It contains 100 videos, 40 for training(annotations public), 25 for validation(annotations public) and 35 for testing(annotations unpublic). For evaluating on test set, please see [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/5830)([Old CodaLab](https://competitions.codalab.org/competitions/35786)). We also have a [Project Page](https://dancetrack.github.io/) for exhibition. 

<div align="center"><img src="assets/demo.jpg" ></div>
</br>

## Paper (CVPR2022)
[DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion](https://arxiv.org/abs/2111.14690)


## News
- (05/2023) We list some awesome research papers and projects here you may find interesting. Welcome new pull request to add new links !
- (04/2022) We are organizing [Multiple Object Tracking and Segmentation in Complex Environments Workshop](https://motcomplex.github.io/), ECCV 2022. 


## Paper List

| Title | Intro | Description | Links |
|:----:|:----:|:----:|:----:|
| [SUSHI](https://arxiv.org/abs/2212.03038) | ![](https://github.com/dvl-tum/SUSHI/raw/main/assets/teaser.png) | Unifying Short and Long-Term Tracking with Graph Hierarchies | [[Github](https://github.com/dvl-tum/SUSHI)]|
| [MOTRv2](https://arxiv.org/abs/2211.09791) | ![](https://raw.githubusercontent.com/zyayoung/oss/main/motrv2_main.jpg) | MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors | [[Github](https://github.com/megvii-research/MOTRv2)]|
| [MOT_FCG](https://arxiv.org/abs/2210.03355) | Multiple Object Tracking from appearance by hierarchically clustering tracklets | Multiple Object Tracking from appearance by hierarchically clustering tracklets | [[Github](https://github.com/NII-Satoh-Lab/MOT_FCG)]|
| [OC-SORT](https://arxiv.org/abs/2203.14360) | ![](https://github.com/noahcao/OC_SORT/raw/master/assets/teaser.png) | Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking| [[Github](https://github.com/noahcao/OC_SORT)]|
| [StrongSORT](https://arxiv.org/abs/2202.13514) | ![](https://github.com/dyhBUPT/StrongSORT/raw/master/assets/MOTA-IDF1-HOTA.png) | StrongSORT: Make DeepSORT Great Again | [[Github](https://github.com/dyhBUPT/StrongSORT)]|
| [MOTR](https://arxiv.org/abs/2105.03247) | ![](https://github.com/megvii-research/MOTR/blob/main/figs/motr.png) | MOTR: End-to-End Multiple-Object Tracking with TRansformer | [[Github](https://github.com/megvii-research/MOTR)]|


## Dataset
Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA) (code:awew).

Organize as follows:
~~~
{DanceTrack ROOT}
|-- dancetrack
|   |-- train
|   |   |-- dancetrack0001
|   |   |   |-- img1
|   |   |   |   |-- 00000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- val
|   |   |-- ...
|   |-- test
|   |   |-- ...
|   |-- train_seqmap.txt
|   |-- val_seqmap.txt
|   |-- test_seqmap.txt
|-- TrackEval
|-- tools
|-- ...
~~~
We align our dataset annotations with MOT, so each line in  gt.txt contains:
~~~
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
~~~



## Evaluation
We use [ByteTrack](https://github.com/ifzhang/ByteTrack) as an example of using DanceTrack. For training details, please see [instruction](ByteTrack/README.md). We provide the trained models in [Google Drive](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea?usp=sharing) or or [Baidu Drive](https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA) (code:awew).

To do evaluation with our provided tookit, we organize the results of validation set as follows:
~~~
{DanceTrack ROOT}
|-- val
|   |-- TRACKER_NAME
|   |   |-- dancetrack000x.txt
|   |   |-- ...
|   |-- ...
~~~
where dancetrack000x.txt is the output file of the video episode dancetrack000x, each line of which contains:
~~~
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
~~~

Then, simply run the evalution code:
```
python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER dancetrack/val --SEQMAP_FILE dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER val/TRACKER_NAME 
```

| Tracker     |   HOTA  |   DetA  |   AssA  |   MOTA  |   IDF1  |
|-------------|---------|---------|---------|---------|---------|
| ByteTrack   |  47.1   |  70.5   |   31.5  |   88.2  |  51.9   |

    
Besides, we also provide the visualization script. The usage is as follow:
``` 
python3 tools/txt2video_dance.py --img_path dancetrack --split val --tracker TRACKER_NAME
```
<p align="center"> <img src='assets/bbox_demo.gif' align="center" height="250px">  </p>



## Competition

Organize the results of test set as follows:
~~~
{DanceTrack ROOT}
|-- test
|   |-- tracker
|   |   |-- dancetrack000x.txt
|   |   |-- ...
~~~
Each line of dancetrack000x.txt contains:
~~~
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
~~~

Archive tracker folder to tracker.zip and submit to [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/5830). Please note: (1) archive tracker folder, instead of txt files. (2) the folder name must be tracker. 

The return will be:  

| Tracker     |   HOTA  |   DetA  |   AssA  |   MOTA  |   IDF1  |
|-------------|---------|---------|---------|---------|---------|
| tracker     |  47.7   |  71.0   |   32.1  |   89.6  |  53.9   |

For more detailed metrics and metrics on each video, click on [download output from scoring step](https://competitions.codalab.org/competitions/35786#participate-submit_results) in CodaLab.

Run the visualization code:
``` 
python3 tools/txt2video_dance.py --img_path dancetrack --split test --tracker tracker
```


## Joint-Training
We use joint-training with other datasets to predict mask, pose and depth.  [CenterNet](https://github.com/xingyizhou/CenterNet) is provided as an example. For details of joint-trainig, please see [joint-training instruction](CenterNet/README.md). We provide the trained models in [Google Drive](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA)(code:awew).
 
For mask demo, run
~~~
cd CenterNet/src
python3 demo.py ctseg --demo  ../../dancetrack/val/dancetrack000x/img1 --load_model ../models/dancetrack_coco_mask.pth --debug 4 --tracking 
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ctseg/default/debug --video_name dancetrack000x_mask.avi
~~~
<p align="center"> <img src='assets/ctseg.gif' align="center" height="250px">  </p>
  

For pose demo, run
~~~
cd CenterNet/src
python3 demo.py multi_pose --demo  ../../dancetrack/val/dancetrack000x/img1 --load_model ../models/dancetrack_coco_pose.pth --debug 4 --tracking 
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/multi_pose/default/debug --video_name dancetrack000x_pose.avi
~~~
<p align="center"> <img src='assets/multi_pose.gif' align="center" height="250px">  </p>


For depth demo, run
~~~
cd CenterNet/src
python3 demo.py ddd --demo  ../../dancetrack/val/dancetrack000x/img1 --load_model ../models/dancetrack_kitti_ddd.pth --debug 4 --tracking --test_focal_length 640 --world_size 16 --out_size 128
cd ../..
python3 tools/img2video.py --img_file CenterNet/exp/ddd/default/debug --video_name dancetrack000x_ddd.avi
~~~
<p align="center"> <img src='assets/ddd.gif' align="center" height="250px">  </p>
  
  


## Agreement
- The annotations of DanceTrack are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).
- The dataset of DanceTrack is available for **non-commercial** research purposes only.
- All videos and images of DanceTrack are obtained from the Internet which are not property of HKU, CMU or ByteDance. These three organizations are not responsible for the content nor the meaning of these videos and images.
- The code of DanceTrack is released under the MIT License.  


  

## Acknowledgement  
 
The evaluation metrics and code are from [MOT Challenge](https://motchallenge.net/) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). The inference code is from [ByteTrack](https://github.com/ifzhang/ByteTrack). The joint-training code is modified from [CenterTrack](https://github.com/xingyizhou/CenterTrack) and [CenterNet](https://github.com/xingyizhou/CenterNet), where the instance segmentation code is from [CenterNet-CondInst](https://github.com/CaoWGG/CenterNet-CondInst). Thanks for their wonderful and pioneering works !
  
  

## Citation

If you use DanceTrack in your research or wish to refer to the baseline results published here, please use the following BibTeX entry:

```BibTeX

@article{peize2021dance,
  title   =  {DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion},
  author  =  {Peize Sun and Jinkun Cao and Yi Jiang and Zehuan Yuan and Song Bai and Kris Kitani and Ping Luo},
  journal =  {arXiv preprint arXiv:2111.14690},
  year    =  {2021}
}

```
