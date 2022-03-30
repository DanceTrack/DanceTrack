# ByteTrack
An example of using DanceTrack. 

This code is based on [ByteTrack](https://github.com/ifzhang/ByteTrack). We provide the trained models in [Google Drive](https://drive.google.com/drive/u/2/folders/1v1hiIrgH0b5-ZavRAVa1MJS8iGr-Pn5U) or [Baidu Drive](https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA)(code:awew).

## Installation
* Follow installation steps in [ByteTrack](https://github.com/ifzhang/ByteTrack). Replace the original files with updated ones here.

* Prepare DanceTrack dataset as in [Dataset](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).

* Convert annotations to coco format:
~~~
cd {DanceTrack ROOT}
python3 tools/convert_dance_to_coco.py
cd ByteTrack/datasets
ln -s ../../dancetrack dancetrack
cd ..
~~~


## Training 
The COCO pretrained YOLOX model can be downloaded from their [model zoo](https://github.com/Megvii-BaseDetection/YOLOX). After downloading the pretrained models, put them under {DanceTrack ROOT}/ByteTrack/pretrained.
~~~
cd {DanceTrack ROOT}/ByteTrack
python3 tools/train.py -f exps/example/dancetrack/yolox_x.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
~~~



## Evaluating
~~~
python3 tools/track.py -f exps/example/dancetrack/yolox_x.py -c YOLOX_outputs/yolox_x/latest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse
~~~
The output txt will be saved in YOLOX_outputs/yolox_x/track_results folder.

Evaluate the results as in [Evaluation](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).


## Test set
~~~
python3 tools/track.py -f exps/example/dancetrack/yolox_x.py -c YOLOX_outputs/yolox_x/latest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse --test
~~~
The output txt will be saved in YOLOX_outputs/yolox_x/track_test_results folder.

Submit the results to CodaLab as in [Competition](https://github.com/DanceTrack/DanceTrack/blob/main/README.md).


## Citation
If you find this project useful for your research, please use DanceTrack and ByteTrack BibTeX entry.

```BibTeX
@article{peize2021dance,
  title   =  {DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion},
  author  =  {Peize Sun and Jinkun Cao and Yi Jiang and Zehuan Yuan and Song Bai and Kris Kitani and Ping Luo},
  journal =  {arXiv preprint arXiv:2111.14690},
  year    =  {2021}
}

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
