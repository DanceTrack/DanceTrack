from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctseg import CTSegDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.dancetrack import DanceTrack
from .dataset.dancetrack_coco import DanceTrackCOCO
from .dataset.dancetrack_coco_hp import DanceTrackCOCOHP
from .dataset.dancetrack_kitti import DanceTrackKITTI

dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP,
    'dancetrack': DanceTrack,
    'dancetrack_coco': DanceTrackCOCO,
    'dancetrack_coco_hp': DanceTrackCOCOHP,
    'dancetrack_kitti': DanceTrackKITTI

}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset,
    'ctseg': CTSegDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset

