from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def dice_loss(pred, target):
    '''
    pred:   (M, H, W)
    target: (N, H, W)
    '''
    smooth = 1.
    pflat = pred.reshape(pred.shape[0], 1, -1)
    tflat = target.reshape(1, target.shape[0], -1)
    intersection = (pflat * tflat).sum(-1)
    
    return 1 - ((2. * intersection + smooth) / ((pflat * pflat).sum(-1) + (tflat * tflat).sum(-1) + smooth))
    


def fuse_mask(tracks, dets, cost_matrics, lambda_=0.8):
    tracks_mask = np.stack([track.mask for track in tracks])
    dets_mask = np.stack([det.mask for det in dets])
    mask_dists = dice_loss(tracks_mask, dets_mask)
    cost_matrics = lambda_ * cost_matrics + (1 - lambda_) * mask_dists

    return cost_matrics    



def oks_dist(predict, anno, delta=0.05):
    '''
    predict: (M, K=17, 2)
    anno:    (N, K=17, 2)
    '''

    M, K = predict.shape[:2]
    N, K = anno.shape[:2]

    predict = predict.reshape(M, 1, K, -1).repeat(N, axis=1)    
    anno = anno.reshape(1, N, K, -1).repeat(M, axis=0)  
    xmax = np.max(np.concatenate((anno[:,:,:, 0], predict[:,:,:, 0]), axis=-1), axis=-1)
    xmin = np.min(np.concatenate((anno[:,:,:, 0], predict[:,:,:, 0]), axis=-1), axis=-1)
    ymax = np.max(np.concatenate((anno[:,:,:, 1], predict[:,:,:, 1]), axis=-1), axis=-1)
    ymin = np.min(np.concatenate((anno[:,:,:, 1], predict[:,:,:, 1]), axis=-1), axis=-1)
    scale = ((xmax - xmin) * (ymax - ymin)).reshape(M, N, 1)    
    
    dist = np.sum((anno - predict)**2, axis=-1)
    oks = np.mean(np.exp(-dist / 2 / scale / delta**2), axis=-1)

    return 1 - oks



def fuse_pose(tracks, dets, cost_matrics, lambda_=0.8):
    tracks_hp = np.stack([np.array(track.hp).reshape(-1, 2) for track in tracks])
    dets_hp = np.stack([np.array(det.hp).reshape(-1, 2) for det in dets])
    pose_dists = oks_dist(tracks_hp, dets_hp)
    cost_matrics = lambda_ * cost_matrics + (1 - lambda_) * pose_dists

    return cost_matrics



def fuse_depth(tracks, dets, cost_matrics, lambda_=0.8):
    
    tracks_depth = np.array([track.backup[10] for track in tracks])
    dets_depth = np.array([det.backup[10] for det in dets])

    tracks_depth = tracks_depth.reshape(-1, 1)
    dets_depth = dets_depth.reshape(1, -1)
    depth_dists = (tracks_depth - dets_depth) ** 2
    depth_dists = depth_dists / np.max(depth_dists)
    
    cost_matrics = lambda_ * cost_matrics + (1 - lambda_) * depth_dists
    
    return cost_matrics