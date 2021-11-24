from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import copy
import os

from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector


class DddDetector(BaseDetector):
    def __init__(self, opt):
        super(DddDetector, self).__init__(opt)
        self.calib = np.array([[707.0493, 0, 960/2, 0],
                               [0, 707.0493, 544/2, 0],
                               [0, 0,        1.,    0]], dtype=np.float32)

    def pre_process(self, image, scale, calib=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        c = np.array([new_width / 2, new_height / 2], dtype=np.float32)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            s = np.array([inp_width, inp_height], dtype=np.int32)
        
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.mean) / self.std
        images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]
#         calib = np.array(calib, dtype=np.float32) if calib is not None \
#             else self.calib
        
        calib = self._get_default_calib(width, height, self.opt.test_focal_length ) if self.opt.test_focal_length > 0 \
            else self.calib
        
        images = torch.from_numpy(images)

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio,
                'calib': calib}
        return images, meta
    
    def _get_default_calib(self, width, height, test_focal_length):
        calib = np.array([[test_focal_length, 0, width / 2, 0], 
                          [0, test_focal_length, height / 2, 0], 
                          [0, 0, 1, 0]], dtype=np.float32)
        return calib

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            wh = output['wh'] if self.opt.reg_bbox else None
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                              output['dim'], wh=wh, reg=reg, K=self.opt.K)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = ddd_post_process(
            dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt, meta['out_height'], meta['out_width'])
        self.this_calib = meta['calib']
        return detections[0]

    def merge_outputs(self, detections):
        results = detections[0]
        for j in range(1, self.num_classes + 1):
            if len(results[j] > 0):
                keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy()
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.std + self.mean) * 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_ct_detection(
            img, dets[0], show_box=self.opt.reg_bbox,
            center_thresh=self.opt.vis_thresh, img_id='det_pred')

    def show_results(self, debugger, image, results):        
        if self.opt.tracking:
            debugger.add_img(image, img_id='add_pred')
            for item in results:
                if item['active'] < 1:
                    continue
                score = item['score']
                bbox = item['bbox']
                cat = item['class']
                tracking_id = item['tracking_id']
                if score > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox, cat, score, track_id=tracking_id, img_id='add_pred')
            
            debugger.assist_bird_view(
                results, center_thresh=self.opt.vis_thresh, img_id='add_pred')
            
            if self.opt.debug == 4:
                if not os.path.exists(self.opt.debug_dir):
                    os.makedirs(self.opt.debug_dir)
                debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt), specific='add_pred')
            else:
                debugger.show_all_imgs(pause=self.pause)
            
        else:
            debugger.add_3d_detection(
                image, results, self.this_calib,
                center_thresh=self.opt.vis_thresh, img_id='add_pred')
            debugger.add_bird_view(
                results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')

            if self.opt.debug == 4:
                if not os.path.exists(self.opt.debug_dir):
                    os.makedirs(self.opt.debug_dir)
                debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt), specific='add_pred')
                debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt), specific='bird_pred')
            else:
                debugger.show_all_imgs(pause=self.pause)