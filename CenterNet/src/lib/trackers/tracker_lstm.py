from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import torch
from .mot_online.kalman_filter import KalmanFilter
from .mot_online.kalman_filter_lstm import KalmanFilterLSTM
from .mot_online.basetrack import BaseTrack, TrackState
from .mot_online import matching
from .utils import fuse_mask, fuse_pose, fuse_depth
from pycocotools import  mask as mask_utils
from opts import opts


opt = opts().parse()

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_lstm = KalmanFilterLSTM(opt)

    def __init__(self, tlwh, score, use_lstm=True, hp=None, mask=None, backup=None):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.hp = hp
        self.mask = mask
        self.backup = backup
        self.tracklet_len = 0
        
        self.use_lstm = use_lstm
        self.opt = opt
        self.last_h = -1
        self.last_w = -1
        self.last_cx = 0
        self.last_cy = 0
        self.first_time = True
        self.last_frame_id = -1
        
        self.future_predictions = {}
        opt.device = torch.device("cuda" if opt.gpus[0] >= 0 else "cpu")
        self.device = opt.device
        self.hn = torch.zeros(1, 1, 128).to(device=self.device).float()
        self.cn = torch.zeros(1, 1, 128).to(device=self.device).float()
        self.covariance = np.eye(4, 4)
        self.observations = []
        self.observations_tlwh = []
        self.observations_tlwh.append(self._tlwh.copy())
        
        
    def predict(self):
        if not self.use_lstm:
            mean_state = self.mean.copy()
            if self.state != TrackState.Tracked:
                mean_state[7] = 0
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
            
    def prediction_at_frame(self, frame_id):
        max_fut = 6
        
        if frame_id in [self.frame_id + i for i in range(1, max_fut)]:
            return self.future_predictions[frame_id - self.frame_id]
        else:
            return self.future_predictions[max_fut - 1]
    
    def prediction_at_frame_tlbr(self, frame_id):
        ret = self.prediction_at_frame(frame_id).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]

        return ret
    
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        if self.use_lstm:
            self.kalman_filter = KalmanFilterLSTM(self.opt)
            self.update_lstm_features(self._tlwh)

        else:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.tlwh_to_xyah(self._tlwh)
            )

    def re_activate(self, new_track, frame_id, new_id=False):

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.hp = new_track.hp
        self.mask = new_track.mask
        self.backup = new_track.backup
        
        if self.use_lstm:
            self.update_lstm_features(new_track.tlwh)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
            )    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.hp = new_track.hp
        self.mask = new_track.mask
        self.backup = new_track.backup
        
        if self.use_lstm:
            self.update_lstm_features(new_track.tlwh)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
            )
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.use_lstm:
            ret = self.observations_tlwh[-1].copy()
        else:
            if self.mean is None:
                return self._tlwh.copy()
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2

        return ret
    
    def update_lstm_features(self, tlwh):

        self.observations_tlwh.append(tlwh.copy())
        self.observations.append(self.tlwh_to_xyah(tlwh).tolist())
        self.covariance = np.cov(np.asarray(self.observations).copy().T)
        tlwh = tlwh.copy()
        tlwh[:2] += tlwh[2:] / 2

        tlwh_list = tlwh.tolist()
        c_x, c_y, w, h = tlwh_list[0], tlwh_list[1], tlwh_list[2], tlwh_list[3]
        h_w_ratio = w / h

        new_features = []
        if self.first_time:
            self.last_h = h
            self.last_w = w
            self.last_cx = c_x
            self.last_cy = c_y
            self.last_frame_id = self.frame_id
            self.first_time = False
            delta_h = 0
            delta_w = 0
            v_x = 0
            v_y = 0
            delta_cx = 0
            delta_cy = 0
        else:
            delta_h = h - self.last_h
            delta_w = w - self.last_w
            v_x = (c_x - self.last_cx) / (self.frame_id - self.last_frame_id)
            v_y = (c_y - self.last_cy) / (self.frame_id - self.last_frame_id)
            delta_cx = (c_x - self.last_cx) / (self.frame_id - self.last_frame_id)
            delta_cy = (c_y - self.last_cy) / (self.frame_id - self.last_frame_id)

            self.last_h = h
            self.last_w = w
            self.last_cx = c_x
            self.last_cy = c_y
            self.last_frame_id = self.frame_id

        new_features.append(
            [
                c_x,
                c_y,
                delta_cx,
                delta_cy,
                h,
                w,
                h_w_ratio,
                delta_h,
                delta_w,
                v_x,
                v_y,
            ].copy()
        )
        new_features = np.array(new_features)
        new_features = torch.from_numpy(new_features).unsqueeze(0)
        new_features = new_features.to(device=self.device).float()

        self.hn, self.cn, self.future_predictions = self.shared_kalman_lstm.predict(
            self.hn, self.cn, new_features
        )

        for key in self.future_predictions:
            self.future_predictions[key][:2] += tlwh[:2]
            self.future_predictions[key][2] += tlwh[3]
            self.future_predictions[key][3] += tlwh[2]
            pred_h = self.future_predictions[key][2]
            pred_w = self.future_predictions[key][3]
            self.future_predictions[key][3] = pred_h
            self.future_predictions[key][2] = pred_w

            self.future_predictions[key][2] /= self.future_predictions[key][3]

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.buffer_size = opt.buffer_size
        self.max_time_lost = self.buffer_size
        self.use_lstm = opt.lstm  
        self.reset()

    def reset(self):
        self.frame_id = 0
        if self.use_lstm:        
            STrack.shared_kalman_lstm = KalmanFilterLSTM(opt)
            self.kalman_filter = KalmanFilterLSTM(opt)
        else:
            self.kalman_filter = KalmanFilter()
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.tracks = []    
    
        
    def step(self, results_pre):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        detections = []
        
        if self.opt.task == 'ctdet':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j])):
                    det = results_pre[j][i]
                    if det[4] < self.opt.track_thresh:
                        continue
                    bbox = det[:4]
                    score = det[4]
                    detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score))
                
        elif self.opt.task == 'ctseg':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j]['boxs'])):
                    boxs = results_pre[j]['boxs'][i]
                    if boxs[4] < self.opt.track_thresh:
                        continue
                    bbox = boxs[:4]
                    score = boxs[4]
#                     mask = mask_utils.decode(results_pre[j]['pred_mask'][i])
                    mask = results_pre[j]['pred_mask'][i]                    
                    detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, mask=mask))
        
        elif self.opt.task == 'multi_pose':
            dets = results_pre[1]
            for i, det in enumerate(dets):
                if det[4] < self.opt.track_thresh:
                    continue
                bbox = det[:4]
                score = det[4]
                hp = det[5:39]
                detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, hp=hp))
        
        elif self.opt.task == 'ddd':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j])):
                    det = results_pre[j][i]
                    score = det[-1]
                    if score < self.opt.track_thresh:
                        continue
#                     bbox = det[1:5]
                    left, top, right, bottom = det[1:5]
                    if left > right:
                        left, right = right, left
                    bbox = [left, top, right, bottom]
                    
                    detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, backup=det))
#             return results_pre
            
        else:
            raise NotImplementedError
        
        ''' Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: Association with Kalman and IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        if not self.use_lstm:
            STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(
            strack_pool, 
            detections,
            self.frame_id,
            use_prediction=self.use_lstm)
        
        if dists.shape[0] > 0 and dists.shape[1] > 0:
            if self.opt.fuse_mask:
                dists = fuse_mask(strack_pool, detections, dists)
            elif self.opt.fuse_pose:
                dists = fuse_pose(strack_pool, detections, dists)        
            elif self.opt.fuse_depth:
                dists = fuse_depth(strack_pool, detections, dists)        
        
        
        #dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.opt.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        ret = []
        for track in output_stracks:
            track_dict = {}
            track_dict['score'] = track.score
            track_dict['bbox'] = track.tlbr
            track_dict['hps'] = track.hp
            track_dict['mask'] = track.mask
            track_dict['active'] = 1 if track.is_activated else 0
            track_dict['tracking_id'] = track.track_id
            track_dict['class'] = 0
            track_dict['backup'] = track.backup
            ret.append(track_dict)
        
        self.tracks = ret
        return ret        


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain

