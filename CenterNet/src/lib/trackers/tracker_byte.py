from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
from .mot_online.kalman_filter import KalmanFilter
from .mot_online.basetrack import BaseTrack, TrackState
from .mot_online import matching
from .utils import fuse_mask, fuse_pose, fuse_depth
from pycocotools import mask as mask_utils


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, hp=None, mask=None, backup=None):
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

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

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
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
#         self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
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
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.hp = new_track.hp
        self.mask = new_track.mask
        self.backup = new_track.backup
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

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
        self.reset()
    
    def reset(self):
        self.frame_id = 0
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
        detections_second = []
        
        if self.opt.task == 'ctdet':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j])):
                    det = results_pre[j][i]
                    score = det[4]
                    if score <= self.opt.out_thresh:
                        continue
                    bbox = det[:4]
                    if score > self.opt.track_thresh:
                        detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score))
                    else:
                        detections_second.append(STrack(STrack.tlbr_to_tlwh(bbox), score))
                        
        elif self.opt.task == 'ctseg':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j]['boxs'])):
                    boxs = results_pre[j]['boxs'][i]
                    score = boxs[4]
                    if score <= self.opt.out_thresh:
                        continue
                    bbox = boxs[:4]
#                     mask = mask_utils.decode(results_pre[j]['pred_mask'][i])
                    mask = results_pre[j]['pred_mask'][i]
                    if score > self.opt.track_thresh:
                        detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, mask=mask))
                    else:
                        detections_second.append(STrack(STrack.tlbr_to_tlwh(bbox), score, mask=mask))
                        
        elif self.opt.task == 'multi_pose':
            dets = results_pre[1]
            for i, det in enumerate(dets):
                score = det[4]
                if score <= self.opt.out_thresh:
                    continue
                bbox = det[:4]
                hp = det[5:39]
                if score > self.opt.track_thresh:
                    detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, hp=hp))
                else:
                    detections_second.append(STrack(STrack.tlbr_to_tlwh(bbox), score, hp=hp))
                    
        elif self.opt.task == 'ddd':
#             num_classes = len(results_pre)
            num_classes = 1
            for j in range(1, num_classes + 1):
                for i in range(len(results_pre[j])):
                    det = results_pre[j][i]
                    score = det[-1]
                    if score <= self.opt.out_thresh:
                        continue
#                     bbox = det[1:5]
                    left, top, right, bottom = det[1:5]
                    if left > right:
                        left, right = right, left
                    bbox = [left, top, right, bottom]
                    
                    if score > self.opt.track_thresh:
                        detections.append(STrack(STrack.tlbr_to_tlwh(bbox), score, backup=det))
                    else:
                        detections_second.append(STrack(STrack.tlbr_to_tlwh(bbox), score, backup=det))
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

        ''' Step 2: First association, with Kalman and IOU, optionall with mask, pose, depth'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)

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

        ''' Step 3: Second association, association the untrack to the low score detections，with IOU'''
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            #track = strack_pool[it]
            track = r_tracked_stracks[it]
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
            if track.score <= self.opt.new_thresh:
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

