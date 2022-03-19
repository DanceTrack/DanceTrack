"""
    Script to calculate the average IoU of the same obejct on consecutive 
    frames, and the relative switch frequency (Figure3(b) and Figure3(c)).
    The original data in paper is calculated on all sets: train+val+test.
    On the train-set:
        * Average IoU on consecutive frames = 0.894
        * Relative Position Switch frequency = 0.031
    On the val-set:
        * Average IoU on consecutive frames = 0.909
        * Relative Position Switch frequency = 0.030
    The splitting of subsets is
"""
import numpy as np 
import os 

source_dir = "train"
# source_dir = "val"

def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height

def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union

def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou


def consecutive_iou(annos):
    """
        calculate the IoU over bboxes on the consecutive frames
    """
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou = 0 
    total_frequency = 0
    for find in range(min_frame, max_frame):
        anno_cur = annos[np.where(annos[:,0]==find)]
        anno_next = annos[np.where(annos[:,0]==find+1)]
        ids_cur = np.unique(anno_cur[:,1])
        ids_next = np.unique(anno_next[:,1])
        common_ids = np.intersect1d(ids_cur, ids_next)
        for tid in common_ids:
            cur_box = anno_cur[np.where(anno_cur[:,1]==tid)][:, 2:6]
            next_box = anno_next[np.where(anno_next[:,1]==tid)][:, 2:6]
            cur_box[:, 2:] += cur_box[:, :2]
            next_box[:, 2:] += next_box[:, :2]
            iou = box_iou(cur_box, next_box).item()
            total_iou += iou 
            total_frequency += 1
    return total_iou, total_frequency


def center(box):
    return (box[0]+0.5*box[2], box[1]+0.5*box[3])

def relative_switch(annos):
    """
        calculate the frequency of relative position switch regarding center location
    """
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    switch = 0
    sw_freq = 0
    for find in range(min_frame, max_frame):
        anno_cur = annos[np.where(annos[:,0]==find)]
        anno_next = annos[np.where(annos[:,0]==find+1)]
        ids_cur = np.unique(anno_cur[:,1])
        ids_next = np.unique(anno_next[:,1])
        common_ids = np.intersect1d(ids_cur, ids_next)
        for id1 in common_ids:
            for id2 in common_ids:
                sw_freq += 1
                if id1 == id2:
                    continue 
                box_cur_1 = anno_cur[np.where(anno_cur[:,1]==id1)][0][2:6]
                box_cur_2 = anno_cur[np.where(anno_cur[:,1]==id2)][0][2:6]
                box_next_1 = anno_next[np.where(anno_next[:,1]==id1)][0][2:6]
                box_next_2 = anno_next[np.where(anno_next[:,1]==id2)][0][2:6]
                left_right_cur = center(box_cur_1)[0] >= center(box_cur_2)[0]
                left_right_next = center(box_next_1)[0] >= center(box_next_2)[0]
                top_down_cur = center(box_cur_1)[1] >= center(box_cur_2)[1]
                top_down_next = center(box_next_1)[1] >= center(box_next_2)[1]
                if (left_right_cur != left_right_next) or (top_down_cur != top_down_next):
                    switch += 1
    return switch, sw_freq
           

if __name__ == "__main__":
    seqs = os.listdir(source_dir)
    all_iou, all_freq = 0, 0 
    all_switch, all_sw_freq = 0, 0
    for seq in seqs:
        if seq == ".DS_Store":
            continue
        anno_file = os.path.join(source_dir, seq, "gt/gt.txt")
        annos = np.loadtxt(anno_file, delimiter=",")
        seq_iou, seq_freq = consecutive_iou(annos)
        seq_switch, seq_sw_freq = relative_switch(annos)
        all_iou += seq_iou
        all_freq += seq_freq
        all_switch += seq_switch
        all_sw_freq += seq_sw_freq
    print("Average IoU on consecutive frames = {}".format(all_iou / all_freq))
    print("Relative Position Switch frequency = {}".format(all_switch / all_sw_freq))