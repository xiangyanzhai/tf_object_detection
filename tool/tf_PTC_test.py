# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = tf.reduce_prod(hw, axis=-1)

    hw = bboxes[:, 2:4] - bboxes[:, :2]
    areas2 = tf.reduce_prod(hw, axis=-1)

    yx1 = tf.maximum(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = tf.minimum(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1
    hw = tf.maximum(hw, 0)
    areas_i = tf.reduce_prod(hw,axis=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou
# def cal_IOU(pre_bboxes, bboxes):
#     # bboxes anchors
#     # r*b*5 a*4
#     x1, y1, x2, y2 = tf.split(pre_bboxes, 4, axis=-1)
#
#     w = x2 - x1
#     h = y2 - y1
#     areas1 = w * h
#
#     xx1 = bboxes[..., 0]
#     yy1 = bboxes[..., 1]
#     xx2 = bboxes[..., 2]
#     yy2 = bboxes[..., 3]
#
#     w = xx2 - xx1
#     h = yy2 - yy1
#     areas2 = w * h
#
#     x1 = tf.maximum(x1, xx1)
#     y1 = tf.maximum(y1, yy1)
#     x2 = tf.minimum(x2, xx2)
#     y2 = tf.minimum(y2, yy2)
#
#     w = tf.maximum(0.0, x2 - x1)
#     h = tf.maximum(0.0, y2 - y1)
#     inter = w * h
#     iou = inter / (areas1 + areas2 - inter)
#     return iou

def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = tf.log(hw / c_hw)
    return tf.concat([t_yx, t_hw], axis=1)
class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = tf.concat([roi, bbox], axis=0)
        IOU = cal_IOU(roi, bbox)

        inds_box = tf.argmax(IOU, axis=1, output_type=tf.int32)
        inds = tf.range(tf.shape(roi)[0])
        inds = tf.concat([tf.reshape(inds, (-1, 1)), tf.reshape(inds_box, (-1, 1))],axis=1)
        iou = tf.gather_nd(IOU, inds)
        indsP = iou >= self.pos_iou_thresh
        indsN = (iou >= self.neg_iou_thresh_lo) & (iou < self.pos_iou_thresh)

        indsP = tf.where(indsP)[:, 0]
        indsN = tf.where(indsN)[:, 0]
        indsP = tf.random_shuffle(indsP)
        indsN = tf.random_shuffle(indsN)

        n_pos = tf.reduce_min([tf.to_int32(self.n_sample * self.pos_ratio), tf.shape(indsP)[0]])

        indsP = indsP[:n_pos]
        indsN = indsN[:self.n_sample - n_pos]

        n_neg=tf.shape(indsN)[0]

        roiP=tf.gather(roi,indsP)
        roiN=tf.gather(roi,indsN)

        inds_box=tf.gather(inds_box,indsP)
        loc=bbox2loc(roiP,tf.gather(bbox,inds_box))
        loc=(loc-tf.constant(loc_normalize_mean,dtype=tf.float32))/tf.constant(loc_normalize_std,dtype=tf.float32)
        label=tf.gather(label,inds_box)+1

        roi=tf.concat([roiP,roiN],axis=0)
        label=tf.concat([label,tf.zeros(n_neg,dtype=tf.int32)],axis=0)
        return roi,loc,label
