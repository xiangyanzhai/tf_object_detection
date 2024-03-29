# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from object_detection.tool import vgg
from object_detection.tool.ROIAlign import roi_align
from object_detection.tool.tf_ATC_test import AnchorTargetCreator
from object_detection.tool.tf_PC_test import ProposalCreator
from object_detection.tool.tf_PTC_test import ProposalTargetCreator
from object_detection.tool.read_Data import readData
from object_detection.tool.get_anchors import get_anchors




class Config():
    def __init__(self, is_train, Mean, files, lr=1e-3, weight_decay=0.0005,
                 num_cls=20, img_max=1000,
                 img_min=600, anchor_scales=[128, 256, 512], anchor_ratios=[0.5, 1, 2],
                 batch_size=1, gpus=1,
                 rpn_n_sample=256,
                 rpn_pos_iou_thresh=0.7, rpn_neg_iou_thresh=0.3,
                 rpn_pos_ratio=0.5,
                 roi_nms_thresh=0.7,
                 roi_train_pre_nms=12000,
                 roi_train_post_nms=2000,
                 roi_test_pre_nms=6000,
                 roi_test_post_nms=300,
                 roi_min_size=16,
                 fast_n_sample=128,
                 fast_pos_ratio=0.25, fast_pos_iou_thresh=0.5,
                 fast_neg_iou_thresh_hi=0.5, fast_neg_iou_thresh_lo=0.0,

                 ):
        self.is_train = is_train
        self.Mean = Mean
        self.files = files
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cls = num_cls
        self.img_max = img_max
        self.img_min = img_min
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.batch_size = batch_size
        self.gpus = gpus
        self.rpn_n_sample = rpn_n_sample
        self.rpn_pos_iou_thresh = rpn_pos_iou_thresh
        self.rpn_neg_iou_thresh = rpn_neg_iou_thresh
        self.rpn_pos_ratio = rpn_pos_ratio
        self.roi_nms_thresh = roi_nms_thresh
        self.roi_train_pre_nms = roi_train_pre_nms
        self.roi_train_post_nms = roi_train_post_nms
        self.roi_test_pre_nms = roi_test_pre_nms
        self.roi_test_post_nms = roi_test_post_nms
        self.roi_min_size = roi_min_size
        self.fast_n_sample = fast_n_sample
        self.fast_pos_ratio = fast_pos_ratio
        self.fast_pos_iou_thresh = fast_pos_iou_thresh
        self.fast_neg_iou_thresh_hi = fast_neg_iou_thresh_hi
        self.fast_neg_iou_thresh_lo = fast_neg_iou_thresh_lo

        print('vgg16')
        print('==============================================================')
        print('Mean:\t', self.Mean)
        print('files:\t', self.files)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('num_cls:\t', self.num_cls)
        print('img_max:\t', self.img_max)
        print('img_min:\t', self.img_min)
        print('anchor_scales:\t', self.anchor_scales)
        print('anchor_ratios:\t', self.anchor_ratios)
        print('batch_size_per_GPU:\t', self.batch_size)
        print('gpus:\t', self.gpus)
        print('==============================================================')
        print('rpn_n_sample:\t', self.rpn_n_sample)
        print('rpn_pos_iou_thresh:\t', self.rpn_pos_iou_thresh)
        print('rpn_neg_iou_thresh:\t', self.rpn_neg_iou_thresh)
        print('rpn_pos_ratio:\t', self.rpn_pos_ratio)
        print('==============================================================')
        print('roi_nms_thresh:\t', self.roi_nms_thresh)
        print('roi_train_pre_nms:\t', self.roi_train_pre_nms)
        print('roi_train_post_nms:\t', self.roi_train_post_nms)
        print('roi_test_pre_nms:\t', self.roi_test_pre_nms)
        print('roi_test_post_nms:\t', self.roi_test_post_nms)
        print('roi_min_size :\t', self.roi_min_size)
        print('==============================================================')
        print('fast_n_sample :\t', self.fast_n_sample)
        print('fast_pos_ratio :\t', self.fast_pos_ratio)
        print('fast_pos_iou_thresh :\t', self.fast_pos_iou_thresh)
        print('fast_neg_iou_thresh_hi :\t', self.fast_neg_iou_thresh_hi)
        print('fast_neg_iou_thresh_lo  :\t', self.fast_neg_iou_thresh_lo)
        print('==============================================================')

        pass


def abs_smooth(net_loc, input_loc):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    x = net_loc - input_loc
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def softmaxloss(score, label):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=label))
    return loss


def SmoothL1Loss(net_loc, input_loc, sigma, num):
    net_loc_yh = tf.abs(net_loc - input_loc)
    part1 = tf.boolean_mask(net_loc_yh, net_loc_yh < 1)
    part2 = tf.boolean_mask(net_loc_yh, net_loc_yh >= 1)
    loss = (tf.reduce_sum((part1 * sigma) ** 2 * 0.5) + tf.reduce_sum(part2 - 0.5 / sigma ** 2)) / (num + 1e-10)
    return loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


class Faster_rcnn16():
    def __init__(self, config):
        self.config = config
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16), self.config.anchor_scales,
                                   self.config.anchor_ratios)

        self.ATC = AnchorTargetCreator(n_sample=config.rpn_n_sample, pos_iou_thresh=config.rpn_pos_iou_thresh,
                                       neg_iou_thresh=config.rpn_neg_iou_thresh, pos_ratio=config.rpn_pos_ratio)
        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)
        self.PTC = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                         pos_ratio=config.fast_pos_ratio, pos_iou_thresh=config.fast_pos_iou_thresh,
                                         neg_iou_thresh_hi=config.fast_neg_iou_thresh_hi,
                                         neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)

    def handle_im(self, im, bboxes):
        H = tf.shape(im)[1]
        W = tf.shape(im)[2]
        H = tf.to_float(H)
        W = tf.to_float(W)
        ma = tf.reduce_max([H, W])
        mi = tf.reduce_min([H, W])
        scale = tf.reduce_min([self.config.img_max / ma, self.config.img_min / mi])
        nh = H * scale
        nw = W * scale
        nh = tf.to_int32(nh)
        nw = tf.to_int32(nw)
        im = tf.image.resize_images(im, (nh, nw))

        bboxes = tf.concat([bboxes[..., :4] * scale, tf.gather(bboxes, [4], axis=-1)], axis=-1)
        return im, bboxes, nh, nw

    def rpn_net(self, net):
        with tf.variable_scope('rpn'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                net_rpn = slim.conv2d(net, 512, [3, 3], scope='conv')
                net_score = slim.conv2d(net_rpn, self.num_anchor * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc = slim.conv2d(net_rpn, self.num_anchor * 4, [1, 1], activation_fn=None, scope='box')
        m = tf.shape(net)[0]
        net_score = tf.reshape(net_score, [m, -1, 2])
        net_loc = tf.reshape(net_loc, [m, -1, 4])
        return net_score, net_loc

    def fast_train_data(self, loc, score, anchor, img_size, bboxes):
        roi = self.PC(loc, score, anchor, img_size)
        roi, loc, label = self.PTC(roi, bboxes[:, :4], tf.to_int32(bboxes[:, -1]))
        roi_inds = tf.zeros(tf.shape(roi)[0], dtype=tf.int32)
        return roi, roi_inds, loc, label

    def pooling(self, net, roi, roi_inds, img_H, img_W, map_H, map_W):
        img_H = tf.to_float(img_H)
        img_W = tf.to_float(img_W)

        map_H = tf.to_float(map_H)
        map_W = tf.to_float(map_W)
        roi_norm = roi / tf.concat([[img_H], [img_W], [img_H], [img_W]], axis=0) * tf.concat(
            [[map_H], [map_W], [map_H], [map_W]], axis=0)
        roi_norm = tf.stop_gradient(roi_norm)
        net_fast = roi_align(net, roi_norm, roi_inds, 7)
        return net_fast

    def fast_net(self, net_fast):
        with tf.variable_scope('vgg_16', reuse=True):
            net_fast = slim.conv2d(net_fast, 4096, [7, 7], padding='VALID', scope='fc6')

            net_fast = slim.dropout(net_fast, 0.5, is_training=False,
                                    scope='dropout6')
            net_fast = slim.conv2d(net_fast, 4096, [1, 1], scope='fc7')
            net_fast = slim.dropout(net_fast, 0.5, is_training=False,
                                    scope='dropout7')
            net_fast = tf.squeeze(net_fast, [1, 2])
        with tf.variable_scope('fast') as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                net_m_score = slim.fully_connected(net_fast, self.config.num_cls + 1, activation_fn=None,
                                                   scope='cls',
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                net_m_loc = slim.fully_connected(net_fast, (self.config.num_cls + 1) * 4, activation_fn=None,
                                                 scope='box',
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.001))

        net_m_loc = tf.reshape(net_m_loc, (-1, self.config.num_cls + 1, 4))
        return net_m_score, net_m_loc

    def build_net(self, Iter):
        im, bboxes, nums = Iter.get_next()
        im, bboxes, img_H, img_W = self.handle_im(im, bboxes)
        im = im - self.Mean
        map_H = tf.to_int32(tf.to_float(img_H) / 16)
        map_W = tf.to_int32(tf.to_float(img_W) / 16)
        with slim.arg_scope(vgg.vgg_arg_scope()):
            outputs, end_points = vgg.vgg_16(im, num_classes=None)
        # var12 = tf.trainable_variables('vgg_16/conv1') + tf.trainable_variables('vgg_16/conv2')
        var_pre = tf.global_variables()[1:]
        net = end_points['vgg_16/conv5/conv5_3']
        rpn_net_score, rpn_net_loc = self.rpn_net(net)

        map_id = 0
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = tf.reshape(tanchors, (-1, 4))
        inds, label, indsP, loc = self.ATC(bboxes[map_id], tanchors, (img_H, img_W))
        net_score_train = tf.gather(rpn_net_score[map_id], inds)
        net_loc_train = tf.gather(rpn_net_loc[map_id], indsP)
        rpn_cls_loss = softmaxloss(net_score_train, label)
        rpn_box_loss = SmoothL1Loss(net_loc_train, loc, 3.0, 240)
        rpn_loss = rpn_cls_loss + rpn_box_loss

        roi, roi_inds, loc, label = self.fast_train_data(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1],
                                                         tanchors, (img_H, img_W), bboxes[0])
        roi = tf.stop_gradient(roi)
        roi_inds = tf.stop_gradient(roi_inds)
        loc = tf.stop_gradient(loc)
        label = tf.stop_gradient(label)
        print(roi, roi_inds)
        net_fast = self.pooling(net, roi, roi_inds, img_H, img_W, map_H, map_W)
        net_m_score, net_m_loc = self.fast_net(net_fast)

        fast_num = tf.shape(roi)[0]
        inds_a = tf.where(label > 0)[:, 0]
        self.fast_num_P = tf.shape(inds_a)[0]
        inds_b = tf.gather(label, inds_a)
        loc = tf.gather(loc, inds_a)
        inds_a = tf.to_int32(inds_a)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        net_m_loc_train = tf.gather_nd(net_m_loc, inds)

        fast_cls_loss = softmaxloss(net_m_score, label)

        fast_box_loss = SmoothL1Loss(net_m_loc_train, loc, 1.0, tf.to_float(fast_num))
        fast_loss = fast_cls_loss + fast_box_loss

        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        self.c = fast_cls_loss
        self.d = fast_box_loss
        self.fast_num = fast_num
        loss = rpn_loss + fast_loss
        return loss, var_pre

    def train(self):

        num_shards = self.config.gpus * self.config.batch_size
        base_lr = self.config.lr * num_shards
        print('*****************', num_shards, base_lr)
        steps = tf.Variable(0.0, name='steps_Faster_rcnn', trainable=False)
        lr = tf.case({steps < 60000.0: lambda: base_lr, steps < 80000.0: lambda: base_lr / 10},
                     default=lambda: base_lr / 100)
        tower_grads = []
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        var_reuse = False
        Iter_list = []

        c = 0
        for i in range(self.config.gpus):
            with tf.device('/gpu:%d' % i):
                loss = 0
                for j in range(self.config.batch_size):
                    Iter = readData(self.config.files, batch_size=1, num_threads=16, num_shards=num_shards,
                                    shard_index=c)
                    Iter_list.append(Iter)
                    with tf.variable_scope('', reuse=var_reuse):
                        if c == 0:
                            pre_loss, var_pre = self.build_net(Iter)
                        else:
                            pre_loss, _ = self.build_net(Iter)
                        var_reuse = True
                        loss += pre_loss
                    c += 1
                loss = loss / self.config.batch_size
                train_vars = [v for v in tf.trainable_variables() if ('conv1' not in v.name and 'conv2' not in v.name)]
                l2_reg = tf.losses.get_regularization_losses()
                l2_reg = [l2 for l2 in l2_reg if ('conv1' not in l2.name and 'conv2' not in l2.name)]
                l2_re_loss = tf.add_n(l2_reg)
                faster_loss = loss + l2_re_loss
                grads_and_vars = opt.compute_gradients(faster_loss, train_vars)
                tower_grads.append(grads_and_vars)
        for v in tf.global_variables():
            print(v)
        grads = average_gradients(tower_grads)
        grads = list(zip(*grads))[0]
        grads, norm = tf.clip_by_global_norm(grads, 10.0)
        train_op = opt.apply_gradients(zip(grads, train_vars), global_step=steps)

        pre_file = '/home/zhai/PycharmProjects/Demo35/tensorflow_zoo/vgg_16.ckpt'
        saver_pre = tf.train.Saver(var_pre)

        saver = tf.train.Saver(max_to_keep=200)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
       
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess,file)
            saver_pre.restore(sess, pre_file)
            for Iter in Iter_list:
                sess.run(Iter.initializer)

            for i in range(0000, 90010):
                if i % 20 == 0:
                    _, loss, a, b, c, d, e, f, h, hh, = sess.run(
                        [train_op, faster_loss, self.a, self.b, self.c, self.d, l2_re_loss,
                         norm, self.fast_num, self.fast_num_P, ])
                    print(datetime.now(), 'loss:%.4f' % loss, 'rpn_cls_loss:%.4f' % a, 'rpn_box_loss:%.4f' % b,
                          'fast_cls_loss:%.4f' % c, 'fast_box_loss:%.4f' % d, 'l2_re_loss:%.4f' % e, 'norm:%.4f' % f, h,
                          hh, i)
                else:
                    sess.run(train_op)
                if (i + 1) % 5000 == 0 or ((i + 1) % 1000 == 0 and i < 10000):
                    saver.save(sess, os.path.join('./models/', 'Faster_Rcnn_vgg16.ckpt'), global_step=i + 1)

            pass

    pass


import cv2


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']
    config = Config(True, Mean, files, lr=0.001)

    faster_rcnn = Faster_rcnn16(config)
    faster_rcnn.train()

    pass
