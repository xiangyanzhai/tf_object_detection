# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
from datetime import datetime
from object_detection.tool import resnet_v1
from object_detection.tool.ROIAlign import roi_align
from object_detection.tool.tf_PC_test import ProposalCreator
from object_detection.tool.get_anchors import get_anchors
from object_detection.tool.faster_predict import predict
from sklearn.externals import joblib



class Config():
    def __init__(self, is_train, Mean, files, lr=1e-3, weight_decay=0.0001,
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
        print('resnet101')
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
        print('batch_size_per_image:\t', self.batch_size)
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


class Faster_rcnn_resnet_101():
    def __init__(self, config):
        self.config = config
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16 + 1), self.config.anchor_scales,
                                   self.config.anchor_ratios)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.argscope = resnet_v1.resnet_arg_scope(weight_decay=config.weight_decay)

    def handle_im(self, im):
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

        return im, nh, nw, scale

    def rpn_net(self, net):
        with tf.variable_scope('rpn'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                net_rpn = slim.conv2d(net, 1024, [3, 3], scope='conv')
                net_score = slim.conv2d(net_rpn, self.num_anchor * 2, [1, 1], activation_fn=None, scope='cls')
                net_t = slim.conv2d(net_rpn, self.num_anchor * 4, [1, 1], activation_fn=None, scope='box')
        m = tf.shape(net)[0]
        net_score = tf.reshape(net_score, [m, -1, 2])
        net_t = tf.reshape(net_t, [m, -1, 4])
        return net_score, net_t

    def roi_layer(self, loc, score, anchor, img_size):
        roi = self.PC(loc, score, anchor, img_size, train=self.config.is_train)
        roi_inds = tf.zeros(tf.shape(roi)[0], dtype=tf.int32)

        return roi, roi_inds

    def pooling(self, net, roi, roi_inds, img_H, img_W, map_H, map_W):
        img_H = tf.to_float(img_H)
        img_W = tf.to_float(img_W)

        map_H = tf.to_float(map_H)
        map_W = tf.to_float(map_W)
        roi_norm = roi / tf.concat([[img_H], [img_W], [img_H], [img_W]], axis=0) * tf.concat(
            [[map_H], [map_W], [map_H], [map_W]], axis=0)
        roi_norm = tf.stop_gradient(roi_norm)
        net_fast = roi_align(net, roi_norm, roi_inds, 14)
        return net_fast

    def fast_net(self, net_fast):
        blocks = [
            resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
        ]
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=self.config.weight_decay)):
            with slim.arg_scope([layers.batch_norm], is_training=False):
                with tf.variable_scope('resnet_v1_101', reuse=True):
                    net_fast = resnet_v1.resnet_utils.stack_blocks_dense(net_fast, blocks)
                    net_fast = tf.reduce_mean(net_fast, [1, 2], name='pool5')
        with tf.variable_scope('fast'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                net_m_score = slim.fully_connected(net_fast, self.config.num_cls + 1, activation_fn=None,
                                                   scope='cls',
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                net_m_t = slim.fully_connected(net_fast, (self.config.num_cls + 1) * 4, activation_fn=None,
                                               scope='box',
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.001))

        net_m_t = tf.reshape(net_m_t, (-1, self.config.num_cls + 1, 4))
        return net_m_score, net_m_t

    def build_net(self):
        self.im_input = tf.placeholder(tf.string, name='input')
        im = tf.image.decode_jpeg(self.im_input, 3)
        im = im[None]
        im, img_H, img_W, self.scale = self.handle_im(im)
        im = im - self.Mean
        im = im[..., ::-1]

        with slim.arg_scope(self.argscope):
            outputs, end_points = resnet_v1.resnet_v1_101(im, is_training=False)

        net = end_points['resnet_v1_101/block3']

        map_H = tf.shape(net)[1]
        map_W = tf.shape(net)[2]
        rpn_net_score, rpn_net_loc = self.rpn_net(net)

        tanchors = self.anchors[:map_H, :map_W]
        tanchors = tf.reshape(tanchors, (-1, 4))

        roi, roi_inds = self.roi_layer(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1],
                                       tanchors, (img_H, img_W))

        net_fast = self.pooling(net, roi, roi_inds, img_H, img_W, map_H, map_W)
        net_m_score, net_m_t = self.fast_net(net_fast)

        net_m_t = tf.reshape(net_m_t, (-1, self.config.num_cls + 1, 4)) * tf.constant([0.1, 0.1, 0.2, 0.2])

        self.result = predict(net_m_t, net_m_score, roi, img_H, img_W)

    def test(self):
        self.build_net()

        # file = '/home/zhai/PycharmProjects/Demo35/my_Faster_tool/train/models/Faster_vgg16.ckpt-89999'
        # file = '/home/zhai/PycharmProjects/Demo35/my_Faster/models/Faster_vgg16.ckpt-89999'
        file = '/home/zhai/PycharmProjects/Demo35/my_Faster_tool/train_5/models/Faster_resnet_101.ckpt-89999'

        file = '../train/models/Faster_Rcnn_resnet_101.ckpt-3000'
        saver = tf.train.Saver()
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            Res = {}
            i = 0
            m = len(names)
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res, s = sess.run([self.result, self.scale], feed_dict={self.im_input: img})
                res[:, :4] = res[:, :4] / s

                Res[name] = res
            print(datetime.now() - time_start)

            joblib.dump(Res, 'Faster_Rcnn_resnet101.pkl')


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

    config = Config(False, Mean, files, lr=0.001)

    faster_rcnn = Faster_rcnn_resnet_101(config)
    faster_rcnn.test()

    pass
