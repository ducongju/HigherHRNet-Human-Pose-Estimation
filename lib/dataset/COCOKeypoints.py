# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import pycocotools
from .COCODataset import CocoDataset
from .target_generators import HeatmapGenerator


logger = logging.getLogger(__name__)


class CocoKeypoints(CocoDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 joints_generator,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT)

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 18, 'Number of joint with center for COCO is 18'
        else:
            assert cfg.DATASET.NUM_JOINTS == 17, 'Number of joint for COCO is 17'

        self.num_scales = self._init_check(heatmap_generator, joints_generator)

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.scale_aware_sigma = cfg.DATASET.SCALE_AWARE_SIGMA
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.int_sigma = cfg.DATASET.INT_SIGMA

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        # TODO(bowen): to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        joints = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list

    def get_joints(self, anno):
        num_people = len(anno)

        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))  # 对于每个人体的每个关节赋予不同的sigma值
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])  # 将一维列表转换为二维列表
            # HigherHRNet没有用上centermap
            if self.with_center:
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
            # 设置人体之间的尺度感知sigma参数, 而人体内部没有尺度感知
            # if self.scale_aware_sigma:
            #     # get person box
            #     box = obj['bbox']
            #     size = max(box[2], box[3])  # sigma大小以人体包围框的长边作为参考, 256时为2
            #     sigma = size / self.base_size * self.base_sigma  # base_size = 256, base_sigma = 2.0
            #     if self.int_sigma:
            #         sigma = int(np.round(sigma + 0.5))  # 对sigma取整
            #     assert sigma > 0, sigma
            #     joints[i, :, 3] = sigma  # 为某一个人的不同关节设置相同的值
            ###########################  人体外部尺度  ################################
            if self.scale_aware_sigma:
            # 人体外部尺度
                box = obj['bbox']
                intersize = max(box[2], box[3])
                base_intersize = 128
                base_intersigma = 2
                # 线性变化
                intersigma = intersize / base_intersize * base_intersigma
                # 非线性变化
                x = intersize / base_intersize
                intersigma = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) * base_intersigma

            # 人体内部尺度
                # 非截断设置
                intrasize = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072,
                             .062, .062, .107, .107, .087, .087, .089, .089])
                # 截断设置
                intrasize = np.array([.062, .062, .062, .062, .062, .079, .079, .072, .072,
                             .062, .062, .107, .107, .087, .087, .089, .089])
                base_intrasize = 0.062
                base_intrasigma = 2
                intrasigma = intrasize / base_intrasize * base_intrasigma

            # 人体综合尺度
                joints[i, :, 3] = intersigma * intrasigma
            ###########################  人体内部尺度  ################################

        return joints

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5

    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(joints_generator)
            )
        return len(heatmap_generator)
