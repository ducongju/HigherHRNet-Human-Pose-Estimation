# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        ############################################################################
        import numpy as np
        import scipy
        import scipy.io as io
        from scipy.ndimage.filters import gaussian_filter
        import os
        import glob
        from matplotlib import pyplot as plt
        import h5py
        import PIL.Image as Image
        from matplotlib import cm as CM

        # partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
        def gaussian_filter_density(img, points):
            '''
            This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
            points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
            img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
            return:
            density: the density-map we want. Same shape as input image but only has one channel.
            example:
            points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
            img_shape: (768,1024) 768 is row and 1024 is column.
            '''
            img_shape = [img.shape[0], img.shape[1]]
            print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
            density = np.zeros(img_shape, dtype=np.float32)
            gt_count = len(points)
            if gt_count == 0:
                return density

            leafsize = 2048
            # build kdtree
            tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
            # query kdtree
            distances, locations = tree.query(points, k=4)

            print('generate density...')
            for i, pt in enumerate(points):
                pt2d = np.zeros(img_shape, dtype=np.float32)
                if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                    pt2d[int(pt[1]), int(pt[0])] = 1.
                else:
                    continue
                if gt_count > 1:
                    sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
                else:
                    sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
                density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
            print('done.')
            return density

        # # test code
        # if __name__ == "__main__":
        #     # show an example to use function generate_density_map_with_fixed_kernel.
        #     root = 'D:\\workspaceMaZhenwei\\GithubProject\\Crowd_counting_from_scratch\\data'
        #
        #     # now generate the ShanghaiA's ground truth
        #     part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
        #     part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
        #     # part_B_train = os.path.join(root,'part_B_final/train_data','images')
        #     # part_B_test = os.path.join(root,'part_B_final/test_data','images')
        #     path_sets = [part_A_train, part_A_test]
        #
        #     img_paths = []
        #     for path in path_sets:
        #         for img_path in glob.glob(os.path.join(path, '*.jpg')):
        #             img_paths.append(img_path)
        #
        #     for img_path in img_paths:
        #         print(img_path)
        #         mat = io.loadmat(
        #             img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        #         img = plt.imread(img_path)  # 768行*1024列
        #         k = np.zeros((img.shape[0], img.shape[1]))
        #         points = mat["image_info"][0, 0][0, 0][0]  # 1546person*2(col,row)
        #         k = gaussian_filter_density(img, points)
        #         # plt.imshow(k,cmap=CM.jet)
        #         # save density_map to disk
        #         np.save(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth'), k)
        #
        #     '''
        #     #now see a sample from ShanghaiA
        #     plt.imshow(Image.open(img_paths[0]))
        #
        #     gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
        #     plt.imshow(gt_file,cmap=CM.jet)
        #
        #     print(np.sum(gt_file))# don't mind this slight variation
        #     '''
            ####################################################################################

        for p in joints:

            for idx, pt in enumerate(p):
                ############人体内部尺度##############
                if idx == 0:
                    sigma = 2
                elif idx == 1:
                    sigma = 2
                elif idx == 2:
                    sigma = 2
                #####################################
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms


class JointsGenerator():
    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))
        output_res = self.output_res
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 \
                   and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = \
                            (idx * output_res**2 + y * output_res + x, 1)
                    else:
                        visible_nodes[i][tot] = \
                            (y * output_res + x, 1)
                    tot += 1
        return visible_nodes
