# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import scipy.ndimage
import visualization as vs

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0] # 128*16384
    H = nhood_size // 2# 1
    # nhood:0~16383*128
    # chan:0,1,2
    # x:-1:1
    # y:-1:1
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    # 对N个图像随机采样128个descriptor
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

#----------------------------------------------------------------------------

class API:
    def __init__(self, num_images, image_shape, image_dtype, minibatch_size):
        self.nhood_size         = 7
        self.nhoods_per_image   = 128
        self.dir_repeats        = 4
        self.dirs_per_repeat    = 128
        self.resolutions = []
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.descriptors = [[] for res in self.resolutions]

    def feed(self, mode, minibatch):
        for lod, level in enumerate(generate_laplacian_pyramid(minibatch, len(self.resolutions))):
            desc = get_descriptors_for_minibatch(level, self.nhood_size, self.nhoods_per_image)
            self.descriptors[lod].append(desc)

    def end(self, mode):
        desc = [finalize_descriptors(d) for d in self.descriptors]
        del self.descriptors
        if mode in ['warmup', 'reals']:
            self.desc_real = desc
        dist = [sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat) for dreal, dfake in zip(self.desc_real, desc)]
        del desc
        dist = [d * 1e3 for d in dist] # multiply by 10^3
        return dist + [np.mean(dist)]

#----------------------------------------------------------------------------
# 测试
if __name__ == '__main__':

    import CelebA_MultiRes_Preprocess as pp

    # 指定路径
    celeba_path = 'I:\CELEBA\img_align_celeba'
    attr_txt_path = r"I:\CELEBA\list_attr_celeba.txt"

    # 获取数据集
    batch = pp.get_specfic_data(celeba_path,attr_txt_path,16,1)
    batch = batch[0:8192]  # 2^13
    batch = (batch).transpose([0,3,1,2])


    PYD = generate_laplacian_pyramid(batch,4)
    DESC = {}
    for pyd in PYD:
        res = pyd.shape[2]
        # vs.CV2_BATCH_RANDOM_SHOW(pyd.transpose([0,2,3,1]), 1, 25, 5, 5, 0)
        desc = get_descriptors_for_minibatch(pyd,7,128)
        desc = finalize_descriptors(desc)
        DESC[str(res)] = desc
        print('成功提取%dx%d特征..'%(res,res))
    del PYD

    # 保存
    # import pickle
    # with open('./DESC.des', 'wb') as d:
    #     pickle.dump(DESC, d)
    #     d.close()


    # 计算swd
    h = batch - pyr_up(pyr_down(batch))
    desc = get_descriptors_for_minibatch(h,7,128)
    desc = finalize_descriptors(desc)
    swd = sliced_wasserstein(desc,DESC['128'],4,128)*1e3
    print(swd)



