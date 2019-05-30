import os
import cv2
import numpy as np
import visualization as vs

# 单个样本下采样
def x_downscale2d(x):
    return cv2.resize(x,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

# 拉普拉斯下采样
def x_pyrDown(x):
    return cv2.pyrDown(x)

# 数据集下采样
def batch_downscale2d(batch):
    ds = []
    for x in batch:
       ds.append(x_downscale2d(x))
    return np.array(ds)

# 数据集拉式下采样
def batch_pyrDown(batch):
    pd = []
    for x in batch:
        pd.append(x_pyrDown(x))
    return np.array(pd)

# 单个样本上采样
def x_upscale2d(x):
    return cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# 拉普拉斯上采样
def x_pyrUp(x):
    return cv2.pyrUp(x)

# 数据集上采样
def batch_upscale2d(batch):
    up = []
    for x in batch:
       up.append(x_upscale2d(x))
    return np.array(up)

# 数据集拉式下采样
def batch_pyrUp(batch):
    pu = []
    for x in batch:
        pu.append(x_pyrUp(x))
    return np.array(pu)

# 单个图像低通滤波
def x_lpf(x):
    return x_pyrUp(x_pyrDown(x))

# 数据集低通滤波：
def batch_lpf(batch):
    batch = batch.copy()
    for idx in range(int(batch.shape[0])):
        batch[idx] = x_lpf(batch[idx])
    return batch

# 单个图像高通滤波
def x_hpf(x):
    return x-x_lpf(x)

# 数据集高通滤波
def batch_hpf(batch):
    batch = batch.copy()
    for idx in range(int(batch.shape[0])):
        batch[idx] = x_hpf(batch[idx])
    return batch

# 构建拉普拉斯金字塔（单个样本）
def x_Lap_Pyd(x,level):
    PYD = {}
    for i in range(level):
        res = int(x.shape[0])
        if i != level-1:
            PYD[str(res)] = x_hpf(x)
            x = x_pyrDown(x)
        else:
            PYD[str(res)] = x
        print('正在构建第%d层..' % (i + 1))
    return PYD

# 构建拉普拉斯金字塔（数据集）
def batch_Lap_Pyd(batch,level):
    PYD = {}
    for i in range(level):
        res = int(batch.shape[1])
        if i != level - 1:
            PYD[str(res)] = batch_hpf(batch)
            batch = batch_pyrDown(batch)
        else:
            PYD[str(res)] = batch
        print('正在构建第%d层..' % (i + 1))
    return PYD

# 复原（单个样本）
def recover_from_lap_pyd(PYD):
    idxs = [int(key) for key in PYD.keys()]
    idxs.sort()
    rec = PYD[str(idxs[0])]
    for idx in idxs[1:]:
        rec = PYD[str(idx)] + x_pyrUp(rec)
        print('正在复原分辨率%d'%idx)
    return rec



if __name__ == '__main__':
    # 选择图片
    img = cv2.imread(r'./man.jpg')
    img = cv2.resize(img,(512,1024))
    vs.CV2_SHOW('',x_hpf(img))

    # 拉普拉斯金字塔
    PYD = x_Lap_Pyd(img,5)
    for laps in PYD.values():
        vs.CV2_SHOW('laps',laps)

    # 复原
    rec = recover_from_lap_pyd(PYD)
    vs.CV2_SHOW('rec',rec)