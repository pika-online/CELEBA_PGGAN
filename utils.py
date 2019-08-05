"""
script: utils
Author:Ephemeroptera
date:2019-8-1
mail:605686962@qq.com

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os
#--------------------------------------------------- format ------------------------------------------------------------
def UINT8(float):
    return np.uint8(float*255)

#--------------------------------------------------- PLT ------------------------------------------------------------
def PLT_PLOT(x):
    plt.plot(x)
    plt.show()
#------------------------------------------------------ CV -------------------------------------------------------------
# 关闭全部窗口
def CV2_ALL_CLOSE():
    cv2.destroyAllWindows()

# 显示单张
def CV2_IMSHOW_HWC(img, title='img',delay=0):
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(delay)

# 显示多张
# define batch showing using CV2
def CV2_IMSHOW_NHWC(NHWC, scale, rows, cols, title='img', delay = 0):
    NHWC = np.array(NHWC)
    assert len(NHWC.shape) == 4 # [N,H,W,C]
    N = NHWC.shape[0]
    assert N <= rows*cols
    H = NHWC.shape[1]
    W = NHWC.shape[2]
    C = NHWC.shape[3]
    # get resized img shape
    S = cv2.resize(NHWC[0], (0, 0), fx=scale, fy=scale).shape
    # build img sets
    IMG = np.zeros(shape=[rows*S[0],cols*S[1],C], dtype=NHWC.dtype)
    # assign
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx == N:
                break
            IMG[r*S[0]:(r+1)*S[0],c*S[1]:(c+1)*S[1],:] = cv2.resize(NHWC[idx], (0, 0), fx=scale, fy=scale)
            idx += 1
    # show
    CV2_IMSHOW_HWC(IMG,title,delay)

# 随机显示N张
def CV2_IMSHOW_NHWC_RAMDOM(NHWC, scale, N, rows, cols,title='img', delay = 0):
    NHWC = np.array(NHWC)
    # randomly sampling
    total = NHWC.shape[0]
    idxes = list(range(total))
    idx = random.sample(idxes, N)
    rands = NHWC[idx]
    CV2_IMSHOW_NHWC(rands, scale, rows, cols, title, delay)

#------------------------------------------------------ hwc ------------------------------------------------------------

# 单个图片上采样
def upsampling_hwc(hwc):
    return cv2.resize(hwc, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# 单个图片任意上采样
def upsize_hwc(hwc,res):
    return cv2.resize(hwc,res,interpolation=cv2.INTER_LINEAR)

# 拉普拉斯上采样
def pyrup_hwc(hwc):
    return cv2.pyrUp(hwc)

# 单个图片下采样
def downsampling_hwc(hwc):
    return cv2.resize(hwc, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 单个图片任意下采样
def downsize_hwc(hwc,res):
    return cv2.resize(hwc,res,interpolation=cv2.INTER_AREA)

# 拉普拉斯下采样
def pyrdown_hwc(hwc):
    return cv2.pyrDown(hwc)

# 低通滤波
def lpf_hwc(hwc):
    return pyrup_hwc(pyrdown_hwc(hwc))

# 高通滤波
def hpf_hwc(hwc):
    return np.clip(hwc -lpf_hwc(hwc),-1.0,1.0 )
    # return hwc - lpf_hwc(hwc)

# 拉普拉斯金字塔(N层)
def lap_pyd_hwc(hwc,N):
    pyd = {}
    for i in range(N):
        res = int(hwc.shape[0]) # 当前分辨率
        if i != N - 1:
            pyd[str(res)] = hpf_hwc(hwc)
            hwc = pyrdown_hwc(hwc)
        else: # 最底层
            pyd[str(res)] = hwc
        print('正在构建第%d层..' % (i + 1))
    return pyd

# 复原（单个样本）
def recover_from_lap_pyd_hwc(pyd):
    # 不同尺寸排序
    res = [int(key) for key in pyd.keys()]
    res.sort()
    # 累加
    hwc = pyd[str(res[0])]
    for r in res[1:]:
        hwc = pyd[str(r)] + pyrup_hwc(hwc)
        # print('正在复原分辨率%d'%r)
    return hwc

#----------------------------------------------------- nhwc ------------------------------------------------------------

# 批图片上采样
def upsampling_nhwc(nhwc):
    us = []
    for hwc in nhwc:
       us.append(upsampling_hwc(hwc))
    return np.array(us)

# 批图片任意上采样
def upsize_nhwc(nhwc,res):
    us = []
    for hwc in nhwc:
       us.append(upsize_hwc(hwc,res))
    return np.array(us)

# 拉普拉斯上采样
def pyrup_nhwc(nhwc):
    us = []
    for hwc in nhwc:
       us.append(pyrup_hwc(hwc))
    return np.array(us)

# 批图片下采样
def downsampling_nhwc(nhwc):
    ds = []
    for hwc in nhwc:
       ds.append(downsampling_hwc(hwc))
    return np.array(ds)

# 批图片任意下采样
def downsize_nhwc(nhwc,res):
    ds = []
    for hwc in nhwc:
       ds.append(downsize_hwc(hwc,res))
    return np.array(ds)

# 拉普拉斯下采样
def pyrdown_nhwc(nhwc):
    ds = []
    for hwc in nhwc:
       ds.append(pyrdown_hwc(hwc))
    return np.array(ds)

# 低通滤波
def lpf_nhwc(nhwc):
    nhwc2 = np.zeros(shape=nhwc.shape,dtype=nhwc.dtype)
    for idx in range(int(nhwc.shape[0])):
        nhwc2[idx] = lpf_hwc(nhwc[idx])
    return nhwc2

# 高通滤波
def hpf_nhwc(nhwc):
    nhwc2 = np.zeros(shape=nhwc.shape, dtype=nhwc.dtype)
    for idx in range(int(nhwc.shape[0])):
        nhwc2[idx] = hpf_hwc(nhwc[idx])
    return nhwc2

# 拉普拉斯金字塔(N层)
def lap_pyd_nhwc(nhwc,N):
    pyd = {}
    for i in range(N):
        res = int(nhwc.shape[1]) # 当前分辨率
        if i != N - 1:
            pyd[str(res)] = hpf_nhwc(nhwc)
            nhwc = pyrdown_nhwc(nhwc)
        else: # 最底层
            pyd[str(res)] = nhwc
        print('正在构建第%d层..' % (i + 1))
    return pyd


#----------------------------------------------------- others ----------------------------------------------------------
# 保存数据
def PICKLE_SAVING(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        file.close()
        print('Saving to %s successfully' % path)

# 加载数据
def PICKLE_LOADING(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        file.close()
    return data

# 创建文件夹
def MKDIR(dir):
    if not os.path.isdir(dir):
        print('DIR:%s NOT FOUND，BUILDING ON CURRENT PATH..'%dir)
        os.mkdir(dir)



"""
test
"""
if __name__ == '__main__':

    # 选择图片
    # img = UINT8(cv2.imread(r'./man.jpg')/255)
    img = cv2.imread(r'./man.jpg')/255
    img = cv2.resize(img,(256,512))
    CV2_IMSHOW_HWC(img)
    img = np.array([img])
    CV2_IMSHOW_NHWC(img,0.5,1,1,'img',0)

    # 缩小和放大
    ds = downsampling_nhwc(img)
    us = upsampling_nhwc(img)
    CV2_IMSHOW_NHWC(ds,2,1,1,'ds',0)
    CV2_IMSHOW_NHWC(us, 2, 1, 1, 'us', 0)

    # 拉式金字塔
    pyd = lap_pyd_nhwc(img,5)
    for laps in pyd.values():
        CV2_IMSHOW_NHWC(laps,1,1,1,'laps',0)

    # 复原
    # rec = recover_from_lap_pyd_hwc(pyd)
    # CV2_IMSHOW_HWC(rec,'rec')
