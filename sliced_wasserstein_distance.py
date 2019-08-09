import numpy as np
import CelebA_Specific_Preprocess as CSP
import utils as us

"""
 script：sliced wasserstein distance
 author:Ephemeroptera
 date:2019-6-1
 contact:605686962@qq.com
"""
"""
论文方法：（1）从数据集抽取16384（2^14）张图片，并建立该图片集的拉式金字塔
         （2）从每张图片的每一层提取128（2^7）个descriptors，那么图片集的每一层将提供（2^21）个descriptors.
              其中每个descriptors为7x7x3的patch
          （3）对于每一层全部的descriptors，我们对各个color通道分别进行均值-方差归一化，再reshape成[N,-1]即向量集
          （4）同上，对同样数目的生成集提取descriptors并归一化，在计算两集合之间的sliced wasserstein distance
          
本文方法：计算swd是一个耗时漫长的过程，为了简化计算，我们采样8192（2^13）个样本，每个样本仅提取64个7x7x3 patchs,使用256个投影
         计算swd。
"""

# 数据集指定频带的descriptors
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape
    assert len(S) == 4 and S[3] == 3
    H = nhood_size // 2 # 卷积核半径
    descriptors = []
    for img in minibatch:
        # 对每张图片，随机采样descriptor[X,Y]
        X =  np.random.randint(H, S[2] - H,nhoods_per_image)
        Y =  np.random.randint(H, S[1] - H, nhoods_per_image)
        # 裁取每个patch
        for x,y in zip(X,Y):
            descriptors.append(img[y-H:y+H+1,x-H:x+H+1,:])
    return np.array(descriptors)

# descriptor归一化
def finalize_descriptors(desc):
    # 各通道归一化
    desc -= np.mean(desc, axis=(0, 1, 2), keepdims=True)
    desc /= np.std(desc, axis=(0, 1, 2), keepdims=True)
    # reshape
    desc = desc.reshape(desc.shape[0], -1)
    return desc

# 获取各频带的descriptors
def get_descriptors_for_all_level(PYD):
    DESC = {}
    for level in PYD.values():
        res = level.shape[1]
        desc = get_descriptors_for_minibatch(level, 7, 64)
        desc = finalize_descriptors(desc)
        DESC[str(res)] = desc
        print('已提取%dx%d频带descriptors..' % (res, res))
    return DESC

# 计算sliced wasserstein distance
def sliced_wasserstein_distance(A, B, dir_repeats, dirs_per_repeat):
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
    return np.mean(results,dtype=A.dtype)

if __name__ == '__main__':

    # 指定路径
    celeba_path = 'F:\ww\CELEBA\img_align_celeba'
    attr_txt_path = r"F:\ww\CELEBA\list_attr_celeba.txt"

    # 获取数据集
    batch = CSP.get_data(celeba_path, attr_txt_path, 9, 1,expect_total=8192).astype(np.float32)#2^13
    us.CV2_IMSHOW_NHWC_RAMDOM(batch,1,25,5,5,'batch',0)

    # 获取拉式金字塔
    PYD = us.lap_pyd_nhwc(batch,4)
    for level in PYD.values():
        us.CV2_IMSHOW_NHWC_RAMDOM(us.UINT8(level), 1, 25, 5, 5, 'batch', 0)

    # 获取样本集各频带descriptors
    DESC = get_descriptors_for_all_level(PYD)
    del PYD

    # 保存
    us.PICKLE_SAVING(DESC,r'./DESC.desc')

    # 测试
    HPF = us.hpf_nhwc(batch)
    del batch
    h_desc = get_descriptors_for_minibatch(HPF, 7, 64)
    h_desc = finalize_descriptors(h_desc)
    swd = sliced_wasserstein_distance(h_desc, DESC['128'], 4, 128) * 1e3
    print('swd = ',swd)



