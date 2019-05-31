import visualization as vs
import numpy as np
import pickle
import CelebA_MultiRes_Preprocess as pp
import ops

# 数据集指定频带的descriptors: [N,H,W,3] to [N,k,k,3]
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape
    H = nhood_size // 2 # 卷积核半径
    descriptors = []
    for img in minibatch:
        # 随机采样descriptor
        X =  np.random.randint(H, S[2] - H,nhoods_per_image)
        Y =  np.random.randint(H, S[1] - H, nhoods_per_image)
        # 裁取每个patch
        for x,y in zip(X,Y):
            descriptors.append(img[y-H:y+H+1,x-H:x+H+1,:])
    return np.array(descriptors)

# descriptor归一化
def finalize_descriptors(desc):
    desc -= np.mean(desc, axis=(0, 1, 2), keepdims=True)
    desc /= np.std(desc, axis=(0, 1, 2), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

# 获取各频带的descriptors
def get_descriptors_for_all_level(PYD):
    DESC = {}
    for level in PYD.values():
        res = level.shape[1]
        desc = get_descriptors_for_minibatch(level, 7, 128)
        desc = finalize_descriptors(desc)
        DESC[str(res)] = desc
        print('已提取%dx%d频带descriptors..' % (res, res))
    return DESC

# 计算sliced wasserstein distance
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
    return np.mean(results)

if __name__ == '__main__':

    # 指定路径
    celeba_path = 'I:\CELEBA\img_align_celeba'
    attr_txt_path = r"I:\CELEBA\list_attr_celeba.txt"

    # 获取数据集
    batch = pp.get_specfic_data(celeba_path,attr_txt_path,16,1)
    batch = batch[0:8192]  # 2^13
    vs.CV2_BATCH_RANDOM_SHOW(batch, 1, 25, 5, 5, 0)


    # 获取拉式金字塔[128,64,32,16]
    PYD = ops.batch_Lap_Pyd(batch,4)
    for pyd in PYD.values():
        vs.CV2_BATCH_RANDOM_SHOW(pyd, 1, 25, 5, 5, 0)

    # 获取样本各频带descriptor
    DESC = get_descriptors_for_all_level(PYD)

    # 保存
    with open('./DESC.des', 'wb') as d:
        pickle.dump(DESC, d)
        d.close()

    # 测试
    H = ops.batch_hpf(batch)
    h_desc = get_descriptors_for_minibatch(H, 7, 128)
    h_desc = finalize_descriptors(h_desc)
    swd = sliced_wasserstein(h_desc, DESC['128'], 4, 128) * 1e3
    print(swd)



