import os
import cv2
import numpy as np
import ops
import tfr_tools as tfr
import visualization as vs

# CelebA属性列表 attr_list.txt
def build_attr_list(attr_txt_path):
    f = open(attr_txt_path)
    img_total = f.readline()
    attrs = f.readline()
    g = open(r"./attr_list.txt",'w')
    for idx,attr in enumerate(attrs.split()):
        g.writelines([attr,':  %d\n'%(idx+1)])
    f.close()
    g.close()

# 获取指定数据
def get_specfic_data(celeba_path,attr_txt_path,attr_idx,flag=1,resize=(128,128),show=False):
    # 打开attr.txt
    f = open(attr_txt_path)
    # 数目和属性
    img_total = f.readline()
    attrs = f.readline()
    # 提取目标数据
    # 指定属性
    line = f.readline()
    num = 0
    data = []
    while line:
        array = line.split()
        target = int(array[attr_idx])
        # 提取目标图片
        if target == flag:
            num += 1
            print('捕捉第%d张图片..' % num)
            filename = array[0]
            filename = os.path.join(celeba_path, filename)
            img = cv2.imread(filename)
            img = cv2.resize(img, resize)
            if show:
                cv2.imshow('CELEBA',img)
                cv2.waitKey(0)
            data.append(img)
        line = f.readline()
    return np.array(data) / 255

if __name__ == '__main__':

    # 指定路径
    celeba_path = r'I:\CELEBA\img_align_celeba'
    attr_txt_path = r"I:\CELEBA\list_attr_celeba.txt"

    # 获取属性列表
    build_attr_list(attr_txt_path)

    # 获取目标数据
    data = get_specfic_data(celeba_path,attr_txt_path,16,1)

    # 缩放 和 保存 [128,64,32,16,8,4]
    for i in range(6):
        # 缩放 data[-1,scale*H,scale*W,3]
        if i>0:
            low = ops.batch_downscale2d(low)
        else:
            low = data
        # 分辨率
        res = low.shape[1]
        print('当前分辨率：%dx%d..'%(res,res))
        # 显示
        vs.CV2_BATCH_RANDOM_SHOW(low,1,25,5,5)
        # 设置标签
        label = np.zeros(low.shape[0], dtype=np.int8)
        # 保存
        savename = './TFR/celeba_glass_%dx%d' % (res, res)
        tfr.Saving_All_TFR(savename,(low/2-1).astype(np.float32),label,5)
