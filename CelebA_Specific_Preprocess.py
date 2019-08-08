"""
script: CelebA_Specific_Preprocess
Author:Ephemeroptera
date:2019-8-1
mail:605686962@qq.com
"""

import os
import cv2
import numpy as np
import utils
import tfr_tools as tfr

"""
CELEBA的标签文件 list_attr_celeba.txt中:
    第1行：图片数目
    第2行：人脸的一些属性
    第>=3行：图片的文件名 + 各个属性标签（1：正例，-1：反例）
    
整理CelebA属性列表并标注序号，保存为 attr_list.txt，便于提取某一相同属性的人脸
"""
def build_attr_list(attr_txt_path):
    f = open(attr_txt_path)
    img_total = f.readline() # 数量行
    attrs = f.readline() # 属性行
    g = open(r"./attr_list.txt",'w')
    for idx,attr in enumerate(attrs.split()):
        g.writelines([attr,':  %d\n'%(idx+1)])
    f.close()
    g.close()





"""
在attr_list.txt中我们可以轻松的选择我们想要的属性，并在标签文件list_attr_celeba.txt文件中获得该属性下全部图像的文件名。
并读取图片且resize目标大小（flag=1表示属性为正例）_
"""
def get_specfic_data(celeba_path, attr_txt_path, attr_idx, flag=1, expect_total=20000, batchs=5, size=(128, 128), show=False):
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

    # batch
    batch_size = expect_total//batchs
    batch_n = 0
    idxs = tfr.get_seg_index(expect_total,batchs)

    while line:
        array = line.split()
        target = int(array[attr_idx])
        # 提取目标图片
        if target == flag:

            filename = array[0]
            filename = os.path.join(celeba_path, filename)
            # 读取图片
            img = cv2.imread(filename)
            # 扣取人脸
            img = utils.CV2_CROP_FACE(img)
            if img is None:
                line = f.readline()
                continue
            num += 1
            print('捕捉第%d张图片:%s..' % (num, filename))
            # resize
            img = cv2.resize(img, size)
            if show:
                cv2.imshow('CELEBA',img)
                cv2.waitKey(0)
            data.append(img)
            # 设定期望数据量
            if num % batch_size==0 and num>0:
                # 多尺度缩放 和 保存 [128,64,32,16,8,4]
                seg = [idxs[batch_n], idxs[batch_n + 1]]
                idx = tfr.get_seg_idx(seg[0], seg[1] - 1)
                for i in range(6):
                    if i > 0:
                        lower = utils.downsampling_nhwc(lower)
                    else:
                        lower = np.array(data)/255

                    # 分辨率
                    res = lower.shape[1]
                    print('当前分辨率：%dx%d..' % (res, res))
                    # 显示
                    # utils.CV2_IMSHOW_NHWC_RAMDOM(lower, 1, 25, 5, 5, 'lower', 0)
                    # 设置标签
                    label = np.zeros(lower.shape[0], dtype=np.int8)
                    # 保存
                    savename = './TFR/celeba_%dx%d' % (res, res)
                    tfr.Saving_Batch_TFR(savename,idx,lower.astype(np.float32),label,batch_n,batchs-1)
                batch_n += 1
                del lower
                del data
                data = []
            # 读取完毕
            if num == expect_total:
                break
        # 读取下一数据
        line = f.readline()


if __name__ == '__main__':

    # 指定路径
    celeba_path = r'F:\ww\CELEBA\img_align_celeba'
    attr_txt_path = r"F:\ww\CELEBA\list_attr_celeba.txt"

    # 建立属性列表
    build_attr_list(attr_txt_path)

    # 获取目标数据
    get_specfic_data(celeba_path,attr_txt_path,9,1,30000,10)


