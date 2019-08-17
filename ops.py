"""
script: ops
Author:Ephemeroptera
date:2019-8-1
mail:605686962@qq.com

数据格式：nf = [num,fmaps] , nhwf = [num,height,width,fmaps] , nd = nf or nhwf = [num,data]
"""
import tensorflow as tf
import numpy as np

#--------------------------------------------------- base operations ---------------------------------------------------

# 获取张量shape
def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

# 获取数据集个数
def get_N(tensor):
    return tf.shape(tensor)[0]

# 获取归一化权值（equalized learning rate）
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    """
    HE公式：0.5*n*var(w)=1 , so：std(w)=sqrt(2)/sqrt(n)=gain/sqrt(fan_in)
    """
    # 某卷积核参数个数(h*w*fmaps1)或dense层输入节点数目fmaps1
    # conv_w:[H,W,fmaps1,fmaps2] or mlp_w:[fmaps1,fmaps2]
    if fan_in is None: fan_in = np.prod(shape[:-1])
    # He init
    std = gain / np.sqrt(fan_in)
    # 归一化
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return  tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal())*wscale
    else:
        return  tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0,std))

# 定义像素归一化操作（pixel normalization）
def PN(nd):
    if len(nd.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    epsilon = 1e-8
    with tf.variable_scope('PixelNorm'):
        return nd * tf.rsqrt(tf.reduce_mean(tf.square(nd), axis=axis_, keep_dims=True) + epsilon)

# 2d卷积
def conv2d(nhwf, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, nhwf.shape[3].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, nhwf.dtype)
    """
    tf.nn.conv2d:
                input:[N,H,W,fmaps1]
                filter:[H,W,fmaps1,fmaps2]
                output:[N,H,W,fmaps2]
    """
    return tf.nn.conv2d(nhwf, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

# dense
def dense(nd, fmaps, gain=np.sqrt(2), use_wscale=False):
    # 平铺至1D
    if len(nd.shape) > 2:
        nd = tf.reshape(nd, [-1, np.prod([d.value for d in nd.shape[1:]])])
    # 获取权值
    w = get_weight([nd.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, nd.dtype)
    """
    tf.matmul:
             input:[N,fmaps1]
             w: [fmaps1,fmaps2]
             output: [N,fmaps2]
    """
    return tf.matmul(nd, w)

# 添加偏置
def add_bias(nd):
    # nums(b) = channels
    if len(nd.shape) == 2:# [N,fmaps1]
        b = tf.get_variable('bias', shape=[nd.shape[1]], initializer=tf.initializers.zeros(), dtype=nd.dtype)
        return nd + b # for FC
    else: # [N,H,W,fmaps1]
        b = tf.get_variable('bias', shape=[nd.shape[3]], initializer=tf.initializers.zeros(), dtype=nd.dtype)
        return nd + tf.reshape(b, [1, 1, 1, -1]) # for CONV

# leaky relu
def lrelu(nd):
    return tf.nn.leaky_relu(nd, alpha=0.2, name='lrelu')

# 上采样
def upsampling2d(nhwf):
    _, h, w, _ = int_shape(nhwf)
    return tf.image.resize_nearest_neighbor(nhwf, (2 * h, 2 * w))

# 下采样
def downsampling2d(nhwf):
    return tf.nn.avg_pool(nhwf, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# 添加多样性特征
def MinibatchstateConcat(nhwf, averaging='all'):
    # input:[N,H,W,fmaps]
    s = nhwf.shape
    # 获取批大小
    group_size = get_N(nhwf)
    """
    计算方法：
            (1)先计算N个特征图的标准差得到特征图fmap1:[1,H,W,fmaps]
            (2)对fmap1求均值 得到值M1:[1,1,1,1]
            (3)复制扩张M2得到N个特征图fmap2:[N,H,W,1]
            (4)将fmap2添加至每个样本的特征图中
    """
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(nhwf, axis=0, keep_dims=True)
    # 求均值
    vals = tf.reduce_mean(vals, keep_dims=True)
    # 复制扩张
    vals = tf.tile(vals, multiples=(group_size, s[1].value, s[2].value, 1))
    # 将统计特征拼接到每个样本特征图中
    return tf.concat([nhwf, vals], axis=3)


#---------------------------------------------------  G/D --------------------------------------------------------------
# 定义不同level所需feature maps数目
def fn(level):
    FN = [512,512,512,512,256,128,64,32,16]
    return FN[level-2]

# 定义生成卷积块
def G_CONV_BLOCK(nhwf, level, use_wscale=False):
    """
    上采样+CONV0 = pyrUp
    """
    # 上采样
    with tf.variable_scope('upscale2d'):
        nhwf = upsampling2d(nhwf)
    # CONV0
    with tf.variable_scope('CONV0'):
        nhwf = PN(lrelu(add_bias(conv2d(nhwf, fmaps=fn(level), kernel=3, use_wscale=use_wscale))))
    # CONV1
    with tf.variable_scope('CONV1'):
        nhwf = PN(lrelu(add_bias(conv2d(nhwf, fmaps=fn(level), kernel=3, use_wscale=use_wscale))))
    return nhwf

# 定义判别卷积块

def D_CONV_BLOCK(nhwf, level):
    """
        CONV1+下采样 = pyrDown
    """
    # CONV0
    with tf.variable_scope('CONV0'):
        nhwf = lrelu(add_bias(conv2d(nhwf, fmaps=fn(level), kernel=3, use_wscale=True)))
    # CONV1,增加特征图个数,fmaps数量改变发生在该卷积，即nf(level) to nf(level-1)
    with tf.variable_scope('CONV1'):
        nhwf = lrelu(add_bias(conv2d(nhwf, fmaps=fn(level - 1), kernel=3, use_wscale=True)))
    # 下采样
    with tf.variable_scope('dowbscale2d'):
        nhwf = downsampling2d(nhwf)
    return nhwf


# 定义toRGB
def toRGB(nhwf, level, use_wscale=False):  # [N,C,H,W] to [N,3,H,W]   W:[1,1,C,3]
    with tf.variable_scope('level_%d_toRGB' % level):
        return add_bias(conv2d(nhwf, fmaps=3, kernel=1, gain=1, use_wscale=use_wscale))

# 定义formRGB
def fromRGB(nhwf, level, fmaps):
    with tf.variable_scope('level_%d_fromRGB' % (level)):
        return lrelu(add_bias(conv2d(nhwf, fmaps=fmaps, kernel=1, use_wscale=True)))

# 定义生成器
def Generator_PG(latents,level,reuse = False,isTransit = False,trans_alpha = 0.0):
    """
    :param latents: 输入分布
    :param level: 网络等级（阶段）
    :param reuse: 变量复用
    :param isTransit: 是否fade_in
    :param trans_alpha: 过度系数
    :return: 生成图片
    """
    """
        说明：（1）Generator构成：scale_2 + scale_3~level + toRGB , 其中toRGB层将全部特征图合成RGB
              (2) 过渡阶段: ① 本阶段RGB将融合上一阶段RGB输出。对于上一阶段RGB处理层而言，通过特征图上采样匹配大小，再toRGB再融合。
                           ② 上一阶段toRGB的卷积核参数对于上采样后的特征图依然有效
    """
    # ******************************* 构造PG生成器 ************************************
    with tf.variable_scope('generator',reuse=reuse):
        # ******** 构造二级初始架构 ******************
        with tf.variable_scope('scale_%d'%(2)):
            nf = PN(latents)
            # 论文:CONV4x4+CONV3x3，这里CONV4x4采用FC替代（参考论文源码）
            with tf.variable_scope('Dense0' ):
                nf = dense(nf,fmaps=fn(2)*4*4,gain=np.sqrt(2)/4,use_wscale=True)# Dense0:[N,512] to [N,4*4*512}
                nhwf = tf.reshape(nf,[-1, 4, 4,fn(2)])# reshape:[N,4*4*512} to [N,4,4,512]
                nhwf = PN(lrelu(add_bias(nhwf)))
            with tf.variable_scope('CONV1'):
                nhwf = PN(lrelu(add_bias(conv2d(nhwf,fmaps=fn(2), kernel=3, use_wscale=True))))

        # ********* 构造拓扑架构（3~level） *********************
        for scale in range(3,level+1):
            if scale == level and isTransit: # 在最后卷积层新建之前，获取当前输出图片并上采样
                RGB0 = upsampling2d(nhwf)  # 上采样
                RGB0 = toRGB(RGB0,scale-1,use_wscale=True)# toRGB
            with tf.variable_scope('scale_%d'%scale):
                nhwf = G_CONV_BLOCK(nhwf,scale,use_wscale=True)# 卷积层拓展

        # ******************* toRGB *****************************
        RGB1 = toRGB(nhwf, level,use_wscale=True)  # 获取最后卷积层输出图像
        # 判断是否为过渡阶段
        if isTransit:
            nhw3 = trans_alpha * RGB1 + (1 - trans_alpha) * RGB0  # 由RGB0平滑过渡到RGB1
        else:
            nhw3 = RGB1

        return nhw3

# 定义判别器
def Discriminator_PG(RGB,level,reuse = False,isTransit = False,trans_alpha=0.0):
    """
    :param RGB: 输入图像
    :param level: 网络等级（阶段）
    :param reuse: 变量复用
    :param isTransit: 是否fade_in
    :param trans_alpha: 过度系数
    :return: 样本评分
    """
    """
        说明：（1）Discriminator构成：fromRGB + scale_level~3 + scale_2 ,其中fromRGB是对RGB信号分解
             （2）过渡阶段：①本阶段新的网络层scale_level 分解的结果融合上一阶段fromRGB。上一阶段通过下采样匹配大小，再fromRGB再融合
                           ②上一阶段fromRGB卷积核参数对于本阶段下采样后的依然有效
    """

    # ********************************* 动态构造判别器 ************************************
    with tf.variable_scope("discriminator",reuse= reuse) :
        # ******************* fromRGB *********************************
        # 降采样分解
        if isTransit:
            RGB0 = downsampling2d(RGB) # 0.5x
            nhwf0 = fromRGB(RGB0, level-1, fn(level-1)) # fromRGB
        # 新增网络层分解
        nhwf = fromRGB(RGB, level, fn(level))

        # ******************* 构造拓扑架构（level~3）***********************
        for scale in range(level,2,-1):
            with tf.variable_scope('scale_%d' % (scale)):
                nhwf = D_CONV_BLOCK(nhwf,scale) # 拓展卷积层
                # 在新建第一层卷积层后，获取该层的卷积结果x。在过渡阶段实现过渡
                if scale==level and isTransit:
                    nhwf = trans_alpha*nhwf+(1-trans_alpha)*nhwf0

        # ****************** 构造二级终极架构 ******************************
        with tf.variable_scope('scale_%d' % (2)):
            # 加入多样性特征
            nhwf = MinibatchstateConcat(nhwf)
            # 卷积层转输出节点，论文：CONV3x3+CONV4x4 , 这里CONV4x4 用FC替代
            with tf.variable_scope('CONV0'):
                nhwf = lrelu(add_bias(conv2d(nhwf, fmaps=fn(2), kernel=3, use_wscale=True)))
            with tf.variable_scope('Dense0'):
                nhwf = lrelu(add_bias(dense(nhwf, fmaps=fn(2), use_wscale=True)))
            with tf.variable_scope('Dense1'):
                logits = add_bias(dense(nhwf, fmaps=1, gain=1, use_wscale=True))

        return  logits

#--------------------------------------------------- 模型参数 ----------------------------------------------------------
# 统计参数个数数目
def COUNT_VARS(vars):
    total_para = 0
    for variable in vars:
        # get each shape of vars
        shape = variable.get_shape()
        variable_para = 1
        for dim in shape:
            variable_para *= dim.value
        total_para += variable_para
    return total_para

# 记录参数信息
def ShowParasList(d_vars,g_vars,level,isTrans):
    p = open('./structure/level%d_trans_%s_Paras.txt'%(level,isTrans), 'w')
    # D paras
    print('正在记录Discriminator参数信息..')
    p.writelines(['Discriminator_vars_total: %d\n'%COUNT_VARS(d_vars)])
    for variable in d_vars:
        p.writelines([variable.name, str(variable.get_shape()),'\n'])

    p.writelines(['\n','\n','\n'])
    # G paras
    print('正在记录Generator参数信息..')
    p.writelines(['Generator_vars_total: %d\n' % COUNT_VARS(d_vars)])
    for variable in g_vars:
        p.writelines([variable.name, str(variable.get_shape()), '\n'])
    p.close()

# 新旧模型参数匹配
def VARS_MATCH(old_model_path, vars):
    # 获取模型文件名
    ckpt = tf.train.get_checkpoint_state(old_model_path)
    latest = ckpt.model_checkpoint_path
    # 读取模型
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(latest)
    # 获取所有变量
    var_to_shape_map = reader.get_variable_to_shape_map()
    # 检查型号是否匹配
    for key in var_to_shape_map.keys():
        tensorName = key
        tensorShape = var_to_shape_map[key]
        for var in vars:
            if tensorName in var.name:
                assert list(var.get_shape()) == tensorShape