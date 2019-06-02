import tensorflow as tf
import numpy as np
import os
import time
import ops
import pickle
import tfr_tools as tfr
import visualization as vs
import sliced_wasserstein_distance as SWDs

#******************************************** utilities ***************************************************************#

# 获取张量shape
def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

# 获取归一化权值（equalized learning rate）
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    """
    HE公式：0.5*n*var(w)=1 , so：std(w)=sqrt(2)/sqrt(n)
    """
    # 某卷积核参数个数(h*w*c)或dense层输入节点数目
    # conv_w:[H,W,C,fmaps] or mlp_w:[input,fmaps]
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
def PN(x):
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    epsilon = 1e-8
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keepdims=True) + epsilon)

# 2d卷积
def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[3].value, fmaps],gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    """
    tf.nn.conv2d:
                input:[N,H,W,fmaps1]
                filter:[H,W,fmaps1,fmaps2]
                output:[N,H,W,fmaps2]
    """
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

# dense
def dense(x, fmaps,gain=np.sqrt(2), use_wscale=False):
    # 平铺至1D
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    # 获取权值
    w = get_weight([x.shape[1].value, fmaps],gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    """
    tf.matmul:
             input:[-1,input_n]
             w: [input_n,fmaps]
             output: [-1,fmaps]
    """
    return tf.matmul(x, w)

# 添加偏置
def add_bias(x):
    # nums(b) = channels
    if len(x.shape) == 2:
        b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros(), dtype=x.dtype)
        return x + b # for FC
    else:
        b = tf.get_variable('bias', shape=[x.shape[3]], initializer=tf.initializers.zeros(), dtype=x.dtype)
        return x + tf.reshape(b, [1, 1, 1, -1]) # for CONV

# leaky relu
def lrelu(x):
    return tf.nn.leaky_relu(x,alpha=0.2,name='lrelu')

# 上采样
def upscale2d(x):
    _, h, w, _ = int_shape(x)
    return tf.image.resize_nearest_neighbor(x, (2*h,2*w))

# 下采样
def downscale2d(x):
    # avgpool wrapper
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID')

# 添加多样性特征
def MinibatchstateConcat(input, averaging='all'):
    # input:[N,H,W,fmaps]
    s = input.shape
    # 获取批大小
    group_size = tf.shape(input)[0]
    """
    计算方法：
            (1)先计算N个特征图的标准差得到特征图fmap1:[1,H,W,fmaps]
            (2)对fmap1求均值 得到值M1:[1,1,1,1]
            (3)复制扩张M2得到N个特征图fmap2:[N,H,W,1]
            (4)将fmap2添加至每个样本的特征图中
    """
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")
    vals = tf.tile(vals, multiples=(group_size, s[1].value, s[2].value, 1))
    return tf.concat([input, vals], axis=3)

# build related dirs
def GEN_DIR():
    if not os.path.isdir('ckpt'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('ckpt')
    if not os.path.isdir('trainLog'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('trainLog')
    if not os.path.isdir('structure'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('structure')

# counting total to vars
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

# display paras infomation
def ShowParasList(d_vars,g_vars,level,isTrans):
    p = open('./structure/level%d_%d_Paras.txt'%(level,isTrans), 'w')
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

# 定义参数匹配检查，用来检查当前网络的部分参数是否可以使用上一级网络的参数
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

# 保存训练记录
def Saving_Train_Log(filename,var,dir=r'./trainLog'):
    var = np.array(var)
    f = open(os.path.join(dir,filename),'wb')
    pickle.dump(var,f)
    f.close()
    print('成功保存记录：%s!'%filename)

#******************************************************* 定义生成器和判别器 *******************************************#
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
              (2) 过渡阶段：① 本阶段RGB将融合上一阶段RGB输出。对于上一阶段RGB处理层而言，通过特征图上采样匹配大小，再toRGB再融合。
                          ② 上一阶段toRGB的卷积核参数对于上采样后的特征图依然有效
        """
    # ***************************** 定义依赖 *************************************
    # 定义特征图数量
    def nf(level):
        NF = [512,512,512,512,256,128,64,32,16]
        return NF[level-2]

    # 定义toRGB:[H,W,fmaps] to [H,W,3]
    def toRGB(x, level,use_wscale=False):  # [N,C,H,W] to [N,3,H,W]   W:[1,1,C,3]
        with tf.variable_scope('level_%d_toRGB' % level):
            return add_bias(conv2d(x, fmaps=3, kernel=1, gain=1, use_wscale=use_wscale))

    # 定义卷积块
    """
    上采样+CONV0 = pyrUp
    """
    def G_CONV_BLOCK(x, level,use_wscale=False):
        # 上采样
        with tf.variable_scope('upscale2d'):
            x = upscale2d(x)
        # CONV0
        with tf.variable_scope('CONV0'):
            x = PN(lrelu(add_bias(conv2d(x,fmaps=nf(level), kernel=3, use_wscale=use_wscale))))
        # CONV1
        with tf.variable_scope('CONV1'):
            x = PN(lrelu(add_bias(conv2d(x, fmaps=nf(level), kernel=3, use_wscale=use_wscale))))
        return x

    # ******************************* 动态构造生成器 ************************************
    with tf.variable_scope('generator',reuse=reuse):
        # ******** 构造二级初始架构 ******************
        with tf.variable_scope('scale_%d'%(2)):
            x = PN(latents)
            # 论文:CONV4x4+CONV3x3，这里CONV4x4采用FC替代（参考论文源码）
            with tf.variable_scope('Dense0' ):
                x = dense(x,fmaps=nf(2)*4*4,gain=np.sqrt(2)/4,use_wscale=True)# Dense0:512 to 512*4*4
                x = tf.reshape(x,[-1, 4, 4,nf(2)])# reshape:4*4*512 to 4x4x512
                x = PN(lrelu(add_bias(x)))
            with tf.variable_scope('CONV1'):
                x = PN(lrelu(add_bias(conv2d(x,fmaps=nf(2), kernel=3, use_wscale=True))))

        # ********* 构造拓扑架构（3~level） *********************
        for scale in range(3,level+1):
            if scale == level and isTransit: # 在最后卷积层新建之前，获取当前输出图片并上采样
                RGB0 = upscale2d(x)  # 上采样
                RGB0 = toRGB(RGB0,scale-1,use_wscale=True)# toRGB
            with tf.variable_scope('scale_%d'%scale):
                x = G_CONV_BLOCK(x,scale,use_wscale=True)# 卷积层拓展

        # ******************* toRGB *****************************
        RGB1 = toRGB(x, level,use_wscale=True)  # 获取最后卷积层输出图像
        # 判断是否为过渡阶段
        if isTransit:
            output = trans_alpha * RGB1 + (1 - trans_alpha) * RGB0  # 由RGB0平滑过渡到RGB1
        else:
            output = RGB1

        return output

# 定义判别器
def Discriminator_PG(RGB,level,reuse = False,isTransit = False,trans_alpha = 0.0):
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
                          ②上一阶段fromRGB卷积核参数对于本阶段下采样后的依然RGB有效
        """
    # ***************************** 定义依赖 *************************************
    # 定义特征图数量
    def nf(level):
        NF = [512,512,512,512,256,128,64,32,16]
        return NF[level-2]

    # 定义formRGB:[N,H,W,3] to [N,H,W,fmaps]
    def fromRGB(x, level, fmaps):
        with tf.variable_scope('level_%d_fromRGB' % (level)):
            return lrelu(add_bias(conv2d(x, fmaps=fmaps, kernel=1, use_wscale=True)))

    # 定义卷积块
    """
    CONV1+下采样 = pyrDown
    """
    def D_CONV_BLOCK(x, level):
        # CONV0
        with tf.variable_scope('CONV0'):
            x = lrelu(add_bias(conv2d(x, fmaps=nf(level), kernel=3, use_wscale=True)))
        # CONV1,增加特征图个数,fmaps数量改变发生在该卷积，即nf(level) to nf(level-1)
        with tf.variable_scope('CONV1'):
            x = lrelu(add_bias(conv2d(x, fmaps=nf(level - 1), kernel=3, use_wscale=True)))
        # 下采样
        with tf.variable_scope('dowbscale2d'):
            x = downscale2d(x)
        return x

    # ********************************* 动态构造判别器 ************************************
    with tf.variable_scope("discriminator",reuse= reuse) :
        # ******************* fromRGB *********************************
        # 降采样分解
        if isTransit:
            RGB0 = downscale2d(RGB) # 0.5x
            x0 = fromRGB(RGB0, level-1, nf(level-1)) # fromRGB
        # 新增网络层分解
        x = fromRGB(RGB, level, nf(level))

        # ******************* 构造拓扑架构（level~3）***********************
        for scale in range(level,2,-1):
            with tf.variable_scope('scale_%d' % (scale)):
                x = D_CONV_BLOCK(x,scale) # 拓展卷积层
                # 在新建第一层卷积层后，获取该层的卷积结果x。在过渡阶段实现过渡
                if scale==level and isTransit:
                    x = trans_alpha*x+(1-trans_alpha)*x0

        # ****************** 构造二级终极架构 ******************************
        with tf.variable_scope('scale_%d' % (2)):
            # 加入多样性特征
            x = MinibatchstateConcat(x)
            # 卷积层转输出节点，论文：CONV3x3+CONV4x4 , 这里CONV4x4 用FC替代
            with tf.variable_scope('CONV0'):
                x = lrelu(add_bias(conv2d(x, fmaps=nf(2), kernel=3, use_wscale=True)))
            with tf.variable_scope('Dense0'):
                x = lrelu(add_bias(dense(x, fmaps=nf(2), use_wscale=True)))
            with tf.variable_scope('Dense1'):
                output = add_bias(dense(x, fmaps=1, gain=1, use_wscale=True))

        return  output

#************************************************ 定义PGGAN计算图 ****************************************************#
def PGGAN(
            latents_size, # 噪声型号
            batch_size, # 批型号
            lowest,# 最低网络级数
            highest,#最高网络级数
            level,# 目标网络等级
            isTransit, # 是否过渡
            epochs, # 训练循环次数
            data_size, # 数据集大小
                ):
    #-------------------- 超参 --------------------------#
    learning_rate = 0.001
    lam_gp = 10
    lam_eps = 0.001
    beta1 = 0.0
    beta2 = 0.99
    max_iters = int(epochs * data_size / batch_size)
    n_critic = 1  # 判别器训练次数

    #---------- （1）创建目录和指定模型路径 -------------#
    GEN_DIR()
    # 当前模型路径
    model_path = './ckpt/PG_level%d_%s' % (level, isTransit)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)  # 创建目录
    # 上一级网络模型路径
    if isTransit:
        old_model_path = r'./ckpt/PG_level%d_%s/' % (level - 1, not isTransit)  # 上一阶段稳定模型
    else:
        old_model_path = r'./ckpt/PG_level%d_%s/' % (level, not isTransit)  # 该阶段过度模型

    #--------------------- (2)定义输入输出 --------------#

    # 图像分辨率
    res = int(2 ** level)
    # 定义噪声输入
    latents = tf.placeholder(name='latents', shape=[None, latents_size], dtype=tf.float32)
    # 定义数据输入
    real_images = tf.placeholder(name='real_images', shape=[None, res, res, 3], dtype=tf.float32)
    # 训练步数
    train_steps = tf.Variable(0, trainable=False, name='train_steps', dtype=tf.float32) # 等于生成器训练次数

    # 生成器和判别器输出
    fake_images = Generator_PG(latents=latents, level=level, reuse=False, isTransit=isTransit,
                               trans_alpha=train_steps / max_iters)
    d_real_logits = Discriminator_PG(RGB=real_images, level=level, reuse=False, isTransit=isTransit,
                                     trans_alpha=train_steps / max_iters)
    d_fake_logits = Discriminator_PG(RGB=fake_images, level=level, reuse=True, isTransit=isTransit,
                                     trans_alpha=train_steps / max_iters)

    #------------ (3)Wasserstein距离和损失函数 --------------#
    # 定义wasserstein距离
    Wass = tf.reduce_mean(d_real_logits-d_fake_logits)

    # 定义G,D损失函数
    d_loss = -Wass  # 判别器损失函数
    g_loss = -tf.reduce_mean(d_fake_logits)  # 生成器损失函数

    # 基于‘WGAN-GP’的梯度惩罚
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)  # 获取[0,1]之间正态分布
    alpha = alpha_dist.sample((batch_size, 1, 1, 1))
    interpolated = real_images + alpha * (fake_images - real_images)  # 对真实样本和生成样本之间插值
    inte_logit = Discriminator_PG(RGB=interpolated, level=level, reuse=True, isTransit=isTransit,
                                  trans_alpha=train_steps / max_iters)  # 求得对应判别器输出

    # 求得判别器梯度
    gradients = tf.gradients(inte_logit, [interpolated, ])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    # 定义惩罚项
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    # d_loss加入惩罚项
    d_loss += gradient_penalty * lam_gp

    # 零点偏移修正
    d_loss += tf.reduce_mean(tf.square(d_real_logits)) * lam_eps

    # ------------ (4)模型保存和恢复 --------------#
    # 获取G,D 所有可训练参数
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    ShowParasList(d_vars, g_vars, level, isTransit)

    # 提取本阶段各级网络层参数（不含RGB处理层）
    d_vars_c = [var for var in d_vars if 'fromRGB' not in var.name]  # discriminator/scale_(0~level)/
    g_vars_c = [var for var in g_vars if 'toRGB' not in var.name]  # generator/scale_(0~level)/

    # 提取上一阶段各级网络层参数（不含RGB处理层）
    d_vars_old = [var for var in d_vars_c if 'scale_%d' % level not in var.name]  # discriminator/scale_(0~level-1)/
    g_vars_old = [var for var in g_vars_c if 'scale_%d' % level not in var.name]  # generator/scale_(0~level-1)/

    # 提取所有阶段RGB处理层参数
    d_vars_rgb = [var for var in d_vars if 'fromRGB' in var.name]  # discriminator/level_*_fromRGB/
    g_vars_rgb = [var for var in g_vars if 'toRGB' in var.name]  # generator/level_*_toRGB/

    # 提取上一阶段RGB处理层参数
    d_vars_rgb_old = [var for var in d_vars_rgb if
                      'level_%d_fromRGB' % level not in var.name]  # discriminator/level_level-1_fromRGB/
    g_vars_rgb_old = [var for var in g_vars_rgb if
                      'level_%d_toRGB' % level not in var.name]  # generator/level_level-1_fromRGB/

    # 保存本阶段所有变量
    saver = tf.train.Saver(d_vars + g_vars)

    # 提取上一阶段全部变量
    old_vars = d_vars_old + g_vars_old + d_vars_rgb_old + g_vars_rgb_old
    if len(old_vars):
        old_saver = tf.train.Saver(d_vars_old + g_vars_old)
        VARS_MATCH(old_model_path, old_vars) # 核对

    # ------------ (5)梯度下降 --------------#
    # G,D梯度下降方式
    d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(g_loss, var_list=g_vars, global_step=train_steps)

    # ------------ (6)数据集读取（TFR） --------------#
    # read TFR
    [num, data, label] = tfr.Reading_TFR(sameName=r'./TFR/celeba_glass_%dx%d-*'%(res,res) ,
                                         isShuffle=False, datatype=tf.float32, labeltype=tf.int8)
    # # get batch
    [num_batch, data_batch, label_batch] = tfr.Reading_Batch_TFR(num, data, label, data_size=res*res*3,
                                                                 label_size=1, isShuffle=False, batchSize=batch_size)

    # ------------------ (7)迭代 ---------------------#
    # GPU配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 保存记录
    losses = []
    GenLog = []
    WASS = []

    # 记录swd
    if res>=512:
        # 加载训练数据的特征集
        d = open(r'./DESC.des', 'rb')
        DESC = pickle.load(d)
        d.close()
        SWD = []

    # 开启会话
    with tf.Session(config=config) as sess:

        # 全局和局部变量初始化
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # 开启协调器
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 加载上一阶段参数
        if level > lowest:
            if isTransit:  # 如果处于过渡阶段
                old_saver.restore(sess, tf.train.latest_checkpoint(old_model_path))  # 恢复历史模型
            else:  # 如果处于稳定阶段
                saver.restore(sess, tf.train.latest_checkpoint(old_model_path))  # 继续训练该架构

        # 迭代
        time_start = time.time()  # 开始计时
        for steps in range(1,max_iters+1):
            # 获取trans_alpha
            trans_alpha = steps / max_iters

            # 输入标准正态分布
            z = np.random.normal(size=(batch_size, latents_size))
            # 获取数据集
            minibatch = sess.run(data_batch)
            # 格式修正
            minibatch = np.reshape(minibatch,[-1,res,res,3]).astype(np.float32)
            # 数据集显示
            # vs.CV2_BATCH_SHOW((minibatch[0:9] + 1) / 2, 1, 3, 3, 0)

            # 数据集过度处理
            if isTransit:
                minibatch_low = ops.batch_lpf2(minibatch)# 低通滤波
                minibatch_input = trans_alpha * minibatch + (1 - trans_alpha) * minibatch_low  # 数据集过渡处理
            else:
                minibatch_input = minibatch
            minibatch_input = minibatch_input*2-1

            # 训练判别器
            for i in range(n_critic):
                sess.run(d_train_opt, feed_dict={real_images: minibatch_input, latents: z})

            # 训练生成器
            sess.run(g_train_opt, feed_dict={latents: z})

            # recording training info
            train_loss_d = sess.run(d_loss, feed_dict={real_images: minibatch_input, latents: z})
            train_loss_g = sess.run(g_loss, feed_dict={latents: z})
            Wasserstein = sess.run(Wass, feed_dict={real_images: minibatch_input, latents: z})

            # recording training_products
            z = np.random.normal(size=[9, latents_size])
            gen_samples = sess.run(fake_images, feed_dict={latents: z})
            vs.CV2_BATCH_SHOW((gen_samples[0:9] + 1) / 2, 1, 3, 3, delay=1)

            # 打印
            print('level:%d(%dx%d)..' % (level, res, res),
                  'isTrans:%s..' % isTransit,
                  'step:%d/%d..' % (steps, max_iters),
                  'Discriminator Loss: %.4f..' % (train_loss_d),
                  'Generator Loss: %.4f..' % (train_loss_g),
                  'Wasserstein:%.3f..'%Wasserstein)

            #  记录训练信息
            if steps % 10 == 0:
                # （1）记录损失函数
                losses.append([steps, train_loss_d, train_loss_g])
                WASS.append([steps,Wasserstein])

            if steps % 50 == 0:
                # （2）记录生成样本
                GenLog.append(gen_samples[0:9])

            # 计算swd
            if steps % 1000 == 0 and res >= 512:
                # 获取2^13个fake 样本
                FAKES = []
                for i in range(64):
                    z = np.random.normal(size=[128, latents_size])
                    fakes = sess.run(fake_images, feed_dict={latents: z})
                    FAKES.append(fakes)
                FAKES = np.concatenate(FAKES, axis=0)
                FAKES = (FAKES + 1) / 2
                # 计算与数据集拉式金字塔指定层的swd
                if res >16:
                    FAKES = ops.batch_hpf(FAKES) # 获取高频信号
                d_desc = SWDs.get_descriptors_for_minibatch(FAKES, 7, 128)# 提取特征
                d_desc = SWDs.finalize_descriptors(d_desc)
                swd = SWDs.sliced_wasserstein(d_desc, DESC[str(res)], 4, 128) * 1e3 # 计算swd*1e3
                SWD.append([steps,swd])
                print('当前生成样本swd(x1e3):', swd, '...')
                del [FAKES, d_desc]

            # 保存生成模型
            if steps % 1000 == 0:
                saver.save(sess, model_path + '/network.ckpt', global_step=steps)  # 保存模型

        # 关闭线程
        coord.request_stop()
        coord.join(threads)

        # 计时结束：
        vs.CV2_CLOSE()
        time_end = time.time()
        print('迭代结束，耗时：%.2f秒' % (time_end - time_start))

    # 保存信息
    Saving_Train_Log('losses_%dx%d_trans_%s'%(res,res,isTransit),losses)
    Saving_Train_Log('WASS_%dx%d_trans_%s'%(res,res,isTransit),WASS)
    Saving_Train_Log('GenLog_%dx%d_trans_%s'%(res,res,isTransit),GenLog)
    if res>=512:
        Saving_Train_Log('SWD_%dx%d_trans_%s'%(res,res,isTransit), SWD)

    # 清理图
    tf.reset_default_graph()

#********************************************************* main *******************************************************#
if __name__ == '__main__':
    # 超参
    latents_size = 512
    batch_size = 16
    lowest = 2
    highest = 7

    epochs = 20
    data_size = 13913
    current_lr = 0.001

    # progressive growing
    PGGAN(latents_size,batch_size,  lowest, highest, level=2, isTransit=False,epochs=epochs,data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=3, isTransit=True, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=3, isTransit=False, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=4, isTransit=True, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=4, isTransit=False, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=5, isTransit=True, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=5, isTransit=False, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=6, isTransit=True, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=6, isTransit=False, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=7, isTransit=True, epochs=epochs, data_size=data_size)
    PGGAN(latents_size, batch_size, lowest, highest, level=7, isTransit=False, epochs=epochs, data_size=data_size)







