import time
import os
from ops import *
import utils as us
import tfr_tools as tfr
import sliced_wasserstein_distance as swd
os.environ['CUDA_VISIBLE_DEVICES']='0'


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

    # 当前模型路径
    model_path = './ckpt/PG_level%d_%s' % (level, isTransit)
    us.MKDIR(model_path)
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
    wass_dist = tf.reduce_mean(d_real_logits-d_fake_logits)

    # 定义G,D损失函数
    d_loss = -wass_dist  # 判别器损失函数
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
    slopes_m = tf.reduce_mean(slopes)
    # 定义惩罚项
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    # d_loss加入惩罚项
    d_loss += gradient_penalty * lam_gp

    # 零点偏移修正
    d_loss += tf.reduce_mean(tf.square(d_real_logits)) * lam_eps

    # ------------ (4)模型可训练参数提取 --------------#
    # 获取G,D 所有可训练参数
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    ShowParasList(d_vars, g_vars, level, isTransit)# 记录参数

    # 提取本阶段各级网络层参数（不含RGB处理层）
    d_vars_c = [var for var in d_vars if 'fromRGB' not in var.name]  # discriminator/scale_(0~level)/
    g_vars_c = [var for var in g_vars if 'toRGB' not in var.name]  # generator/scale_(0~level)/

    # 提取上一阶段各级网络层参数（不含RGB处理层）
    d_vars_old = [var for var in d_vars_c if 'scale_%d' % level not in var.name]  # discriminator/scale_(0~level-1)/
    g_vars_old = [var for var in g_vars_c if 'scale_%d' % level not in var.name]  # generator/scale_(0~level-1)/

    # 提取本次和上次阶段RGB处理层参数
    d_vars_rgb = [var for var in d_vars if 'fromRGB' in var.name]  # discriminator/level_*_fromRGB/
    g_vars_rgb = [var for var in g_vars if 'toRGB' in var.name]  # generator/level_*_toRGB/

    # 提取上一阶段RGB处理层参数
    d_vars_rgb_old = [var for var in d_vars_rgb if
                      'level_%d_fromRGB' % level not in var.name]  # discriminator/level_level-1_fromRGB/
    g_vars_rgb_old = [var for var in g_vars_rgb if
                      'level_%d_toRGB' % level not in var.name]  # generator/level_level-1_fromRGB/

    # 提取上一阶段全部变量
    old_vars = d_vars_old + g_vars_old + d_vars_rgb_old + g_vars_rgb_old

    # ------------ (5)梯度下降 --------------#
    # G,D梯度下降方式
    d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(g_loss, var_list=g_vars, global_step=train_steps)
    # 为保持全局平稳学习，我们将保存adam参数的更新状态
    all_vars = tf.all_variables()
    adam_vars = [var for var in all_vars if 'Adam' in var.name]
    adam_vars_old = [var for var in adam_vars if 'level_%d' % level not in var.name and 'scale_%d' % level not in var.name]

    # ------------ (6)模型保存与恢复 ------------------#
    # 保存本阶段所有变量
    saver = tf.train.Saver(d_vars + g_vars + adam_vars,max_to_keep=3)
    # 提取上一阶段所有变量
    if level > lowest:
        VARS_MATCH(old_model_path, old_vars)  # 核对
        old_saver = tf.train.Saver(old_vars + adam_vars_old)

    # ------------ (7)数据集读取（TFR） --------------#
    # read TFR
    [num, data, label] = tfr.Reading_TFR(sameName=r'./TFR/celeba_glass_%dx%d-*'%(res,res) ,
                                         isShuffle=False, datatype=tf.float32, labeltype=tf.int8)
    # # get batch
    [num_batch, data_batch, label_batch] = tfr.Reading_Batch_TFR(num, data, label, data_size=res*res*3,
                                                                 label_size=1, isShuffle=False, batchSize=batch_size)

    # ------------------ (8)迭代 ---------------------#
    # GPU配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 保存记录
    losses = []
    Genlog = []
    Wass = []
    SWD = []

    # 加载数据集的descriptors集合
    # if res>=16:
    #     # 加载训练数据的特征集
    #     DESC = us.PICKLE_LOADING(r'./DESC.desc')

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
        if level>lowest:
            if isTransit:  # 如果处于过渡阶段
                old_saver.restore(sess, tf.train.latest_checkpoint(old_model_path))  # 恢复历史模型
                print('成功读取上一阶段参数...')
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
            # us.CV2_IMSHOW_NHWC_RAMDOM(minibatch, 1, 9, 3, 3, 'minibatch', 0)

            # 数据集过度处理
            if isTransit:
                # minibatch_low = us.lpf_nhwc(minibatch)
                # minibatch_input = trans_alpha * minibatch + (1 - trans_alpha) * minibatch_low  # 数据集过渡处理
                trans_res = int(0.5*res+0.5*trans_alpha*res)
                minibatch_input = us.upsize_nhwc(us.downsize_nhwc(minibatch,(trans_res,trans_res)),(res,res))
            else:
                minibatch_input = minibatch
            # 规格化【-1，1】
            minibatch_input = minibatch_input*2-1

            # 训练判别器
            for i in range(n_critic):
                sess.run(d_train_opt, feed_dict={real_images: minibatch_input, latents: z})

            # 训练生成器
            sess.run(g_train_opt, feed_dict={latents: z})

            # recording training info
            [d_loss2,g_loss2,wass_dist2,slopes2] = sess.run([d_loss,g_loss,wass_dist,slopes_m], feed_dict={real_images: minibatch_input, latents: z})

            # recording training_products
            z = np.random.normal(size=[9, latents_size])
            gen_samples = sess.run(fake_images, feed_dict={latents: z})
            us.CV2_IMSHOW_NHWC_RAMDOM((gen_samples+1)/2, 1, 9, 3, 3, 'minibatch', 10)

            # 打印
            print('level:%d(%dx%d)..' % (level, res, res),
                  'isTrans:%s..' % isTransit,
                  'step:%d/%d..' % (sess.run(train_steps), max_iters),
                  'Discriminator Loss: %.4f..' % (d_loss2),
                  'Generator Loss: %.4f..' % (g_loss2),
                  'Wasserstein:%.3f..'% wass_dist2,
                  'Slopes:%.3f..'%slopes2)


            #  记录训练信息
            if steps % 10 == 0:
                # （1）记录损失函数
                losses.append([steps, d_loss2, g_loss2])
                Wass.append([steps,wass_dist2])

            # if steps % 50 == 0:
                # （2）记录生成样本
                # GenLog.append(gen_samples[0:9])

            # 计算swd模块
            # if steps % 1000 == 0 and res>=16:
            #     # 获取2^13个fake 样本
            #     FAKES = []
            #     for i in range(64):
            #         z = np.random.normal(size=[128, latents_size])
            #         fakes = sess.run(fake_images, feed_dict={latents: z})
            #         FAKES.append(fakes)
            #     FAKES = np.concatenate(FAKES, axis=0)
            #     FAKES = (FAKES + 1) / 2
            #     # 计算与数据集拉式金字塔指定层的swd
            #     if res >16:
            #         FAKES = us.hpf_nhwc(FAKES) # 获取高频信号
            #     d_desc = swd.get_descriptors_for_minibatch(FAKES, 7, 64)# 提取特征
            #     del FAKES
            #     d_desc = swd.finalize_descriptors(d_desc)
            #     swd2 = swd.sliced_wasserstein_distance(d_desc, DESC[str(res)], 4, 64) * 1e3 # 计算swd*1e3
            #     SWD.append([steps,swd2])
            #     print('当前生成样本swd(x1e3):', swd2, '...')
            #     del d_desc

            # 保存生成模型
            if steps % 1000 == 0:
                saver.save(sess, model_path + '/network.ckpt', global_step=steps)  # 保存模型

        # 关闭线程
        coord.request_stop()
        coord.join(threads)

        # 计时结束：
        us.CV2_ALL_CLOSE()
        time_end = time.time()
        print('迭代结束，耗时：%.2f秒' % (time_end - time_start))

    # 保存信息
    us.PICKLE_SAVING(losses,'./trainlog/losses_%dx%d_trans_%s'%(res,res,isTransit))
    us.PICKLE_SAVING(Wass, './trainlog/Wass_%dx%d_trans_%s' % (res, res, isTransit))
    # us.PICKLE_SAVING(Genlog, './trainlog/Genlog_%dx%d_trans_%s' % (res, res, isTransit))
    # if res>=16:
    #     us.PICKLE_SAVING(SWD,'SWD_%dx%d_trans_%s'%(res,res,isTransit))

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

    us.MKDIR('ckpt')
    us.MKDIR('structure')
    us.MKDIR('trainlog')

    # progressive growing
    time0 = time.time()  # 开始计时
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
    time1 = time.time()  # 开始计时
    print('全部训练耗费时间：%.2f..'%(time1-time0))







