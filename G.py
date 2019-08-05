import numpy as np
import tensorflow as tf
import utils

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

meta_path = r'./ckpt/PG_level7_False/network.ckpt-16000.meta'

# 加载模型
meta_graph = tf.train.import_meta_graph(meta_path)

# 设置默认图
graph = tf.get_default_graph()

# 获取相关张量
latents = graph.get_tensor_by_name("latents:0")
fake_SS = graph.get_tensor_by_name("generator/level_7_toRGB/add:0")

# GPU配置
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 程序按需申请内存

with tf.Session() as sess:
    # init
    init = [tf.initialize_all_variables(),tf.local_variables_initializer()]
    sess.run(init)

    meta_graph.restore(sess,tf.train.latest_checkpoint(r'./ckpt/PG_level7_False'))

    z = np.random.normal(size=(25, 512))
    G = (sess.run(fake_SS,feed_dict={latents:z})+1)/2
    utils.CV2_IMSHOW_NHWC_RAMDOM(G,1,25,5,5,'G')

    Wass = utils.PICKLE_LOADING(r'./trainlog/Wass_8x8_trans_True')
    utils.PLT_PLOT(Wass[1])


