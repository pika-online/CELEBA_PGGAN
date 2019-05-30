# 导入包
import tensorflow as tf
import numpy as np
import os

"""
    介绍：
    tfrecord格式是tensorflow官方推荐的数据格式，把数据、标签进行统一的存储
    tfrecord文件包含了tf.train.Example 协议缓冲区(protocol buffer，协议缓冲区包含了特征 Features)， 能让tensorflow更好的利用内存。

    Author:Ephemerptero
    Version:1.4.0
    Date:2019-5-24
    QQ:605686962
"""

"""
# 定义生成整数型,字符串型和浮点型属性的方法，这是将数据填入到Example协议
# 内存块(protocol buffer)的第一步，以后会调用到这个方法
"""

def Int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def Bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def Float_frature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# to array
def to_array(LIST):
    return [np.array(each) for each in LIST]

# 检查目录是否存在
def check_dir(path):
    dir,fn = os.path.split(path)
    if not os.path.isdir(dir):
        print('文件夹%s未创建，现在在当前目录下创建..' % (dir))
        os.mkdir(dir)

# 获取分段下标
def get_seg_index(end,segs,start=0):
    return [int(i) for i in np.linspace(start, end, segs + 1)]

# 获取分段序号项
def get_seg_idx(start,end):
    return np.arange(start,end,dtype=np.uint32)

# 大数据集分批存储
def Saving_Batch_TFR(path,idxs,batchs,labels,current,total):
    # 转化为数组
    batch = np.array(batchs)
    label = np.array(labels)

    # 检查目录
    check_dir(path)

    # 保存tfr
    tfr_name = path+'-%.4d-%.4d'%(current,total)
    writer = tf.python_io.TFRecordWriter(tfr_name)
    for idx,date,label in zip(idxs,batchs,labels):
        data_raw = date.tostring()
        label_raw = label.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'idx': Int64_feature(idx),
            'data': Bytes_feature(data_raw),
            'label': Bytes_feature(label_raw)
        }))
        writer.write(example.SerializeToString())

    print('成功写入至第%d/%d个文件，序号项类型：%s，数据项类型：%s ， 标签项类型：%s' % (current+1,total+1,idx.dtype,
                                                           date.dtype,label.dtype))
    writer.close()

    # 全部存储完毕
    if current==total:
        fileSets = os.listdir(os.path.split(path)[0])
        print('成功存储为TFRecord格式！！，%s文件夹下生成文件如下：\n' % (os.path.split(path)[0]))
        print(fileSets)


# 存储全部数据
def Saving_All_TFR(path, datas, labels ,batchs):
    # 序列化
    [datas,labels] = to_array([datas,labels])
    # 数量
    total = datas.shape[0]
    # 分段区间
    index = get_seg_index(total,batchs)
    # 分批保存
    for i in range(batchs):
        seg = [index[i],index[i+1]]
        idx = get_seg_idx(seg[0],seg[1]-1)
        data = datas[seg[0]:seg[1]]
        label = labels[seg[0]:seg[1]]
        Saving_Batch_TFR(path,idx,data,label,i,batchs-1)



def Reading_TFR(sameName, isShuffle, datatype, labeltype):
    # 生成文件列表
    fileslist = tf.train.match_filenames_once(sameName)

    # 由文件列表生成文件队列
    filename_queue = tf.train.string_input_producer(fileslist, shuffle=isShuffle)

    # 实例化TFRecordReader类，读取每个样本
    reader = tf.TFRecordReader()

    # 序列化入内存
    _, serialization = reader.read(filename_queue)

    # 解析样本
    features = tf.parse_single_example(serialization,
        features={
            "idx": tf.FixedLenFeature([], tf.int64),    # 序号
            "data": tf.FixedLenFeature([], tf.string),  # 数据内容
            "label": tf.FixedLenFeature([], tf.string)   # 标签内容
            ## 解析其他属性
        })

    # decode_raw()字符信息解码
    idx = tf.cast(features["idx"], tf.int32)
    data = tf.decode_raw(features['data'], datatype)
    label = tf.decode_raw(features['label'], labeltype)
    # int64类型可以用tf.cast()转换成其他类型

    return [idx,data,label]


def Reading_Batch_TFR(idx,data,label,isShuffle, batchSize,data_size,label_size):

    # 设置batch属性
    min_after_dequeue = 3 * batchSize  # 队列中至少保留个数，否则等待
    capacity = 5 * batchSize  # 队列最大容量
    data.set_shape(data_size)
    label.set_shape(label_size)

    # 是否打乱
    if isShuffle:
        return tf.train.shuffle_batch([idx, data,label], batch_size=batchSize,capacity=capacity,
                                      min_after_dequeue=min_after_dequeue)
    else:
        return tf.train.batch([idx, data,label], batch_size=batchSize,capacity=capacity)

# 测试
if __name__=='__main__':

    # 导入opencv
    import cv2
    def show(img):
        cv2.imshow(' ',img)
        cv2.waitKey(0)

    # 导入MNISI数据
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(r'I:\MNIST', one_hot=True)

    # 读取mnist数据。
    images = mnist.train.images.reshape([-1,28,28,1])  # 55000x28x28x1
    labels = mnist.train.labels  # 55000x10

    # 设置存储路径，注意：TFR是文件夹，MNIST是文件名，TFR文件为二进制无后缀名
    savepath = r'.\TFR\MNIST'

    # 分批保存
    Saving_All_TFR(savepath,images,labels,5)

    # 读取单个tfr
    [idx,data,label] = Reading_TFR(r'./TFR/MNIST-*',isShuffle=False,datatype=tf.float32,labeltype=tf.float64)

    # 读取成批tfr
    [idx_batch,data_batch,label_batch] = Reading_Batch_TFR(idx,data,label,isShuffle=False,batchSize=10,data_size=28*28,
                                                           label_size=10)

    with tf.Session() as sess:
        # 初始化变量
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # 开启协调器
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 测试
        # TDB = sess.run([idx,data,label])
        # TDB2 = sess.run([idx,data,label])
        TDB = sess.run([idx_batch,data_batch,label_batch])
        for i in range(10):
            print('idx:',TDB[0][i],'label:',TDB[2][i])
            show(TDB[1][i].reshape([28,28,1]))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


