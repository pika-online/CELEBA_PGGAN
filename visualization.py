"""
Introduction: to display the imgs of dataset,training course and prdiction ,and show losses variation of G,D
Author:Ephemeroptera
Email:605686962@qq.com
Date:2019-5-1
"""

# import dependency
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os

# define one showing using CV2
# RGB or GRAY
def CV2_SHOW(title,img):
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

# define batch showing using CV2
def CV2_BATCH_SHOW(batch,scale,rows,cols,delay = 0):
    batch = np.array(batch)
    assert len(batch.shape)==4 # [N,H,W,C]
    N = batch.shape[0]
    assert N <= rows*cols
    H = batch.shape[1]
    W = batch.shape[2]
    C = batch.shape[3]
    # get resized img shape
    S = cv2.resize(batch[0],(0,0),fx=scale,fy=scale).shape
    # build img sets
    IMG = np.zeros(shape=[rows*S[0],cols*S[1],C])
    # assign
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx == N:
                break
            IMG[r*S[0]:(r+1)*S[0],c*S[1]:(c+1)*S[1],:] = np.reshape(cv2.resize(batch[idx],(0,0),fx=scale,fy=scale),[S[0],S[1],C])
            idx += 1

    IMG = IMG.reshape([IMG.shape[0],IMG.shape[1],C])
    # show
    cv2.namedWindow('show',flags=cv2.WINDOW_NORMAL)
    cv2.imshow('show',mat=IMG)
    cv2.waitKey(delay)

# define batch showing randomly using CV2
def CV2_BATCH_RANDOM_SHOW(batch,scale,N,rows,cols,delay = 0):
    batch = np.array(batch)
    total = batch.shape[0]
    idxes = list(range(total))
    import random
    idx = random.sample(idxes, N)
    rands = batch[idx]
    CV2_BATCH_SHOW(rands,scale,rows,cols,delay)

# define genlog showing using cv2
def CV2_GENLOG_SHOW(genlog,scale,rows,cols,delay=0):
    import random
    M = genlog.shape[0]
    N = genlog.shape[1]
    # lineaer display
    idx_r = np.int32(np.linspace(0, M-1,rows))
    # sampling randomly
    samples = genlog[idx_r[0]]
    idx_c = list(range(N))
    idx_c = random.sample(idx_c, cols)
    samples = samples[idx_c]
    for r in idx_r[1:]:
        samples_r = genlog[r]
        idx_c = list(range(N))
        idx_c = random.sample(idx_c, cols)
        samples_r = samples_r[idx_c]
        samples = np.concatenate([samples,samples_r],axis=0)
    CV2_BATCH_SHOW(samples,scale,rows,cols,delay)
        

# G/D_loss showing using plt
# losses[0]:steps , losses[1]:D , losses[2]:G
def losses_show(path):
    with open(path, 'rb') as l:
        losses = pickle.load(l)
        fig, ax = plt.subplots(figsize=(20, 7))
        plt.plot(losses.T[0],losses.T[1], label='Discriminator  Loss')
        plt.plot(losses.T[0],losses.T[2], label='Generator Loss')
        plt.title("Training Losses")
        plt.legend()
        plt.show()

# showing model paras
def MODEL_PARAS(old_model_path):
    ckpt = tf.train.get_checkpoint_state(old_model_path)
    latest = ckpt.model_checkpoint_path  # get the latest model path
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(latest)  # read latest model
    var_to_shape_map = reader.get_variable_to_shape_map()  # get all vars
    for key in var_to_shape_map.keys(): # display
        tensorName = key
        tensorShape = var_to_shape_map[key]
        print(tensorName,tensorShape)

# example
if __name__ == '__main__':
    mnist_dir = r'.\mnist_dataset'
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(mnist_dir)
    batch = mnist.train.images.reshape([-1,28,28,1])
    CV2_BATCH_RANDOM_SHOW(batch.reshape([-1,28,28,1]),0.8,8,3,3,0)









