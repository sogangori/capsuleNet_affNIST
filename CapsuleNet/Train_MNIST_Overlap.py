from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
from time import localtime, strftime
from tensorflow.examples.tutorials.mnist import input_data
import affNIST
import util
import CapsuleLayer

modelName = './weights/caps_overlap.pd'
AFFIN = True
RECONSTRUCT = True
FREQ = 10
epoch = 100
BATCH = 300#must even number
REDUCE_DATA_COUNT_RATIO = 1
learning_rate = 1e-3
isNewTrain = not True     

def shuffle_no_equal_neighbor(images, labels):
    m = len(images)
    image_list = []
    label_list = []
    image_list.append(images[0])
    label_list.append(labels[0])
    for i in range(1,m):
        if labels[i-1]!=labels[i]:
            image_list.append(images[i])
            label_list.append(labels[i])
    
    images = np.asarray(image_list)
    labels = np.asarray(label_list)
    print ('images',images.shape)
    print ('labels',labels.shape)
    if len(image_list)%2==1:
        images = images[:-1]
        labels = labels[:-1]
    return images, labels

def main(arg=None):
    
    affNIST_in,affNIST_out = affNIST.load_affNIST()
    mnist = input_data.read_data_sets('/mnist')
    
    print ('affNIST min',np.min(affNIST_in[0]),np.max(affNIST_in[0]))
    print ('  MNIST min',np.min(mnist.train.images[0]),np.max(mnist.train.images[0]))

    trainIn, trainOut = shuffle_no_equal_neighbor(mnist.train.images,mnist.train.labels)
    validIn, validOut = shuffle_no_equal_neighbor(mnist.test.images, mnist.test.labels)    
    affNIST_in,affNIST_out = shuffle_no_equal_neighbor(affNIST_in,affNIST_out)

    h = w = 28
    if AFFIN: h = w = 40

    X = tf.placeholder(tf.float32, [None, None,None,1])
    Y = tf.placeholder(tf.float32, [None])

    y_int = tf.cast(Y, tf.int32)    
    Y_ONE_HOT = tf.one_hot(y_int,10)

    x_resize = tf.image.resize_bilinear(X, [28, 28])
    x_overlap = tf.clip_by_value(x_resize[0::2]+x_resize[1::2],0,1)

    y_0 = Y_ONE_HOT[0::2]
    y_1 = Y_ONE_HOT[1::2]
    y_overlap = y_0+y_1
    y_overlap = tf.clip_by_value(y_overlap,0,1)
    
    DigitCaps = CapsuleLayer.capsnet_forward(x_overlap)
    hyperthesis = tf.norm(DigitCaps, ord=2, axis=-1)
           
    recon_x_0 = CapsuleLayer.reconstruct(DigitCaps,y_0)
    recon_x_1 = CapsuleLayer.reconstruct(DigitCaps,y_1)
    recon_x = tf.clip_by_value(recon_x_0 + recon_x_1,0,1)
        
    margin_loss = CapsuleLayer.margin_loss(y_overlap,hyperthesis)    
    restruc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_overlap-recon_x), axis=[1,2]))
    loss = margin_loss
    if RECONSTRUCT: loss += 5e-5 * restruc_loss
        
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    top_values, top_predict = tf.nn.top_k(hyperthesis,2)
    
    y_gt = tf.stack([y_int[0::2],y_int[1::2]], 1)
        
    predict_sort = tf.py_func(np.sort, [top_predict], tf.int32)
    y_gt_sort = tf.py_func(np.sort, [y_gt], tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_sort, y_gt_sort), tf.float32))

    sess = tf.Session()
    saver = tf.train.Saver()
    if isNewTrain: 
        sess.run(tf.global_variables_initializer())
        print('Initialized!')
    else :        
        saver.restore(sess, modelName)
        print("Model restored")

    start_sec = time.time()
    print ('    train:%d, valid:%d, test:%d, REDUCE_DATA_COUNT_RATIO:%d' %( len(mnist.train.images),len(affNIST_in),len(mnist.test.images),REDUCE_DATA_COUNT_RATIO))

    def feed_all(x, y, train=False, Pad=False):
        m = (int)(len(y)/REDUCE_DATA_COUNT_RATIO)
        iter = (int)((m-1)/BATCH+1)
        acc_sum = np.zeros((1), np.float)
        for i in range(iter):
            start = i * BATCH
            end =  np.minimum(start + BATCH, m)
            batch_x = x[start:end]
            if Pad: batch_x = util.padding(batch_x)
            else: batch_x = np.reshape(batch_x, [-1,h,w,1])
            feed = {X:batch_x , Y: y[start:end]}    
            #equalRatio = np.mean(np.equal(y[::2], y[1::2]))
            #print (i,'equalRatio ',equalRatio )
            if train: _,ML,RL,acc = sess.run([train_step,margin_loss,restruc_loss,accuracy],feed)            
            else : ML,RL,acc = sess.run([margin_loss,restruc_loss,accuracy],feed)
            acc_sum += acc/iter
        return acc_sum,ML,RL
        
    for i in range(epoch):
        train_accuracy,ML_tr,RL_tr = feed_all(trainIn, trainOut,train=True, Pad=True)
            
        if i<10 or i % FREQ == 0:
            valid_accuracy,ML_v,RL_v = feed_all(validIn, validOut,train=False, Pad=True)
            test_accuracy,ML_te,RL_te = feed_all(affNIST_in,affNIST_out,train=False,Pad=False)
            now = strftime("%H:%M:%S", localtime())
            print('step %d/%d, accuracy train:%.3f valid:%.3f test:%.3f loss:(%.7f, %.4f) %s' % (i,epoch, train_accuracy,valid_accuracy,test_accuracy,ML_tr,RL_tr,now))

        this_sec = time.time()
        if i==epoch-0 or this_sec - start_sec > 60 * 5 :
            start_sec = this_sec
            save_path = saver.save(sess, modelName)            
            print("Model Saved, time:%s, %s" %(now, save_path))         
               
    for i in range(10):
        start = i 
        end =  start + 2
        batch_x = mnist.train.images[start:end]        
        batch_x = util.padding(batch_x)                
        batch_y = mnist.train.labels[start:end]
        feed = {X:batch_x , Y: batch_y}    
        acc,x_overlap_in, recon_0,recon_1, ori_arr,y_gt_out,predict2 = sess.run([accuracy,x_overlap,recon_x_0,recon_x_1,x_resize,y_gt,top_predict],feed) 
        print ('ori_arr',ori_arr.shape)
        print ('recon_0',recon_0.shape)
        print('y_gt_out',y_gt_out)       
                
        in_rgb = np.stack([x_overlap_in[0],x_overlap_in[0],x_overlap_in[0]],2)

        r = ori_arr[0]
        g = ori_arr[1]
        b = np.zeros_like(r)
        ori_rgb = np.stack([r,g,b],2)
        
        r = recon_0[0]        
        g = recon_1[0]        
        recon_rgb = np.stack([r,g,b],2)

        dual_image = np.stack([in_rgb,ori_rgb,recon_rgb])
        print ('dual_image ',dual_image.shape )
        recon_image = np.reshape(dual_image,[28*3,28,3])
        util.save(recon_image,y_gt_out,'./reconstruct/',predict2)
    save_path = saver.save(sess, modelName) 

tf.app.run()