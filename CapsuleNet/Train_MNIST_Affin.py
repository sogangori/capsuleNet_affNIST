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

modelName = './weights/caps.pd'
AFFIN = True
RECONSTRUCT = True
FREQ = 10
epoch = 500
BATCH = 200
REDUCE_DATA_COUNT_RATIO = 100
learning_rate = 1e-3
isNewTrain =  True     

def main(arg=None):
    
    affNIST_in,affNIST_out = affNIST.load_affNIST()
    mnist = input_data.read_data_sets('/mnist')
    
    print ('affNIST min',np.min(affNIST_in[0]),np.max(affNIST_in[0]))
    print ('  MNIST min',np.min(mnist.train.images[0]),np.max(mnist.train.images[0]))

    h = w = 28
    if AFFIN: h = w = 40

    X = tf.placeholder(tf.float32, [None, None,None,1])
    Y = tf.placeholder(tf.float32, [None])
    y_int = tf.cast(Y, tf.int32)
    Y_ONE_HOT = tf.one_hot(y_int,10)

    x_4d = tf.image.resize_bilinear(X, [28, 28])
    DigitCaps = CapsuleLayer.capsnet_forward(x_4d)
    hyperthesis = tf.norm(DigitCaps, ord=2, axis=-1)#(?, 10)
    print ('    hyperthesis',hyperthesis)     
           
    recon_x = CapsuleLayer.reconstruct(DigitCaps,Y_ONE_HOT)    
        
    pos_loss_p = tf.reduce_mean(Y_ONE_HOT*tf.square(tf.maximum(0.0, 0.9 - hyperthesis)))
    pos_loss_n = tf.reduce_mean((1-Y_ONE_HOT)*tf.square(tf.maximum(0.0, hyperthesis - 0.1 )))
    
    margin_loss = pos_loss_p + 0.5 * pos_loss_n
    restruc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_4d-recon_x), axis=[1,2]))
    loss = margin_loss
    if RECONSTRUCT: loss += 5e-5 * restruc_loss
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    predict = tf.cast(tf.argmax(hyperthesis, 1),tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_int), tf.float32))

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
        acc_sum = np.zeros_like((1), np.float)
        for i in range(iter):
            start = i * BATCH
            end =  np.minimum(start + BATCH, m)
            batch_x = x[start:end]
            if Pad: batch_x = util.padding(batch_x)
            else: batch_x = np.reshape(batch_x, [-1,h,w,1])
            feed = {X:batch_x , Y: y[start:end]}    
            if train: _,ML,RL,acc = sess.run([train_step,margin_loss,restruc_loss,accuracy],feed)            
            else : ML,RL,acc = sess.run([margin_loss,restruc_loss,accuracy],feed)
            acc_sum += acc/iter
        return acc_sum,ML,RL
        
    for i in range(epoch):
        train_accuracy,ML_tr,RL_tr = feed_all(mnist.train.images, mnist.train.labels,train=True, Pad=True)
            
        if i<10 or i % FREQ == 0:
            valid_accuracy,ML_v,RL_v = feed_all(mnist.test.images, mnist.test.labels,train=False, Pad=True)
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
        end =  start + 1
        batch_x = mnist.train.images[start:end]        
        batch_x = util.padding(batch_x)                
        batch_y = mnist.train.labels[start:end]
        feed = {X:batch_x , Y: batch_y}    
        acc,recon_arr, ori_arr = sess.run([accuracy,recon_x,x_4d],feed) 
        dual_image = np.stack([ori_arr,recon_arr])
        recon_image = np.reshape(dual_image,[28*2,28])
        util.save(recon_image,batch_y,'./reconstruct/',i)
    save_path = saver.save(sess, modelName)            

tf.app.run()