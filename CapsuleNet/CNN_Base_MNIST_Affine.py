from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
from PIL import Image
from time import localtime, strftime
from tensorflow.examples.tutorials.mnist import input_data
import affNIST
import util

modelName = './weights/baseCNN.pd'

FREQ = 2
epoch = 2
BATCH = 200
REDUCE_DATA_COUNT_RATIO = 100
learning_rate = 1e-3
isNewTrain =  True     

def baseCNN(x,isTrain=False):
    
    with slim.arg_scope([slim.conv2d], normalizer_fn=None, padding='VALID', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        pool = slim.conv2d(x, 255,[5,5],[1,1])        
        pool = slim.conv2d(pool, 255,[5,5],[1,1])        
        pool = slim.conv2d(pool, 128,[5,5],[1,1])#(?,16,16,128)
        print ('    pool',pool)  
        fc = slim.flatten(pool) #(?,32768)
        print ('    fc',fc)  
        fc  = slim.fully_connected(fc, 328)
        fc  = slim.fully_connected(fc, 192)
        fc  = slim.dropout(fc, 0.8, is_training = isTrain )
        out = slim.fully_connected(fc, 10)
        print ('    out',out)        
        
    return out

def main(arg=None):
    
    affNIST_in,affNIST_out = affNIST.load_affNIST()
    mnist = input_data.read_data_sets('/mnist')
    
    print ('affNIST min',np.min(affNIST_in[0]),np.max(affNIST_in[0]))
    print ('  MNIST min',np.min(mnist.train.images[0]),np.max(mnist.train.images[0]))
        
    h = w = 40

    X = tf.placeholder(tf.float32, [None, None,None,1])
    Y = tf.placeholder(tf.float32, [None])
    D = tf.placeholder(tf.bool)
    y_int = tf.cast(Y, tf.int32)
    Y_ONE_HOT = tf.one_hot(y_int,10)

    x_4d = tf.image.resize_bilinear(X, [28, 28])
    hyperthesis = baseCNN(x_4d,D)    
    print ('    hyperthesis',hyperthesis)     
        
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_int, logits=hyperthesis))        
    train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predict = tf.cast(tf.argmax(hyperthesis, 1),tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_int), tf.float32))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if isNewTrain: 
            tf.global_variables_initializer().run()    
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
                feed = {X:batch_x , Y: y[start:end],D:train}    
                if train: _,ML,acc = sess.run([train_step,loss,accuracy],feed)            
                else : ML,acc = sess.run([loss,accuracy],feed)
                acc_sum += acc/iter
            return acc_sum,ML
        
        for i in range(epoch):
            train_accuracy,ML_tr = feed_all(mnist.train.images, mnist.train.labels,train=True, Pad=True)
            
            if i<20 or i % FREQ == 0:
                valid_accuracy,ML_v = feed_all(mnist.test.images, mnist.test.labels,train=False, Pad=True)
                test_accuracy,ML_te = feed_all(affNIST_in,affNIST_out,train=False,Pad=False)
                now = strftime("%H:%M:%S", localtime())
                print('step %d/%d, accuracy train:%.3f valid:%.3f test:%.3f loss:(%.7f) %s' % (i,epoch, train_accuracy,valid_accuracy,test_accuracy,ML_tr,now))

            this_sec = time.time()
            if i==epoch-1 or this_sec - start_sec > 60 * 5 :
                start_sec = this_sec
                save_path = saver.save(sess, modelName)            
                print("Model Saved, time:%s, %s" %(now, save_path)) 
        
      
     
tf.app.run()