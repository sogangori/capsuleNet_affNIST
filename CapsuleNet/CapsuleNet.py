#reference https://github.com/laodar/tf_CapsNet/blob/master/CapsNet.py
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

modelName = './weights/caps.pd'
AFFIN = True
RECONSTRUCT = True
ROUT_COUNT = 3
FREQ = 2
epoch = 500
BATCH = 50
REDUCE_DATA_COUNT_RATIO = 1
learning_rate = 1e-3
isNewTrain =  True     

def forward(x):
    
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, padding='VALID', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        conv1 = slim.conv2d(x, 256,[9,9],[1,1])
        print ('    conv1',conv1)        
        caps0 = slim.conv2d(conv1, 8*32,[9,9],[2,2])
        print ('    caps0',caps0)        
        caps0_2d = tf.reshape(caps0,[-1,6*6*8*32])
        print ('    caps0_2d',caps0_2d)        

        cap1 = slim.fully_connected(caps0_2d,10*16)
        cap1 = tf.reshape(cap1,[-1,10,16])
        print ('    cap1',cap1)        

        out = tf.norm(cap1,axis=-1)
        print ('    out',out)        
        
    return out

def padding(x,max_offset=12):
    x = np.reshape(x, [-1,28,28])
    bz,h,w = x.shape
    bg = np.zeros([bz,40,40])
    offsets = np.random.randint(0,max_offset,2)
    #offsets = [6,6]
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w] = x
    bg = np.expand_dims(bg,-1)
    return bg

def squash(s, axis=-1):
    length_s = tf.reduce_sum(s ** 2.0, axis=axis, keep_dims=True) ** 0.5
    v = s * length_s / (1.0 + length_s ** 2.0)
    return v

def capsNetBasic(x, reuse = False):
    with tf.variable_scope('CapsNet',reuse=reuse):
        wcap = tf.get_variable('wcap',[1,6,6,32,8,10,16],initializer=tf.truncated_normal_initializer(stddev=0.02))

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, padding='VALID', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        conv1 = slim.conv2d(x, 256,[9,9],[1,1])
        print ('    conv1',conv1)        
        u = slim.conv2d(conv1, 8*32,[9,9],[2,2])
        print ('    u',u)
        u = tf.reshape(u,[-1,6,6,32,8,1,1])
        
        u_ = u*wcap
        print ('    u_',u_)#(?, 6, 6, 32, 8, 10, 16)        

        u_ = tf.reshape(u_,[-1,6*6*32,8,10,16])
        print ('    u_',u_)#(?, 1152, 8, 10, 16)

        u_hat = tf.reduce_sum(u_, axis=[2])
        print ('    u_hat',u_hat)#(?, 1152, 10, 16)
        
        s = tf.reduce_sum(u_hat, axis=1)
        print ('    s',s)

        v = squash(s)        
        print ('    v',v)

        out = tf.norm(v,axis=-1)
        print ('    out',out)        
        
    return out

def capsNet(x, reuse = False):
    with tf.variable_scope('CapsNet',reuse=reuse):
        wcap = tf.get_variable('wcap',[1,6*6*32,8,10,16],initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('coupling_coefficient_logits',[1,6*6*32,1,10,1],initializer=tf.constant_initializer(0.0))

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, padding='VALID', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        
        conv1 = slim.conv2d(x, 256,[9,9],[1,1])
        print ('    conv1',conv1)        
        u = slim.conv2d(conv1, 8*32,[9,9],[2,2])
        print ('    u',u)
        u = tf.reshape(u,[-1,6*6*32,8,1,1])
                
        uw = u*wcap
        print ('    uw',uw)#(?, 6, 6, 32, 8, 10, 16)        

        u_ = tf.reduce_sum(uw, axis=[2],keep_dims=True)
        print ('    u_',u_)#(10, 1152, 1, 10, 16)
        
        for i in range(ROUT_COUNT):            
            c = tf.stop_gradient(tf.nn.softmax(b, dim=3))
            #c = tf.nn.softmax(b, dim=3)
            print (i,'    c',c)#(10, 1152, 1, 10, 1)
                        
            uc = u_ * c
            print (i,'    uc',uc)#(?, 1152, 8, 10, 16)

            s = tf.reduce_sum(uc, axis=[1], keep_dims=True)#(?, 10, 16)
            print ('    s',s)#(10, 1, 1, 10, 16),

            v = squash(s)#(?, 10, 1, 1, 16)
            print ('    v',v)

            a = tf.reduce_sum(u_*v,axis=-1,keep_dims=True)#(?, 1152, 1, 10, 1) 
            print (i,'    a',a)

            b += tf.reduce_sum(a,axis=-1,keep_dims=True)            
            print (i,'    b+',b)#(10, 1152, 1, 10, 1)
        
        print ('    v squeeze before',v)
        v = tf.squeeze(v,axis=[1,2])
        print ('    v squeeze after',v)
                
    return v

def reconstruct(DigitCaps,mask):
    print ('    DigitCaps',DigitCaps)
    print ('    mask',mask)
    y_m = tf.expand_dims(mask,-1) * DigitCaps 
    print ('    y_m',y_m)#(?, 10, 16),

    flat = slim.flatten(y_m)
    print ('    flat',flat)
    fc = slim.fully_connected(flat,512,activation_fn=tf.nn.relu)
    fc = slim.fully_connected(fc,1024,activation_fn=tf.nn.relu)
    fc = slim.fully_connected(fc,28*28, activation_fn=tf.nn.sigmoid)
    fc = tf.reshape(fc, [-1,28,28,1])
    print ('    fc',fc)
    return fc

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
    DigitCaps = capsNet(x_4d)
    hyperthesis = tf.norm(DigitCaps, ord=2, axis=-1)#(?, 10)
    print ('    hyperthesis',hyperthesis)     
       
    recon_x = reconstruct(DigitCaps,Y_ONE_HOT)
        
    pos_loss_p = tf.reduce_mean(Y_ONE_HOT*tf.square(tf.maximum(0.0, 0.9 - hyperthesis)))
    pos_loss_n = tf.reduce_mean((1-Y_ONE_HOT)*tf.square(tf.maximum(0.0, hyperthesis - 0.1 )))
    print ('    Y_ONE_HOT',Y_ONE_HOT)
    print ('    pos_loss_p',pos_loss_p)
    print ('    pos_loss_n',pos_loss_n)
    margin_loss = pos_loss_p + 0.5 * pos_loss_n
    restruc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_4d-recon_x), axis=[1,2]))
    loss = margin_loss
    if RECONSTRUCT: loss += 5e-5 * restruc_loss
    
    train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss)

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
                if Pad: batch_x = padding(batch_x)
                else: batch_x = np.reshape(batch_x, [-1,h,w,1])
                feed = {X:batch_x , Y: y[start:end]}    
                if train: _,ML,RL,acc = sess.run([train_step,margin_loss,restruc_loss,accuracy],feed)            
                else : ML,RL,acc = sess.run([margin_loss,restruc_loss,accuracy],feed)
                acc_sum += acc/iter
            return acc_sum,ML,RL
        
        for i in range(epoch):
            train_accuracy,ML_tr,RL_tr = feed_all(mnist.train.images, mnist.train.labels,train=True, Pad=True)
            
            if i<20 or i % FREQ == 0:
                valid_accuracy,ML_v,RL_v = feed_all(mnist.test.images, mnist.test.labels,train=False, Pad=True)
                test_accuracy,ML_te,RL_te = feed_all(affNIST_in,affNIST_out,train=False,Pad=False)
                now = strftime("%H:%M:%S", localtime())
                print('step %d/%d, accuracy train:%.3f valid:%.3f test:%.3f loss:(%.7f, %.4f) %s' % (i,epoch, train_accuracy,valid_accuracy,test_accuracy,ML_tr,RL_tr,now))

            this_sec = time.time()
            if i==epoch-1 or this_sec - start_sec > 60 * 5 :
                start_sec = this_sec
                save_path = saver.save(sess, modelName)            
                print("Model Saved, time:%s, %s" %(now, save_path)) 
        
        for i in range(10):
            start = i 
            end =  start + 1
            batch_x = mnist.train.images[start:end]        
            batch_x = padding(batch_x)                
            batch_y = mnist.train.labels[start:end]
            feed = {X:batch_x , Y: batch_y}    
            acc,recon_arr, ori_arr = sess.run([accuracy,recon_x,x_4d],feed) 
            dual_image = np.stack([ori_arr,recon_arr])
            recon_image = np.reshape(dual_image,[28*2,28])
            save(recon_image,batch_y,i)

def save(arr, label,index):
    print ('    arr',arr.shape, np.min(arr), np.mean(arr), np.max(arr),label)
    image_re = Image.fromarray(np.uint8(arr*255))         
    folder = './reconstruct/' + str(label)+ '_recon_'
    path = folder+str(index)+'.png'    
    #if not os.path.exists(folder):os.makedirs(folder)
    image_re.save(path)     
     
tf.app.run()