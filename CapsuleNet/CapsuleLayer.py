﻿import tensorflow as tf
import tensorflow.contrib.slim as slim

ROUT_COUNT = 3

def test(x):
    
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

        v = util.squash(s)        
        print ('    v',v)

        out = tf.norm(v,axis=-1)
        print ('    out',out)        
        
    return out

def capsnet_forward(x, reuse = False):
    with tf.variable_scope('CapsNet',reuse=reuse):
        wcap = tf.get_variable('wcap',[1,6*6*32,8,10,16],initializer=tf.truncated_normal_initializer(stddev=0.02))

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, padding='VALID', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        
        conv1 = slim.conv2d(x, 256,[9,9],[1,1])
        print ('    conv1',conv1)        
        u = slim.conv2d(conv1, 8*32,[9,9],[2,2])
        print ('    u',u)
        u = tf.reshape(u,[-1,6*6*32,8,1,1])
                
        uw = u*wcap
        print ('    uw',uw)#(?, 6*6*32, 8, 10, 16)        

        u_ = tf.reduce_sum(uw, axis=2)
        print ('    u_',u_)#(10, 1152, 10, 16)
        b = tf.zeros([1,6*6*32,10,1],tf.float32)
        
        for i in range(ROUT_COUNT):            
            c = tf.nn.softmax(b, dim=2)
            c = tf.stop_gradient(c)
            
            print (i,'    c',c)#(10, 1152, 10, 1)
                        
            cu = c * u_
            print (i,'    cu',cu)#(?, 1152,10, 16)

            s = tf.reduce_sum(cu, axis=1, keep_dims=True)#(?,1, 10, 16)
            print ('    s',s)#(?, 1, 10, 16),

            v = squash(s)#(?, 1, 10, 16)
            print ('    v',v)

            a = tf.reduce_sum(u_*v,axis=-1,keep_dims=True)#(?, 1152, 10, 1) 
            print (i,'    a',a)

            b += tf.reduce_sum(a,axis=-1,keep_dims=True)            
            print (i,'    b+',b)#(?, 1152, 10, 1)
        
        print ('    v squeeze before',v)#(?, 1, 10, 16)
        v = tf.squeeze(v,axis=1)
        print ('    v squeeze after',v)#(?, 10, 16)
                
    return v

def reconstruct(DigitCaps,mask):
    with tf.variable_scope('reconstruct',reuse=tf.AUTO_REUSE):
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

def squash(s, axis=-1):
    length_s = tf.sqrt(tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True))
    v = tf.square(length_s)/(1+tf.square(length_s)) * (s/length_s)    
    return v

def margin_loss(Y,H):
    pos_loss_p = Y*tf.square(tf.maximum(0.0, 0.9 - H))
    pos_loss_n = (1-Y)*tf.square(tf.maximum(0.0, H - 0.1 ))    
    margin_loss = tf.reduce_mean(pos_loss_p + 0.5 * pos_loss_n)
    return margin_loss
