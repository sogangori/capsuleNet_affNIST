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

modelName = './weights/cnn_overlap.pd'
AFFIN = True
RECONSTRUCT = True
FREQ = 10
epoch = 1000
BATCH = 600#must even number
REDUCE_DATA_COUNT_RATIO = 1
learning_rate = 1e-3
isNewTrain = not True     
 
def conventionalCNN(x):
    
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.00001)):
        pool = slim.conv2d(x, 32,[3,3],[1,1])
        pool = slim.max_pool2d(pool,[2,2],[2,2])#14x14
        print ('    pool',pool)  

        pool = slim.conv2d(pool, 64,[3,3],[1,1])
        pool = slim.max_pool2d(pool,[2,2],[2,2])#7x7
        print ('    pool',pool)  

        pool = slim.conv2d(pool, 128,[3,3],[1,1])
        pool = slim.max_pool2d(pool,[2,2],[2,2])#3x3
        print ('    pool',pool)  

        pool = slim.conv2d(pool, 10,[3,3],[1,1], padding='VALID')
        
        out = tf.squeeze(pool,[1,2])
        print ('    out',out)        
        
    return out

def main(arg=None):
    
    affNIST_in,affNIST_out = affNIST.load_affNIST()
    mnist = input_data.read_data_sets('/mnist')
    
    print ('affNIST min',np.min(affNIST_in[0]),np.max(affNIST_in[0]))
    print ('  MNIST min',np.min(mnist.train.images[0]),np.max(mnist.train.images[0]))

    trainIn, trainOut = util.skip_no_equal_neighbor(mnist.train.images,mnist.train.labels)
    validIn, validOut = util.skip_no_equal_neighbor(mnist.test.images, mnist.test.labels)    
    affNIST_in,affNIST_out = util.skip_no_equal_neighbor(affNIST_in,affNIST_out)

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
    
    hyperthesis = conventionalCNN(x_overlap)
        
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_overlap, logits=hyperthesis))                
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
            if train: _,ML,acc = sess.run([train_step,loss,accuracy],feed)            
            else : ML,acc = sess.run([loss,accuracy],feed)
            acc_sum += acc/iter
        return acc_sum,ML
        
    for i in range(epoch):
        train_accuracy,ML_tr = feed_all(trainIn, trainOut,train=True, Pad=True)
            
        if i<10 or i % FREQ == 0:
            valid_accuracy,ML_v = feed_all(validIn, validOut,train=False, Pad=True)
            test_accuracy,ML_te = feed_all(affNIST_in,affNIST_out,train=False,Pad=False)
            now = strftime("%H:%M:%S", localtime())
            print('step %d/%d, accuracy train:%.3f valid:%.3f test:%.3f loss:(%.7f) %s' % (i,epoch, train_accuracy,valid_accuracy,test_accuracy,ML_tr,now))

        this_sec = time.time()
        if i==epoch-1 or this_sec - start_sec > 60 * 5 :
            start_sec = this_sec
            save_path = saver.save(sess, modelName)            
            print("Model Saved, time:%s, %s" %(now, save_path))                        

tf.app.run()