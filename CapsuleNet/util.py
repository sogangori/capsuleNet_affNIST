import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
from PIL import Image

def skip_no_equal_neighbor(images, labels):
    m = len(images)
    image_list = []
    label_list = []
    image_list.append(images[0])
    label_list.append(labels[0])
    for i in range(1,m):
        if label_list[-1]!=labels[i]:
            image_list.append(images[i])
            label_list.append(labels[i])
    
    images = np.asarray(image_list)
    labels = np.asarray(label_list)
    print ('images',images.shape)
    print ('labels',labels.shape)
    if labels.shape[0]%2==1:
        images = images[:-1]
        labels = labels[:-1]
    print ('images',images.shape)
    print ('labels',labels.shape)
    return images, labels

def padding(x):    
    bz = x.shape[0]
    if np.rank(x)==2:
        wh = x.shape[1]
        w = h = (int)(np.sqrt(wh))
    else:
        w = h = x.shape[1]

    x = np.reshape(x, [-1,h,w])
    bg = np.zeros([bz,40,40])
    max_offset = 40-h
    offsets = np.random.randint(0,max_offset,2)
    #offsets = [6,6]
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w] = x
    bg = np.expand_dims(bg,-1)
    return bg

def removeFiles(folder):
    if os.path.exists(folder):       
        list_path = glob.glob(folder+'/*.png')
        for i in range(len(list_path )):                       
            os.remove(list_path [i])
    else:        
        os.makedirs(folder)    

def save(arr, label,folder,index):
    print ('    arr',arr.shape, np.min(arr), np.mean(arr), np.max(arr),label)
    image_re = Image.fromarray(np.uint8(arr*255))         
    
    folder = folder + str(label)+ '_recon_'
    path = folder+str(index)+'.png'    
    image_re.save(path)     
     