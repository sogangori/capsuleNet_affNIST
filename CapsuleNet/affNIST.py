import numpy as np
import scipy.io as spio
from matplotlib import pyplot as plt
from matplotlib import cm

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def visualization(x, y,count,index):
    x = np.reshape(x, (40, 40))
    
    plt.subplot(1,count,index)
    plt.imshow(x, cmap=cm.Greys_r)
    plt.title(y)
    plt.axis('off')   

def visualization_overlap(x0,x1, y0,y1,count,index):
    r = np.reshape(x0, (40, 40))
    g = np.reshape(x1, (40, 40))
    b = np.zeros_like(r)
    rgb = np.stack([r,g,b],-1)
    
    plt.subplot(1,count,index)
    plt.imshow(rgb)
    plt.title('R:('+ str(y0)+','+str(y1)+')')
    plt.axis('off')   

def load_affNIST():
    path = './valid/1.mat'
    dataset = loadmat(path)

    ans_set = dataset['affNISTdata']['label_int']
    train_set = dataset['affNISTdata']['image'].transpose()/255.0
    print ('train_set',train_set.shape)# (10000, 1600)
    print ('ans_set',ans_set.shape)#(10000,)
    return train_set,ans_set

if __name__ == '__main__':
    
    OVERLAP = not True
    count = 10

    train_set,ans_set = load_affNIST()
    print ('min',np.min(train_set[0]),np.max(train_set[0]))
    #train_set_overlap = train_set[0::2]+train_set[1::2]    
    for j in range((int)(len(ans_set)/count)):
        for i in range(count):
            index = np.minimum(j*count+i, len(ans_set)-1)
            if OVERLAP:
                visualization_overlap(train_set[index],train_set[index+1],ans_set[index],ans_set[index+1],count,i+1)
            else: 
                visualization(train_set_overlap[index], ans_set[index],count,i+1)

        plt.show()
