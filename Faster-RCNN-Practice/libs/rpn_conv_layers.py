import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np

#using a sliding window to slide on the feature map and ganerate 1-d features
def RPN_conv1(feature_map):
    w1 = slim.variable(tf.random_normal([3,3,256,256]))
    L1 = tf.nn.conv2d(feature_map,w1,strides=[1,1,1,1],padding="SAME")
    print(L1.shape()[2])
    _feature_per_slide = tf.convert_to_tensor(feature_per_slid(L1,L1.shape()[2],L1.shape()[1],L1.shape()[0]))
    


def feature_per_slid(input_data,numx,dimx,dimy):
    input_np = input_data.eval()
    feature1d = np.array()
    feature_all = np.array()
    for n in range(numx):
        for i in range(dimx):
            for j in range(dimy):
                feature1d = np.hstack((feature1d,input_np[n,i,j]))
        feature_all = np.column_stack((feature_all,feature1d))
    
    return feature_all



    





