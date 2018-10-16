import tensorflow as tf 
import numpy as np
import anchor_target_layer

#using a sliding window to slide on the feature map and ganerate 1-d features
class RPN_layers:
    __feature_vector = []
    def __init__(self,feature_map,feature_shape,gt_boxs,mode,dimx,dimy,strides):
        self.feature_map = feature_map
        self.feature_shape = feature_shape
        self.gt_boxs = gt_boxs
        self.mode = mode
        self.pic_dimx = dimx
        self.pic_dimy = dimy
        self.feat_strides = strides
        self.__build_rpn()

    def __build_rpn(self):
        self.__RPN_conv1()
        self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights =\
            anchor_target_layer.anchor_target_layer(self.__cls_conv(),self.gt_boxs,self.pic_dimx,self.pic_dimy,self.feat_strides)

    def __RPN_conv1(self):
        x = tf.placeholder(tf.float32,[self.feature_shape[0],self.feature_shape[1],self.feature_shape[2],self.feature_shape[3]])
        w1 = tf.Variable(tf.random_normal([3, 3, 1024, 256]))
        L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding="SAME")
        L1 = tf.nn.relu(L1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x:self.feature_map}
            result = sess.run(L1,feed_dict = feed_dict)
        self.__feature_vector = self.__feature_per_slid(result,result.shape[3],result.shape[2],result.shape[1])

    def __feature_per_slid(self,input_data,numx,dimx,dimy):
        input_np = np.array(input_data)
        feature1d = np.array([])
        feature_all = np.zeros(shape = (756,1))
        for n in range(numx):
            for i in range(dimx):
                for j in range(dimy):
                    feature1d = np.hstack((feature1d,input_np[0,j,i,n]))
            feature_all = np.column_stack((feature_all,feature1d))
            feature1d = np.array([])
        
        self.feature_all = feature_all[:,1:]
        return self.feature_all

    def __cls_conv(self):
        with tf.variable_scope("cls"):
            w_cls = tf.Variable(tf.random_normal([256,18],dtype = tf.float64))
            l_cls = tf.matmul(self.__feature_vector,w_cls)
        
        x_cls = tf.placeholder(tf.float32,[None,256])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x_cls:self.__feature_vector}
            result = sess.run(l_cls,feed_dict = feed_dict)

        return result

    def __reg_conv(self):
        with tf.variable_scope("reg"):
            w_cls = tf.Variable(tf.random_normal([256,36],dtype = tf.float64))
            l_cls = tf.matmul(self.__feature_vector,w_cls)

        x_reg = tf.placeholder(tf.float32,[None,256])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x_reg:self.__feature_vector}
            result = sess.run(l_cls,feed_dict = feed_dict)

        return result




    





