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
        self.expose_cls_result
    def __build_rpn(self):
        self.__RPN_conv1()
        self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights =\
            anchor_target_layer.anchor_target_layer(self.__cls_conv(),self.gt_boxs,self.pic_dimx,self.pic_dimy,self.feat_strides)

    def __RPN_conv1(self):
        w1 = tf.Variable(tf.random_normal([3, 3, 512, 256]))
        L1 = tf.nn.conv2d(self.feature_map,w1,strides=[1,1,1,1],padding="SAME")
        L1 = tf.nn.relu(L1)
        self.__shape1 = L1.get_shape()[3]
        self.__shape2 = L1.get_shape()[2]
        self.__shape3 = L1.get_shape()[1]
        L1 = tf.reshape(L1,[-1])
        self.__feature_vector = tf.reshape(L1,[1,1,self.__shape2*self.__shape3,self.__shape1])
        print(self.__feature_vector.get_shape())
        self.__feature_vector = tf.to_double(self.__feature_vector)
        #self.__feature_vector = self.__feature_per_slid(L1,L1.get_shape()[3],L1.get_shape()[2],L1.get_shape()[1])#(result,result.shape[3],result.shape[2],result.shape[1])

    def __cls_conv(self):
        with tf.variable_scope("cls"):
            w_cls = tf.Variable(tf.random_normal([1,1,256,18],dtype = tf.float64))
            
            l_cls = tf.matmul(self.__feature_vector,w_cls)

            l_cls = tf.reshape(l_cls,[1,self.feature_shape[0],self.feature_shape[1],18])
            
            self.expose_cls_result = l_cls
            print("l_cls.get_shape(): ",l_cls.get_shape())
        """x_cls = tf.placeholder(tf.float32,[None,256])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x_cls:self.__feature_vector}
            result = sess.run(l_cls,feed_dict = feed_dict)"""

        return l_cls

    def __reg_conv(self):
        with tf.variable_scope("reg"):
            w_reg = tf.Variable(tf.random_normal([1,1,256,36],dtype = tf.float64))
            l_reg = tf.matmul(self.__feature_vector,w_reg)

            l_reg = tf.reshape(l_reg,[1,self.feature_shape[0],self.feature_shape[1],36])
        """x_reg = tf.placeholder(tf.float32,[None,256])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x_reg:self.__feature_vector}
            result = sess.run(l_cls,feed_dict = feed_dict)"""

        return l_reg

    def get_rpn_cls_score(self):
        return self.__cls_conv()

    def get_rpn_cls_loss(self): 
        rpn_cls_score = self.__cls_conv()
        rpn_labels =self.rpn_labels
    
        return self.__rpn_cls_loss(rpn_cls_score,rpn_labels)
    
    def get_rpn_bbox_pred(self):
        return self.__reg_conv()


    def __rpn_cls_loss(self, rpn_cls_score, rpn_labels):

        shape                     = tf.shape(rpn_cls_score)
        
        # Stack all classification scores into 2D matrix
        rpn_cls_score             = tf.transpose(rpn_cls_score,[0,3,1,2])
        rpn_cls_score             = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
        rpn_cls_score             = tf.transpose(rpn_cls_score,[0,2,3,1])
        rpn_cls_score             = tf.reshape(rpn_cls_score,[-1,2])
        
        # Stack labels
        rpn_labels                = tf.reshape(rpn_labels,[-1])
        
        # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
        rpn_cls_score             = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
        rpn_labels                = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
        # Cross entropy error
        rpn_cross_entropy         = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))
    
        return rpn_cross_entropy   

    def smoothL1( self, x, sigma):
        conditional               = tf.less(tf.abs(x), 1/sigma**2)
        close                     = 0.5* (sigma * 2 ) **2
        far                       = tf.abs(x) - 0.5/sigma ** 2

        return tf.where(conditional, close, far)
        
    def __rpn_bbox_loss(self, rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
        
        rpn_bbox_targets          = tf.transpose( rpn_bbox_targets,   [ 0, 2, 3, 1])
        rpn_inside_weights        = tf.transpose( rpn_inside_weights, [ 0, 2, 3, 1])
        rpn_outside_weights       = tf.transpose( rpn_outside_weights,[ 0, 2, 3, 1]) 
        
        diff                      = tf.multiply( rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
        diff_sL1                  = self.smoothL1(diff,3.0)

        rpn_bbox_reg              = 10*tf.reduce_sum(tf.multiply(rpn_outside_weights, diff_sL1))
        return rpn_bbox_reg

    def get_rpn_reg_loss(self):
        rpn_bbox_score = self.__reg_conv()
        rpn_bbox_targets         = self.rpn_bbox_targets
        rpn_bbox_inside_weights  = self.rpn_bbox_inside_weights
        rpn_bbox_outside_weights = self.rpn_bbox_outside_weights

        return self.__rpn_bbox_loss(rpn_bbox_score,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights)

    





