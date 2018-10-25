import tensorflow as tf   
import numpy as np 
import Dataset
import resnet
import rpn_layers as rpn
import roi_proposal_layer

class Fast_RCNN:
    __current_labels = []
    def ResNet_get_data_test(self):
        test = Dataset.Dataset_cus()
        data_index,self.__current_labels,dimx,dimy,shape,channels = test.next_batch()
        data_index = np.array(data_index)
        data_index = data_index[np.newaxis,:,:,:]
        test_Resnet = resnet.Resnet(dimx,dimy,channels)
        feature_maps,feature_shape,strides = test_Resnet.build_main_structure()
        rpn_test = rpn.RPN_layers(feature_maps,feature_shape,self.__current_labels,"train",dimx,dimy,strides)
        roi_test = roi_proposal_layer.roi_proposal_layer(rpn_test,self.__current_labels,shape,"train")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {test_Resnet.x:data_index,test_Resnet.training:False}
            test,feature = sess.run([roi_test.rpn_cls_score,feature_maps],feed_dict=feed_dict)
        print(feature)
if(__name__=='__main__'):
    model_test = Fast_RCNN()
    model_test.ResNet_get_data_test()
        
