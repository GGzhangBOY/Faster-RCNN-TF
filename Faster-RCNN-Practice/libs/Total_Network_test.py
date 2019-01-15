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
        data_index,self.__current_labels,gt_boxs,dimx,dimy,channels = test.next_batch()
        test_Resnet = resnet.Resnet()
        data_index = np.array(data_index)
        data_index = data_index[np.newaxis,:,:,:]
        feature_maps,feature_shape,strides = test_Resnet.build_main_structure(data_index,dimx,dimy,channels)
        rpn_test = rpn.RPN_layers(feature_maps,feature_shape,self.__current_labels,"train",dimx,dimy,strides)
        rpn_proposal = roi_proposal_layer.roi_proposal_layer(rpn_test,data_index,)

if(__name__=='__main__'):
    model_test = Fast_RCNN()
    model_test.ResNet_get_data_test()
        
