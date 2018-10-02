import tensorflow as tf   
import numpy as np 
import Dataset
import resnet
import rpn_conv_layers

class Fast_RCNN:
    __current_labels = []
    def ResNet_get_data_test(self):
        test = Dataset.Dataset_cus()
        data_index,self.__current_labels,dimx,dimy,channels = test.next_batch()
        test_Resnet = resnet.Resnet()
        data_index = np.array(data_index)
        data_index = data_index[np.newaxis,:,:,:]
        feature_maps,feature_shape = test_Resnet.build_main_structure(data_index,dimx,dimy,channels)
        rpn_conv_layers.RPN_conv1(feature_maps,feature_shape)

if(__name__=='__main__'):
    model_test = Fast_RCNN()
    model_test.ResNet_get_data_test()
        
