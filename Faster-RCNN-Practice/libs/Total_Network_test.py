import tensorflow as tf   
import numpy as np 
import Dataset
import resnet

class Fast_RCNN:
    __current_labels = []
    def ResNet_get_data_test(self):
        test = Dataset.Dataset_cus()
        data_index,self.__current_labels,dimx,dimy,channels = test.next_batch()
        test_Resnet = resnet.Resnet()
        feature_maps = test_Resnet.build_main_structure(data_index,dimx,dimy,channels)


if(__name__=='__main__'):
    model_test = Fast_RCNN()
    model_test.ResNet_get_data_test()
        
