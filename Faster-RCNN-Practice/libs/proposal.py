import tensorflow as tf    
import numpy as np
from gen_anchors import generate_anchors


class gan_anchor:
    dimx = 0
    dimy = 0
    def __init__(self,dimx,dimy,stride):
        self.dimx = dimx
        self.dimy = dimy
        self.sorce_anchors = self.__project_back_to_source(dimx,dimy,stride)
    def __rule(self,base_anchors,num_x,num_y,stride):
        anchors = np.array([np.newaxis,np.newaxis,np.newaxis,np.newaxis])
        x_tl = stride*num_x
        y_tl = stride*num_y
        for i in range(9):
            width = base_anchors[i,2]-base_anchors[i,0]
            height = base_anchors[i,3]-base_anchors[i,1]
            if(x_tl+(base_anchors[i,2]-base_anchors[i,0])>self.dimx):
                width = self.dimx
            if(y_tl+(base_anchors[i,3]-base_anchors[i,1])>self.dimy):
                height = self.dimy
        
            x = x_tl
            y = y_tl
            w = width
            h = height
            x1 = x+w
            y1 = y+h
            anchor = np.array([x,y,x1,y1])
            anchors = np.vstack((anchors,anchor))

        anchors = anchors[1:]
        #this list return the top-letf and the w ,h
        return anchors

    def __project_back_to_source(self,source_dimx, source_dimy, stride):
        base_anchors = generate_anchors()
        source_anchors_list = []
        for i in range(int(source_dimx/stride)):
            for j in range(int(source_dimy/stride)):
                temp = self.__rule(base_anchors,i,j,stride)
                source_anchors_list.append(temp.tolist())
                
        source_anchors = np.array(source_anchors_list)
        return source_anchors
    
    def get_anchors(self):
        result = self.sorce_anchors
        return result

if(__name__=="__main__"):
    test = gan_anchor(409,301,16)
    print(test.get_anchors())

