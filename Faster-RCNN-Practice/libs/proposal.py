import tensorflow as tf    
import numpy as np
from gen_anchors import generate_anchors


def _rule(base_anchors,num_x,num_y,stride):
    anchors = np.array([np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    x_ctr = stride*num_x
    y_ctr = stride*num_y
    for i in range(9):
        if(x_ctr+base_anchors[i,0]<0):
            x = 0
        if(x_ctr+base_anchors[i,2]>num_x):
            x = num_x
        if(x_ctr+base_anchors[i,1]<0):
            y = 0
        if(x_ctr+base_anchors[i,3]>num_y):
            y = num_y
    
        x = x_ctr
        y = y_ctr
        w = base_anchors[i,2] - base_anchors[i,0]
        h = base_anchors[i,3] - base_anchors[i,1]

        anchor = np.array([x,y,w,h])
        anchors = np.vstack((anchors,anchor))

    anchors = anchors[1:]
    return anchors

def _project_back_to_source(source_dimx, source_dimy, stride):
    base_anchors = generate_anchors()
    source_anchors_list = []
    for i in range(int(source_dimx/stride)):
        for j in range(int(source_dimy/stride)):
            temp = _rule(base_anchors,i,j,stride)
            source_anchors_list.append(temp.tolist())
            
    source_anchors = np.array(source_anchors_list)
    return source_anchors

if(__name__=="__main__"):
    _project_back_to_source(224,224,16)

