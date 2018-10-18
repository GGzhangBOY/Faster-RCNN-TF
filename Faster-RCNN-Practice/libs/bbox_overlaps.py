import tensorflow as tf 
import numpy as np 

anchor_state = {'positive': 1, 'negtive': 0, 'unknown': -1}

def cal_IOU(anchor, gt_boxes):
    """
    #this function calculate the anchor's overlap with the gt boxs, and return the largest overlap
    """
    anchor = [anchor[0], anchor[1], np.abs(
        anchor[0]-anchor[2]), np.abs(anchor[1]-anchor[3])]
    IOU_list = []
    if(len(np.array(gt_boxes).shape) > 1):
        i = 0
        for gt_box in gt_boxes:
            # 候选框与大框相交
            IOU_list.append(cal_alg(anchor, gt_box))
        pos = IOU_list.index(max(IOU_list))
        result = max(IOU_list)
    else:
        IOU_list.append(cal_alg(anchor, gt_boxes))
        result = max(IOU_list)
        pos = IOU_list.index(max(IOU_list))
    if(result > 0.7):
        return anchor_state['positive'], pos
    elif(result < 0.3):
        return anchor_state['negtive'], pos
    else:
        return anchor_state['unknown'], pos

def cal_alg(Reframe, GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]

    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1, x2+width2)
    startx = min(x1, x2)
    width = np.abs(width1+width2-(endx-startx))  # 相交方框的宽

    endy = max(y1+height1, y2+height2)

    starty = min(y1, y2)

    height = np.abs(height1+height2-(endy-starty))  # 相交方框的高

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0

    else:
        Area = width*height  # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio