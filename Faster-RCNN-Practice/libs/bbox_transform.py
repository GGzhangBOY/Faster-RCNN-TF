import numpy as np 

def bbox_transform(anchor, gt_box):
    """
    compute the transform that the bbox need to transform to the gt_box
    """
    width = anchor[2]-anchor[0]
    height = anchor[3]-anchor[1]
    tx = (gt_box[0]-anchor[0])/width
    ty = (gt_box[1]-anchor[1])/height
    tw = np.log(gt_box[2]/width)
    th = np.log(gt_box[3]/height)

    result = [tx, ty, tw, th]

    return result