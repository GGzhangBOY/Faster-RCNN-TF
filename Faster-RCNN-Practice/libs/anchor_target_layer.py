import copy
import random

import numpy as np
import tensorflow as tf

import gen_anchors
import proposal

anchor_state = {'positive': 1, 'negtive': 0, 'unknown': -1}


class anchor_label:
    def __init__(self, anchor, label, pos):
        self.anchor = anchor
        self.label = label
        self.pos = pos


def anchor_target_layer(rpn_cls_score, gt_boxes, dimx, dimy, feat_strides=16, anchor_scales=2**np.arange(2, 6)):
    """rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
    tf.py_func( anchor_target_layer_python,[rpn_cls_score, gt_boxes, dimx, dimy, feat_strides, anchor_scales],[tf.float32, tf.float32, tf.float32, tf.float32])"""

    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer_python(
            rpn_cls_score, gt_boxes, dimx, dimy, feat_strides, anchor_scales)
    rpn_labels = tf.convert_to_tensor(
        tf.cast(rpn_labels, tf.int32), name="rpn_labels")
    rpn_bbox_targets = tf.convert_to_tensor(
        rpn_bbox_targets, name="rpn_bbox_targets")
    rpn_bbox_inside_weights = tf.convert_to_tensor(
        rpn_bbox_inside_weights, name="rpn_bbox_inside_weights")
    rpn_bbox_outside_weights = tf.convert_to_tensor(
        rpn_bbox_outside_weights, name="rpn_bbox_outside_weights")

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def anchor_target_layer_python(rpn_cls_score, gt_boxes, dimx, dimy, feat_strides=16, anchor_scales=2**np.arange(2, 6)):
    """
    #this function is used to filter the out of range anchor and label each anchor for whether it's fg or bg
    """
    anchor_geter = proposal.gan_anchor(dimx, dimy, feat_strides)
    anchor_scales = np.array(anchor_scales)
    anchors = anchor_geter.get_anchors()
    num_anchors = 9
    total_anchors = anchors.shape[0]*9

    height = dimx
    width = dimy
    """
    #generate anchors and drop those out of range, for those servived, calculate the IOU and label each box
    """
    canadidate = anchors.tolist()
    ins_anchors = []
    labels = []
    bbox_inside_weights = []
    bbox_outside_weights = []
    for i in range(len(canadidate)):
        for j in range(len(canadidate[i])):
            anchor = canadidate[i][j]
            if((anchor[0] > dimx) | (anchor[1] > dimy)):
                continue
            if(np.abs(anchor[0]+anchor[2]) > dimx):
                continue
            if(np.abs(anchor[1]+anchor[3]) > dimy):
                continue
        # calculate the overlap size,and judge the labels
            current_label, pos = cal_IOU(anchor, gt_boxes)
            labels.append(current_label)
            if(current_label == 1):
                ins_anchors.append(anchor_label(anchor, current_label, pos))
            if(current_label == 0):
                ins_anchors.append(anchor_label(anchor, current_label, pos))
            if(current_label == -1):
                ins_anchors.append(anchor_label(anchor, current_label, pos))
        """End of for loop"""
        # subsample positive labels if there are too many
    num_fg = int(0.5 * 256)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1  # set it to don't care

        # subsample negative lables if there are too many
    num_bg = 256 - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # rebounding the label to the ins_anchors
    for i in range(len(labels)):
        ins_anchors[i].label = labels[i]

    # generate the tx,ty,tw,th of the boxes, alert the dim of the bbox_targets is the same as the ins_anchors
    bbox_targets = np.zeros((len(ins_anchors), 4), dtype=np.float32)
    for i in range(len(ins_anchors)):
        ins_anchor = ins_anchors[i]
        if(len(np.array(gt_boxes).shape) > 1):
            bbox_targets[i] = np.array(compute_target(
                ins_anchor.anchor, gt_boxes[ins_anchor.pos]))
        else:
            bbox_targets[i] = np.array(
                compute_target(ins_anchor.anchor, gt_boxes))
    # set the value to 1 if the anchor is fg
    print(np.array(labels).shape)
    print(np.array(ins_anchors).shape)
    bbox_inside_weights = np.zeros((len(ins_anchors), 4), dtype=np.float32)
    for i in range(len(labels)):
        if(labels[i] == 1):
            bbox_inside_weights[i] = np.array([1, 1, 1, 1])

    bbox_outside_weights = np.zeros((len(ins_anchors), 4), dtype=np.float32)

    labels = np.array(labels)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0/num_examples
    negative_weights = np.ones((1, 4)) * 1.0/num_examples

    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # labels
    # map to orignal set of anchors
    labels = unmap(labels, total_anchors, ins_anchors, fill=-1)
    bbox_targets = unmap(bbox_targets, total_anchors, ins_anchors, fill=0)
    bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, ins_anchors, fill=0)
    bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, ins_anchors, fill=0)

    labels = labels.reshape(1, int(
        height/feat_strides), int(width/feat_strides), num_anchors).transpose(0, 3, 1, 2)
    labels = labels.reshape(
        (1, 1, num_anchors * int(height/feat_strides) * int(width/feat_strides)))
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets.reshape(
        (1, int(height/feat_strides), int(width/feat_strides), num_anchors * 4)).transpose(0, 3, 1, 2)
    rpn_bbox_inside_weights = bbox_inside_weights.reshape(
        (1, int(height/feat_strides), int(width/feat_strides), num_anchors * 4)).transpose(0, 3, 1, 2)
    rpn_bbox_outside_weights = bbox_outside_weights.reshape(
        (1, int(height/feat_strides), int(width/feat_strides), num_anchors * 4)).transpose(0, 3, 1, 2)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        for i in range(len(inds)):
            ret[i] = data[i]
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        for i in range(len(inds)):
            ret[i, :] = data[i, :]

    return ret


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


def compute_target(anchor, gt_box):
    tx = (gt_box[0]-anchor[0])/anchor[2]
    ty = (gt_box[1]-anchor[1])/anchor[3]
    tw = np.log(gt_box[2]/anchor[2])
    th = np.long(gt_box[3]/anchor[3])

    result = [tx, ty, tw, th]

    return result
