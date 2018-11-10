import numpy as np
import tensorflow as tf


def bbox_transform(anchor, gt_box):
    """
    compute the transform that the bbox need to transform to the gt_box
    """
    #print("anchor point ",anchor[0],anchor[1],anchor[2],anchor[3])
    width = anchor[2]-anchor[0]
    height = anchor[3]-anchor[1]
    gt_width = gt_box[2]-gt_box[0]
    gt_height = gt_box[3]-gt_box[1]
    tx = (gt_box[0]-anchor[0])/width
    ty = (gt_box[1]-anchor[1])/height
    tw = np.log(gt_width/width)
    th = np.log(gt_height/height)
    print("tw th",tw,th)
    result = [tx, ty, tw, th]

    return result


def bbox_transform_inv(boxes, deltas):
    """
    Applied deltas to box cooridate to obtain new boxes
    input: bboxes is in the form of [x1,y1, x2, y2], where (Ax,Ay) is the top-left corner coordinate, and Aw and Ah are bounding box width and height

    output: 
     predicted boxes
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    # width, height, center of bounding box
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # delta
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # pred = bounding box + delta
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    clip boxes to image boundaries
    input:
    im_shape[0] : height
    im_shape[1] : width

    """
    im_shape = im_shape[np.newaxis, :]
    # x1 >= 0
    print("boxes:", boxes.shape)
    print("imshape: ", im_shape.shape)
    boxes[:, 0::4] = np.maximum(np.minimum(
        boxes[:, 0::4], im_shape[0, 1]-1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(
        boxes[:, 1::4], im_shape[0, 0]-1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(
        boxes[:, 2::4], im_shape[0, 1]-1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(
        boxes[:, 3::4], im_shape[0, 0]-1), 0)
    return boxes
