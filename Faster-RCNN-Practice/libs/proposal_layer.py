import tensorflow as tf #FIXME: import the tensorflow to test the param
import numpy as np
import tensorflow as tf
import bbox_transform
import gen_anchors
import cpu_nms
# author: R.B.G(rbg@facebook.com)
def proposal_layer(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, mode, feat_strides, anchor_scales):
    mode = tf.convert_to_tensor(mode)
    #launch the graph for the test use
    with tf.Session():
        rpn_bbox_cls_prob = rpn_bbox_cls_prob.eval()
        rpn_bbox_pred     = rpn_bbox_pred.eval()
        im_dims           = im_dims.eval()
    blob = tf.py_func(proposal_layer_py, [rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, mode, feat_strides, anchor_scales], [tf.float32],[-1,5])
    return tf.reshape(blob, [-1, 5])

def proposal_layer_py(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, mode, feat_strides, anchor_scales):

    anchors                    = gen_anchors.generate_anchors()
    num_anchors                = anchors.shape[0]
    rpn_bbox_cls_prob          = np.transpose( rpn_bbox_cls_prob, [0,3,1,2]) # [1, 9*2, height, width ]
    rpn_bbox_pred              = np.transpose( rpn_bbox_pred, [0,3, 1, 2])   # [1, 9*4, height, width ]  
    
    if mode == 'train':
        pre_nms_topN           =mode
        post_nms_topN          = 2000
        nms_thresh             = 0.7
        min_size               = 16
    else:
        pre_nms_topN           = 6000 
        post_nms_topN          = 300
        nms_thresh             = 0.7
        min_size               = 16

    # the first set of num_anchors channels are bg probabilities, the second set are the fg probablilities. 
    scores                     = rpn_bbox_cls_prob[:, :num_anchors, :, : ] # score for fg probablilities, [1, 9, height, width]
    bbox_deltas                = rpn_bbox_pred                             # [1, 9*4, height, width] 
   
    # step1 : generate proposal from bbox deltas and shifted anchors
    height, width              = scores.shape[-2:]
    shift_x                    = np.arange(0, width ) * feat_strides 
    shift_y                    = np.arange(0, height) * feat_strides
    shift_x, shift_y           = np.meshgrid( shift_x, shift_y )
    shifts                     = np.vstack( ( shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel() ) )
    shifts                     = shifts.transpose() # 
   
    A                          = num_anchors        # number of anchor per shift = 9
    K                          = shifts.shape[0]    # number of shift
    aaa                        = anchors.reshape((1, A, 4 ))
    bbb                        = shifts.reshape( 1, K, 4).transpose((1, 0, 2)) 
    anchors                    = aaa + bbb
    #anchors                    = anchors.reshape((1, A, 4 )) + shifts.reshape( 1, K, 4).transpose((1, 0, 2)) 
    anchors                    = anchors.reshape((K * A),4) # [ K*A, 4]       
    
    # transpose and reshape predicted bbox transformations to get the same order as anchors
    # bbox_deltas is [1, 4*A, H, W ]
    bbox_deltas                = bbox_deltas.transpose((0,2,3,1)).reshape((-1,4)) # [ A*K, 4]
    scores                     = scores.transpose((0,2,3,1)).reshape((-1,1)) # [ A*K, 1]
    
    # convert anchor into proposals via bbox transformations
    proposals                  = bbox_transform.bbox_transform_inv(anchors, bbox_deltas)  # [K*A, 4]
    # remove those proposals that out of range
    proposals                  = bbox_transform.clip_boxes(proposals,im_dims)
    keep                       = filter_boxes( proposals, min_size)
    proposals                  = proposals[keep,:]
   
    scores                     = scores[keep]
   
    # step4: sort all (proposal, score) pairs by score from highest to lowest   
    order                      = scores.ravel().argsort()[::-1] 
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]

    # step5: take top pre_nms_topN 
    proposals                  = proposals[order, :]
    scores                     = scores[order]
    
    # step6: apply nms ( e.g. threshold = 0.7 )
    keep                       = cpu_nms.cpu_nms( np.hstack( ( proposals, scores )), nms_thresh)
   
    if post_nms_topN > 0:                    
        keep = keep[:post_nms_topN]

    # step7: take after_nms_topN 
    proposals                  = proposals[keep,:]
    scores                     = scores[keep]
    print("proposals.shape after nms", proposals.shape)
    print("scores.shape", scores.shape)
    # step8: return the top proposal
    batch_inds                 = np.zeros( (proposals.shape[0], 1), dtype=np.float32) # [ len(keep), 1]
    
    blob                       = np.hstack( (batch_inds, proposals.astype(np.float32, copy=False))) # proposal structure: [0,x1,y1,x2,y2]
    print("blob.shape", blob.shape)
    return blob


def filter_boxes(boxes, min_size):
    """
    remove all blxes with any side smaller than min_size 
    
    input: boxes [x1,y1,x2,y2]
    """
    ws                        = boxes[:,2] - boxes[:,0] + 1
    hs                        = boxes[:,3] - boxes[:,1] + 1
    keep                      = np.where(( ws >= min_size) & (hs >= min_size))[0]
    #keep = np.array(keep)
    return keep



    
    
