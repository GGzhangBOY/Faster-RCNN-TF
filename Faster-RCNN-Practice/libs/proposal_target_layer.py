import numpy as np
import tensorflow as tf
import bbox_transform
import proposal

def proposal_target_layer(rpn_rois, gt_boxes, num_classes):

	
	rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func( proposal_target_layer_py, [ rpn_rois, gt_boxes, num_classes], [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32])
	rois                   = tf.reshape( rois, [-1, 5],                      name = 'rois')
	labels                 = tf.convert_to_tensor( tf.cast(labels, tf.int32),name = 'labels')
	bbox_targets           = tf.convert_to_tensor( bbox_targets,             name = 'bbox_targets')
	bbox_inside_weights    = tf.convert_to_tensor( bbox_inside_weights,      name = 'bbox_inside_weights')
	bbox_outside_weights   = tf.convert_to_tensor( bbox_outside_weights,     name = 'bbox_outside_weights')
	return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

def proposal_target_layer_py(rpn_rois, gt_boxes, num_classes):
    # Proposal ROIs (0, x1, y1, x2, y2) come from RPN blob
	all_rois                                        = rpn_rois
	print("all_rois.shape", all_rois.shape)
	# Include ground-truth boxes in the set of candidates rois
	zeros                                           = np.zeros((gt_boxes.shape[0],1), dtype=gt_boxes.dtype)
	all_rois                                        = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:,:])))) # append gt_boxes at the last row of all_rois 
	all_rois 										= all_rois.astype(np.float32)
	print("all_rois.shape", all_rois.shape)
	
	num_images                                      = 1 
	rois_per_image                                  = 128
	fg_rois_per_image                               = np.round( 0.25 * rois_per_image).astype(np.int32)

	print("fg_rois_per_image", fg_rois_per_image, "rois_per_image", rois_per_image)
	