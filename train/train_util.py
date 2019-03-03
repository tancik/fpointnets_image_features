''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np

def get_batch(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    batch_dim_int = np.zeros((bsize, 512))
    batch_orient_int = np.zeros((bsize, 256))
    batch_conf_int = np.zeros((bsize, 256))
    batch_conv_int = np.zeros((bsize, 25088))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec,dim_int,orient_int,conf_int,conv_int = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,dim_int,orient_int,conf_int,conv_int = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
        batch_dim_int[i] = dim_int
        batch_orient_int[i] = orient_int
        batch_conf_int[i] = conf_int
        batch_conv_int[i] = conv_int
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec, \
            batch_dim_int, batch_orient_int, batch_conf_int, batch_conv_int
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle, \
            batch_dim_int, batch_orient_int, batch_conf_int, batch_conv_int

def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                 num_point, num_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    batch_dim_int = np.zeros((bsize, 512))
    batch_orient_int = np.zeros((bsize, 256))
    batch_conf_int = np.zeros((bsize, 256))
    batch_conv_int = np.zeros((bsize, 25088))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,rotangle,prob,onehotvec,dim_int,orient_int,conf_int,conv_int = dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,rotangle,prob,dim_int,orient_int,conf_int,conv_int = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
        batch_dim_int[i] = dim_int
        batch_orient_int[i] = orient_int
        batch_conf_int[i] = conf_int
        batch_conv_int[i] = conv_int
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec, batch_dim_int, batch_orient_int, batch_conf_int, batch_conv_int
    else:
        return batch_data, batch_rot_angle, batch_prob, batch_dim_int, batch_orient_int, batch_conf_int, batch_conv_int
