import torch
import numpy as np

def calc_acc(label, gt_label):
    label = label.numpy()
    gt_label = gt_label.numpy()
    batch_size = label.shape[0]
    
    cls_ = np.argmax(label, 1)
    gt_cls = np.argmax(gt_label, 1)
    is_incor = (cls_ != gt_cls)
    
    cor_cls_pred = np.max(label * gt_label, -1)
    acc = 1 - np.mean(is_incor)
    
    incor_pred = cor_cls_pred[is_incor]
    incor_idx = np.array(range(batch_size))[is_incor]
    incor_cls = cls_[is_incor]
    cor_cls = gt_cls[is_incor]
    
    print(incor_cls)
    
    ans_list = list(zip(incor_idx, incor_pred, incor_cls, cor_cls))
    return sorted(ans_list, key = lambda x: -x[1])
