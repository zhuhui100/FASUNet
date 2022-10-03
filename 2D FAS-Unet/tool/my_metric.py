import numpy as np
import SimpleITK as sitk
from medpy import metric
import datetime


def fast_hist(label_true, label_pred, n_class):

    label_true = label_true.flatten()
    label_pred = label_pred.flatten()
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
            n_class * label_pred[mask].astype(int) + \
            label_true[mask].astype(int), 
            minlength=n_class ** 2).reshape(n_class, n_class)
    
    return hist

def class_dice(hist):


    dice = 2 * np.diag(hist)/(hist.sum(1) + hist.sum(0))
    mDice = np.nanmean(dice[1:])
    return list(dice) + [mDice]

def my_calculate_metric(hist):

    
    dice = class_dice(hist)

    precision = np.diag(hist)/hist.sum(1)
    recall = np.diag(hist)/hist.sum(0)
    
    acc = np.diag(hist).sum()/hist.sum(1).sum()
    results = {
    "acc": acc, "precision": precision, 
        "recall": recall, "dice": dice, 
        "hist": hist
        }
    return results


def calculate_metric_medpy(gt, pred, n_class, voxelspacing=[0.98, 0.98, 2.5]):


    results = np.zeros((n_class, 12))

    for i in range(n_class):
        cur_gt = (gt == i).astype(np.int16)
        cur_pred = (pred == i).astype(np.int16)

        results[i,0] = metric.binary.dc(result=cur_pred, reference=cur_gt)
        results[i,1] = metric.binary.jc(cur_pred, cur_gt)
        if cur_pred.sum() > 0 and cur_gt.sum() > 0:
            results[i,2] = metric.binary.hd95(cur_pred, cur_gt, voxelspacing=voxelspacing)
        else:
            results[i,2] = float('nan')
        results[i,3] = metric.binary.precision(cur_pred, cur_gt)
        results[i,4] = metric.binary.sensitivity(cur_pred, cur_gt) 
        results[i, 5] = metric.binary.assd(cur_pred, cur_gt, voxelspacing=voxelspacing) 

    results_dict = {"dice":results[:,0],
                    "jc": results[:,1],
                    "hd95": results[:,2],
                    "precision":results[:,3],
                    "sen":results[:,4],
                    "assd": results[:,5]
        }

    return results_dict



