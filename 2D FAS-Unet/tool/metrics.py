import numpy as np
import cv2
import datetime

    
def fast_hist(label_true, label_pred, n):

    label_true = label_true.flatten()
    label_pred = label_pred.flatten()
    mask = (label_true >= 0) & (label_true < n)
    hist = np.bincount(
            n * label_pred[mask].astype(int) + \
            label_true[mask].astype(int), 
            minlength=n ** 2).reshape(n, n)
    
    return hist


def class_dice(hist):

    dice = 2 * np.diag(hist)/(hist.sum(1) + hist.sum(0))
    mDice = np.nanmean(dice[1:])
    return list(dice) + [mDice]



def dice(im1, im2,tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


if __name__ == "__main__":

    gt = np.load("./mg_unet_d1/test_img_result/training_axial_crop_pat3_gt.npy")
    pred = np.load("./mg_unet_d1/test_img_result/training_axial_crop_pat3_pred.npy")
    print(gt.shape, pred.shape)

    
    hist = fast_hist(gt, pred, 3)
    dice_1 = class_dice(hist)

    dice_2 = dice(gt[:,:,50:80], pred[:,:,50:80], 1)
    print(dice_1)
    print(dice_2)
    