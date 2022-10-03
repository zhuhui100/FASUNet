
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def plot_figure(dice_list, name, save_path):
    fig = plt.figure(figsize = (12,7))
    plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06, hspace=None, wspace=None)
    plt.margins(0, 0)
    
    plt.grid() 
    d = dict(facecolor='r', edgecolor='r', alpha=0.5 )
    for i in range(0, len(dice_list[0])):
        plt.plot(np.array(dice_list)[:,i], label=str(i))
    plt.legend()   
    plt.title(str(name), bbox=d)
    plt.savefig(save_path)  



def save_3D_result(image_3D, pred_3D, gt_3D, img_name, save_dir, color_dict):


    for k in range(pred_3D.shape[2]):           

        image = image_3D[:, :, k, 0] - image_3D[:, :, k, 0].min()
        image = image_3D[:, :, k, 0]/image_3D[:, :, k, 0].max() *255    
        pred =  pred_3D[:, :, k]               
        gt = gt_3D[:, :, k]                   
        img_name_2d = img_name + "_" + str(k)
        
        ###-------------------------------
        img_edge = add_edge(image, gt, pred)[...,::-1]  ### bgr
        error_img = 1 - ((gt == pred) + 0)

        ###------------------------------- 
        color_gt = gray_to_color(gt, color_dict)
        color_pred = gray_to_color(pred, color_dict)

        gt_dir = save_dir + "gt/" + img_name +"_gt/"
        img_dir = save_dir + "img/" + img_name +"_img/"
        pred_dir = save_dir + "pred/" + img_name + "_pred/"
        os.makedirs(img_dir, exist_ok=True)

        cv2.imwrite(img_dir + img_name_2d + ".png", img_edge)
        cv2.imwrite(img_dir + img_name_2d + "_gt.png", color_gt)
        cv2.imwrite(img_dir + img_name_2d + "_pred.png", color_pred)
        cv2.imwrite(img_dir + img_name_2d + "_err.png", error_img*255)



def save_2D_result(image_2D, pred_2D, gt_2D, img_name, save_dir, color_dict):
    img_name_2d = img_name 
    ###-------------------------------
    img_edge = add_edge(image_2D, gt_2D, pred_2D)[...,::-1]  ### bgr
    error_img = 1 - ((gt_2D == pred_2D) + 0)

    ###------------------------------- 保存
    color_gt = gray_to_color(gt_2D, color_dict)
    color_pred = gray_to_color(pred_2D, color_dict)

    # gt_dir = save_dir + "gt/" + img_name +"_gt/"
    # img_dir = save_dir + "img/" + img_name[0:11] +"_img/"
    # pred_dir = save_dir + "pred/" + img_name + "_pred/"

    gt_dir = save_dir + "gt/"  
    img_dir = save_dir + "img/" 
    pred_dir = save_dir + "pred/" 

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)        
    os.makedirs(pred_dir,  exist_ok=True)

    cv2.imwrite(img_dir + img_name_2d + ".png", img_edge)
    cv2.imwrite(gt_dir + img_name_2d + "_gt.png", color_gt)
    cv2.imwrite(pred_dir + img_name_2d + "_pred.png", color_pred)
    cv2.imwrite(img_dir + img_name_2d + "_err.png", error_img*255)



# 绘制损失
def plot_loss_value(train_loss, test_loss, save_path):
    fig = plt.figure(figsize = (12,7))
    plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.grid() 
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.legend()
    plt.title("loss")
    plt.savefig(save_path)   


def save_object(data_3d, save_path, gt_3d=None):
    
    os.makedirs(save_path, exist_ok=True)
    
    for z in range(data_3d.shape[-1]):
        img = (data_3d[:, :, z] / data_3d.max()) *255
        
        if gt_3d is not None:
            gt = gt_3d[:,:,z]
            img = add_edge(img, gt)
            
        cv2.imwrite(save_path + str(z) + ".png", img)



def add_edge(image, gt, pred):
    
    if len(image.shape) == 3 and image.shape[-1] == 1:
        img_rgb_edge = np.concatenate([image, image, image], axis=2)
    elif len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
        img_rgb_edge = np.concatenate([image, image, image], axis=2)
    else:
        img_rgb_edge = np.array(image)
    if gt is not None:
        # print("----------------------")
        max_pix = gt.max() if gt.max() != 0 else 255

        edge_gt = cv2.Canny(np.uint8(gt), 10, 150)

        img_rgb_edge[:,:,0] = img_rgb_edge[:, :, 0] * (1 - edge_gt/max_pix)
        img_rgb_edge[:,:,1] = img_rgb_edge[:, :, 1] * (1 - edge_gt/max_pix)
        img_rgb_edge[:,:,2] = img_rgb_edge[:, :, 2] * (1 - edge_gt/max_pix)

        img_rgb_edge[:,:,1] = img_rgb_edge[:,:,1] + (edge_gt)


    max_pix = pred.max() if pred.max() != 0 else 255
    edge_pred = cv2.Canny(np.uint8(pred), 10, 150)

    img_rgb_edge[:,:,0] = img_rgb_edge[:,:,0] * (1-(edge_pred/max_pix))
    img_rgb_edge[:,:,1] = img_rgb_edge[:,:,1] * (1-(edge_pred/max_pix))
    img_rgb_edge[:,:,2] = img_rgb_edge[:,:,2] * (1-(edge_pred/max_pix))

    img_rgb_edge[:,:,0] = img_rgb_edge[:,:,0] + ((edge_pred))
    
    return img_rgb_edge


def gray_to_color(gray_image, color_dict):
    '''
    color_dict = {
    (255,0,0):  64,
}
    '''
    color_image = np.zeros((gray_image.shape[0], 
                            gray_image.shape[1], 3), dtype=np.uint8)

    for pattern, index in color_dict.items():
        temp = np.array(gray_image == np.array(index))
        color_image[temp] = np.array(pattern)
    
    return color_image[:,:,::-1]               # bgr


def Npy3dToImg(array_3d, name, save_dir):

    for i in range(array_3d.shape[-1]):
        img_name = name + "_" + str(i) + ".png"
        cv2.imwrite(save_dir + img_name, array_3d[:,:,i])