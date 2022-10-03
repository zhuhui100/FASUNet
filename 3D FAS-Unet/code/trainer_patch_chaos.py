
from cv2 import IMWRITE_PNG_BILEVEL
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
from skimage import measure

import cv2
import datetime
import os
from data_tool.my_metrics import *
import SimpleITK as sitk
import math


img_info = {
              "5": [512, 512, 95],
              "18": [512, 512, 111],
              "24": [512, 512, 123],
              "29": [512, 512, 214]
              }

patch_size = [128, 128, 64]
slide_stride = [64, 64, 32]


def print_to_terminal_and_txt(message, out_txt):
  print(message)
  print(message, file=out_txt)


def patch_assembled(patch, patch_name, patch_size, slide_stride, img_B):

  hpb1, hpb2 = int(slide_stride[0]/2), int(patch_size[0] - slide_stride[0]/2) 
  wpb1, wpb2 = int(slide_stride[1]/2), int(patch_size[1] - slide_stride[1]/2)
  dpb1, dpb2 = int(slide_stride[2]/2), int(patch_size[2] - slide_stride[2]/2)
  patch_box = patch[..., hpb1:hpb2, wpb1:wpb2, dpb1:dpb2]
  

  patch_id = patch_name.split("_")[1].split(".")[0]
  h1 = int(patch_id[0]) * (patch_size[0]/2)
  h2 = int(h1 + (patch_size[0]/2 ) )
  w1 = int(patch_id[1]) * (patch_size[1]/2)
  w2 = int(w1 + (patch_size[1]/2 ))
  d1 = int(patch_id[2]) * (patch_size[2]/2)
  d2 = int(d1 + (patch_size[2]/2 ))


  if len(patch.shape) == 4:
    img_B[..., int(h1):h2, int(w1):w2, int(d1):d2,] = patch_box
  if len(patch.shape) == 3:
    img_B[int(h1):h2, int(w1):w2, int(d1):d2,] = patch_box
  
  return img_B


def general_img_B(img_id, patch_size):

  h, w, d  = img_info[img_id][0], img_info[img_id][1], img_info[img_id][2]

  height_patch_number = math.ceil(h / patch_size[0])    
  width_patch_number = math.ceil(w / patch_size[1])    
  deepth_patch_number = math.ceil(d / patch_size[2]) 

  h1 = int(height_patch_number * patch_size[0])
  w1 = int(width_patch_number * patch_size[1])
  d1 = int(deepth_patch_number * patch_size[2])
  
  if len(patch_size) == 4 :
    img_B = np.zeros([patch_size[0], h1, w1, d1])
  else:
    img_B = np.zeros([h1, w1, d1])

  return img_B



def connected_component(image, thr=1):

  label, num = measure.label(image, connectivity=1, return_num=True)
  if num < 1:
    return image

  region = measure.regionprops(label)


  num_list = [i for i in range(1, num+1)]
  area_list = [region[i-1].area for i in num_list] 
  num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]


  if len(num_list_sorted) > thr: 
    for i in num_list_sorted[thr:]:
      label[region[i-1].slice][region[i-1].image] = 0
    num_list_sorted = num_list_sorted[:thr]

  label[label > 0] = 1 
  return label


def train(args, model, device, criterion, optimizer, epoch, train_loader):
  epoch_loss = 0
  hist = np.zeros((args.num_classes, args.num_classes))
  trian_metrics = dict()      
  out_data = dict()           
  model.train()

  for i, (x, y, name) in enumerate(train_loader):

    inputs = x.to(device)
    target = (y//args.min_pix).to(device)

    optimizer.zero_grad()

    output = model(inputs)
    loss = criterion(output, target)  
    loss.backward()
    optimizer.step()


  pred = torch.argmax(output, dim=1).cpu().detach()
  hist = fast_hist(target.cpu().numpy(), pred.numpy(), args.num_classes)

  Dices = np.array(class_dice(hist))
    
  trian_metrics["batch_loss"] = loss.item() 
  trian_metrics["batch_dice"] = Dices

  out_data["img"], out_data["gt"] = x.detach(), y.detach()
  out_data["name"] = name
  out_data["output"], out_data["pred"] = output.detach(), pred


  return out_data, trian_metrics



def validation(args, model, device, criterion, epoch, val_loader):


  epoch_loss = 0
  hist = np.zeros((args.num_classes, args.num_classes))
  out_img, val_metrics = dict(), {}
  patch_num = len(val_loader)
  previous_img_id = -1

  model.eval()

  with torch.no_grad():
    for i, (x, y, name) in enumerate(val_loader):

      inputs = x.to(device)
      target = (y//args.min_pix)[0].numpy()    # torch.Size([1, 128, 128, 64]
      output = model(inputs)                    # [1, 2, 128, 128, 64]
      pred = torch.argmax(output, dim=1).cpu().detach()[0]  # torch.Size([1, 128, 128, 64]
      output = output[0].cpu().detach()

      
      patch_name = name[0].split("/")[-1]  # 5_010.npy
      img_id = patch_name.split("_")[0]    # 5

      if previous_img_id != img_id or i == (patch_num-1):
      
        if previous_img_id != -1:
          h_pre, w_pre = img_info[previous_img_id][0], img_info[previous_img_id][1]
          d_pre = img_info[previous_img_id][2]

          pred_original = pred_B[0:h_pre, 0:w_pre, 0:d_pre]
          gt_original = gt_B[0:h_pre, 0:w_pre, 0:d_pre]
          output_original = output_B[..., 0:h_pre, 0:w_pre, 0:d_pre]

          output_original = torch.tensor(output_original, dtype=torch.float32)
          gt_original_torch = torch.tensor(gt_original, dtype=torch.int64)
          output_original = torch.unsqueeze(output_original, 0).to(device)
          gt_original_torch = torch.unsqueeze(gt_original_torch, 0).to(device)

          loss = criterion(output_original, gt_original_torch)

          epoch_loss += loss.item()
          hist += fast_hist(gt_original, pred_original, args.num_classes)


        output_B_shape = [2] + patch_size 
        output_B = general_img_B(img_id=img_id, patch_size=output_B_shape)
        pred_B = general_img_B(img_id=img_id, patch_size=patch_size)
        gt_B = general_img_B(img_id=img_id, patch_size=patch_size)

      output_B = patch_assembled(output, patch_name, patch_size, slide_stride, output_B)
      pred_B = patch_assembled(pred, patch_name, patch_size, slide_stride, pred_B)
      gt_B = patch_assembled(target, patch_name, patch_size, slide_stride, gt_B)
      previous_img_id = img_id


  batch_loss = epoch_loss / len(img_info)
  Dices = np.array(class_dice(hist))

  val_metrics["avg_loss"], val_metrics["dice"] = batch_loss, Dices

  out_img["gt"] = gt_original
  out_img["name"] = name
  out_img["pred"] = pred_original

  return out_img, val_metrics




def print_to_terminal_and_txt(message, output_txt_file, term = True):


    if type(message) == list:
        if term:
            print(*message)
        with open(output_txt_file, "a+") as f:
            print(*message, file=f)
    else:
        if term:
            print(message)
        with open(output_txt_file, "a+") as f:
            print(message, file=f)
            
            


def save_3D_prediction_to_npy(args, save_dir, model, device, data_loader, output_txt_file, post_process=False, nii=False):

  model.eval()
  model.load_state_dict(torch.load(args.best_model))
  now = datetime.datetime.now()
  print_to_terminal_and_txt(["time 1：", now.strftime("%m-%d/%H:%M:%S"),
        "\nmodel name: ", args.best_model,
        "\nsave dir:", save_dir,
        '\nload pth file success ...'], output_txt_file)

  hist = np.zeros((args.num_classes, args.num_classes))
  patch_num = len(data_loader)
  previous_img_id = -1
  os.makedirs(save_dir + "/npy/", exist_ok=True)
  os.makedirs(save_dir + "/nii/", exist_ok=True)


  with torch.no_grad():
    for batch_count, (x, y, name) in enumerate(data_loader):

      inputs = x.to(device)
      target = (y//args.min_pix)[0].numpy()    # torch.Size([1, 128, 128, 64]
      output = model(inputs)                    # [1, 2, 128, 128, 64]
      pred = torch.argmax(output, dim=1).cpu().detach()[0]  # torch.Size([1, 128, 128, 64]
      output = output[0].cpu().detach()
      patch_name = name[0].split("/")[-1]  # 5_010.npy
      img_id = patch_name.split("_")[0]    # 5
      inputs = inputs[0].cpu().detach()

      if previous_img_id != img_id or batch_count == (patch_num-1):
        print("%s a CT data finish"%(previous_img_id, img_id))
      
        if previous_img_id != -1:
          h_pre, w_pre = img_info[previous_img_id][0], img_info[previous_img_id][1]
          d_pre = img_info[previous_img_id][2]

          img_original = img_B[0:h_pre, 0:w_pre, 0:d_pre] 
          pred_original = pred_B[0:h_pre, 0:w_pre, 0:d_pre].astype(np.int16)
          gt_original = gt_B[0:h_pre, 0:w_pre, 0:d_pre]

          if post_process:
            pred_original = connected_component(1 - pred_original)
            pred_original = connected_component(1 - pred_original)
          if len(previous_img_id) < 2:
            sample_id = "0%s"%(previous_img_id)
          else:
            sample_id = previous_img_id
          
          np.save(save_dir + "/npy/"+ sample_id +".npy", img_original)
          np.save(save_dir+"/npy/" + sample_id + "_gt.npy", gt_original)
          np.save(save_dir+"/npy/"+ sample_id + "_pred.npy", pred_original)
          if nii:
            sitk.WriteImage(img_original, save_dir + "/nii/"+ sample_id + ".nii.gz")
            sitk.WriteImage(gt_original, save_dir + "/nii/"+ sample_id + "_gt.nii.gz")
            sitk.WriteImage(pred_original, save_dir + "/nii/"+ sample_id + "_pred.nii.gz")


          print("batch_count = ", batch_count)


          hist_batch = fast_hist(gt_original, pred_original, args.num_classes)
          hist += hist_batch
          dice_person = np.array(class_dice(hist_batch))
          print_message = "%s: size: %s, dice:%s"%(previous_img_id, str(img_original.shape), 
            str(np.round(dice_person * 100, 2)))
          print_to_terminal_and_txt(print_message, output_txt_file)
          print("\n")

        else:
          print("-1 denote nan\n")

        # output_B_shape = [2] + patch_size 
        # output_B = general_img_B(img_id=img_id, patch_size=output_B_shape)
        pred_B = general_img_B(img_id=img_id, patch_size=patch_size)
        gt_B = general_img_B(img_id=img_id, patch_size=patch_size)
        img_B = general_img_B(img_id=img_id, patch_size=patch_size)

      # output_B = patch_assembled(output, patch_name, patch_size, slide_stride, output_B)
      pred_B = patch_assembled(pred, patch_name, patch_size, slide_stride, pred_B)
      gt_B = patch_assembled(target, patch_name, patch_size, slide_stride, gt_B)
      img_B = patch_assembled(inputs, patch_name, patch_size, slide_stride, img_B)

      previous_img_id = img_id


  end_time = now = datetime.datetime.now()
  print_to_terminal_and_txt(["start time ：", end_time.strftime("%m-%d/%H:%M:%S")], output_txt_file)



import glob
import cv2
import shutil



def mask_modify_color(segmentation_gray, color_dict):
  '''
  color_dict = {
  [255,0,0]:  (0, 0, 255),
  }
  '''

  if len(segmentation_gray.shape) != 3:
    segmentation_gray = np.expand_dims(segmentation_gray, axis=2)
    segmentation_gray = np.repeat(segmentation_gray, 3, axis=2)

  seg_color = np.zeros_like(segmentation_gray, dtype=np.float32)

  for old_color, new_color in color_dict.items():
    seg_color = np.where(segmentation_gray==old_color, new_color, seg_color)
  
  return seg_color


def overlayMaskandImage(img_gray, color_dict, segmentation_gray, alpha):

  if len(img_gray.shape) != 3:
    img_gray = np.expand_dims(img_gray, axis=2)
    img_gray = np.repeat(img_gray, 3, axis=2)

  colorizedMask = mask_modify_color(segmentation_gray, color_dict)[:,:,::-1]

  segbin = np.greater(segmentation_gray, 0)
  repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)

  overlayed = np.where(
        repeated_segbin,
        np.round(alpha*colorizedMask + (1-alpha)*img_gray).astype(np.uint8),
        np.round(img_gray).astype(np.uint8)
    )

  return overlayed


def img_norm_to_255(img):
  img = (img-img.min())/(img.max()-img.min()) *255

  return img


def visual_volume_pred_gt(data_dir, save_dir, data_type = "npy"):
  """
    data_dir: ./val/npy/
    save path: ./png/01/01.png
  """

  gt_path_list = glob.glob("%s*_gt.npy"%(data_dir))

  color_dict = {
      (1, 1, 1):    [150, 0, 0],   # R, G, B  
  }

  alpha = 0.7
  
  txt_file = "./slice_dice.txt"


  for gt_path in gt_path_list:
    pred_path = gt_path.replace("_gt", "_pred")
    img_path = gt_path.replace("_gt", "")

    if data_type == "npy":
      gt_volume = np.load(gt_path)
      pred_volume = np.load(pred_path)
      img_volume = np.load(img_path)
    elif data_type == "nii":
      img_data = sitk.ReadImage(img_path)
      img_volume = sitk.GetArrayFromImage(img_data)
      gt_data = sitk.ReadImage(gt_path)
      gt_volume = sitk.GetArrayFromImage(gt_data)
      pred_data = sitk.ReadImage(pred_path)
      pred_volume = sitk.GetArrayFromImage(pred_data)
    else:
      print("data name error")

    img_volume = img_norm_to_255(img_volume)

    img_id = img_path.split("/")[-1].split(".")[0]
    save_path_i = os.path.join(save_dir, img_id)

    if os.path.exists(save_path_i):
      shutil.rmtree(save_path_i)  

    os.makedirs(save_path_i)

    for i in range(img_volume.shape[2]):
      gt = gt_volume[:,:,i]
      pred = pred_volume[:,:,i]
      img = img_volume[:, :, i]
      
      
      slice_dice = class_dice(fast_hist(gt, pred, 2))
      print_to_terminal_and_txt([img_path, " ", i, " ", slice_dice], txt_file)

      cv2.imwrite("%s/%s.png"%(save_path_i, str(i)), img)

      img_pred = overlayMaskandImage(img, color_dict, pred, alpha)
      cv2.imwrite("%s/%s_pred.png"%(save_path_i, str(i)), img_pred)

      img_gt = overlayMaskandImage(img, color_dict, gt, alpha)
      cv2.imwrite("%s/%s_gt.png"%(save_path_i, str(i)), img_gt)

