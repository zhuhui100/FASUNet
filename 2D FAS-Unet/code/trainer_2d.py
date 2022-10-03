## 训练函数，测试函数，预测函数
import numpy as np
import torch.optim as optim
import torch

import cv2
import datetime
import os
from tool.metrics import *
from tool.draw import save_2D_result
import shutil




def train(args, model, device, criterion, optimizer, epoch, train_loader):
    hist = np.zeros((args.num_classes, args.num_classes))
    metrics = dict()            
    out_data = dict()           
    
    model.train()
    for i, (x, y, _, _, _, name) in enumerate(train_loader):
        
        inputs = x.to(device)
        target = y.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, target) 
        loss.backward()
        optimizer.step()
        
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    hist = fast_hist(target.cpu().numpy(), pred, args.num_classes)
            
    batch_loss = loss.item() 
    Dices = np.array(class_dice(hist))
    
    metrics["batch_loss"] = loss.item() 
    metrics["dice"] = Dices

    out_data["img"], out_data["gt"] = x.detach(), y.detach()
    out_data["name"] = name
    out_data["output"], out_data["pred"] = output.detach(), pred
        
    return out_data, metrics



def validation_2d_to_3d(args, model, device, criterion, test_loader):

    epoch_loss, step,  = 0, 0
    hist = np.zeros((args.num_classes, args.num_classes))
    out_data, metrics = dict(), {}
    hist_3d, person_dice = np.zeros_like(hist), []
    
    model.eval()
    
    with torch.no_grad():
        for i, (x, y, _, _, _, name) in enumerate(test_loader):
            if i == 0:
                person_id_pre = name[0].split(".")[0].split("_")[1]

            inputs = x.to(device)
            target = y.to(device)

            output = model(inputs)

            loss = criterion(output, target)
            epoch_loss += loss.item()
            
            pred = torch.argmax(output, dim=1).cpu().detach().numpy()
            hist += fast_hist(target.cpu().numpy(), pred, args.num_classes)

            hist_3d += fast_hist(target.cpu().numpy(), pred, args.num_classes)
            person_id = name[0].split(".")[0].split("_")[1]
            if person_id != person_id_pre or (i+1) == len(test_loader) :
                temp_dice = np.array(class_dice(hist_3d))[1:-1]
                person_dice.append(temp_dice)
                person_id_pre = person_id
                hist_3d = np.zeros_like(hist)


    avg_loss = epoch_loss / (i+1)
    Dices = np.array(class_dice(hist))

    person_dice = np.array(person_dice)
    avg_dice = np.round(np.mean(np.mean(person_dice, axis=0), axis=0), 4)  # 压缩行，列
    # print(person_dice, "\n", person_dice.shape, "\n", avg_dice, Dices)
    
    
    metrics["avg_loss"], metrics["dice"] = avg_loss, Dices
    metrics["avg_dice"] = avg_dice

    out_data["img"], out_data["gt"] = x.detach(), y.detach()
    out_data["name"] = name
    out_data["output"], out_data["pred"] = output.detach(), pred
    
    return out_data, metrics




def print_to_terminal_and_txt(message, output_txt_file, term = True):
    """
    message: list
    """

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
            
            

def save_2D_predict(args, save_dir, model, device, data_loader, color_dict, txt_file):

    model.eval()
    model.load_state_dict(torch.load(args.best_model))
    now = datetime.datetime.now()
    print_to_terminal_and_txt(["time：", now.strftime("%m-%d/%H:%M:%S"), 
            "\nmodel name: ", args.best_model,
            '\nload success**********'], txt_file)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir + "img/", exist_ok=True)

    hist = np.zeros((args.num_classes, args.num_classes))  
    with torch.no_grad():
        for j, (x, y, _, _, _, name) in enumerate(data_loader):

            inputs = x.to(device)
            y = y.numpy()
            
            output = model(inputs)
            pred_all = torch.argmax(output, dim=1).cpu().detach()
            pred_all = pred_all.numpy()

            hist_batch = fast_hist((y), pred_all, args.num_classes)
            hist += hist_batch
            dice_batch = np.array(class_dice(hist_batch))
            
            x_cpu = x.detach().cpu().numpy()
            # 维度重排
            image_all = np.transpose(x_cpu, (0, 2, 3, 1)) - x_cpu.min()

            for i in range(image_all.shape[0]):
                img_name = name[i].split('/')[-1].split(".")[0]
                
                image_2D = (image_all[i]/(image_all[i].max()-image_all[i].min())) * 255

                save_2D_result(image_2D, pred_all[i], y[i], img_name, save_dir, color_dict)

                print_to_terminal_and_txt(img_name + ", net output size:" + \
                    str(output.detach().cpu().numpy().shape) + \
                    "\tdice%:" + str(np.round(dice_batch *100 , 2)), txt_file)


        dice = np.array(class_dice(hist))
        print_to_terminal_and_txt("average dice%: "+ str(np.round(dice * 100 , 2)), txt_file)

    
    return hist, y, pred_all
    
    
    
    
def save_to_3d_data(args, model, device, test_loader, file_name):
    
    pred_list, gt_list = [], []
    model.eval()
    if device == "cpu":
        torch.load(args.best_model, map_location=torch.device('cpu'))
    else:
        model.load_state_dict(torch.load(args.best_model))

    with torch.no_grad():
        for i, (x, y, _, _, _, name) in enumerate(test_loader):
        
            inputs = x.to(device)
            target = y.detach().numpy()
            output = model(inputs)
            pred = torch.argmax(output, dim=1).cpu().detach().numpy()

            pred_list.append(pred)
            gt_list.append(target)
    pred_list = np.squeeze(np.array(pred_list, np.int8))
    gt_list = np.squeeze(np.array(gt_list, np.int8))
    pred_list = np.transpose(pred_list, (1, 2, 0))  # np.transpose 不改变内部本身
    gt_list = np.transpose(gt_list, (1, 2, 0))


    os.makedirs("./pred_3d/", exist_ok=True)
    np.save("./pred_3d/" + file_name + "_gt.npy", gt_list)
    np.save("./pred_3d/" + file_name + "_pred.npy", pred_list)