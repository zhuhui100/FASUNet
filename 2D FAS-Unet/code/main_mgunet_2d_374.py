#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
from data_parallel_my import BalancedDataParallel


import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tool.mydataset import MyDataset
from tool.draw import plot_figure, plot_loss_value
from tool.myTransforms import (MinMaxNormalize, PadZ_8, Compose)

from trainer_2d import train, validation_2d_to_3d , save_2D_predict, save_to_3d_data

import numpy as np
import datetime

import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')


from fasunet_2d_374 import FASUNet                                            


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default = 0, help="epoch")
parser.add_argument("--num-classes", type=int, default = 5, 
                        help="defiault for SegTHOR dataseta")

parser.add_argument("--lr", type = float, default=1e-2)
parser.add_argument("--model-dir", type=str, default = "./weight/")
parser.add_argument("--output_dir", type=str, default = "./output/")
parser.add_argument("--root_dir", type=str, default = ".",
                        help="") 


### 训练txt 文件路径
parser.add_argument("--train_txt", type=str, 
                        default = "../dataset_2D/txt/train_304.txt",
                        )   
### 测试txt 文件路径
parser.add_argument("--test-txt", type=str, 
                        default = "../dataset_2D/txt/test_304.txt") 
parser.add_argument("--val_txt", type=str, 
                        default = "../dataset_2D/txt/temp_train_304.txt") 

parser.add_argument("--epochs", type=int, default = 200, help="150 for segthor dataset")    
parser.add_argument("--batch-size", type=int, default = 16) 

exp_id = "2"         
ch = 64
model_name = "FAS"
block="374"

### 一些文件保存的名字
parser.add_argument("--channels",  type=int, default = ch)
parser.add_argument("--final-model",  
                    type=str, default = "./weight/%s_%s_final_%s_%d_uf.pth"%(model_name, block, exp_id, ch))
parser.add_argument("--best_model",  
                    type=str, default = "./weight/%s_%s_best_%s_%d_uf.pth"%(model_name, block, exp_id, ch))
parser.add_argument("--avg_best_model",  
                    type=str, default = "./weight/%s_%s_avg_best_%s_%d_uf.pth"%(model_name, block, exp_id, ch))
parser.add_argument("--log_txt",
                    type=str, default="./output/%s_%s_%s_%d_uf.txt"%(model_name, block, exp_id, ch))
parser.add_argument("--train_dice_name",
                    type=str, default = "./output/%s_%s_%s_%d_uf_train_dice.png"%(model_name, block, exp_id, ch))
parser.add_argument("--test_dice_name", 
                    type=str, default = "./output/%s_%s_%s_%d_uf_val_dice.png"%(model_name, block, exp_id, ch))
parser.add_argument("--loss_name", 
                    type=str, default = "./output/%s_%s_%s_%d_uf_loss.png"%(model_name, block, exp_id, ch))

parser.add_argument("--val_data_dict", 
                    type=str, default = "./output/%s_%s_%s_%d_val.npy"%(model_name, block, exp_id, ch))
parser.add_argument("--train_data_dict", 
                    type=str, default = "./output/%s_%s_%s_%d_train.npy"%(model_name, block, exp_id, ch))
                    
parser.add_argument("--information", type=str,  
        default = "%s, %s, %d, u=f,"%(model_name, block, ch)) 
args = parser.parse_args()


os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
np.set_printoptions(formatter={'all':lambda x: "%5.2f"%x})
train_loss, train_dice, test_loss, test_dice = [], [], [], []


color_dict = {
    (0,0,0):      0,
    (0, 0, 255):    1,     
    (0, 180, 180):  2,  
    (0, 200, 0):  3,     
    (180, 0, 0):  4,    
}   


def print_measure(epoch, train_info, test_info):


    now = datetime.datetime.now()
    message = [
        '{}/{}'.format(epoch, args.epochs), 
            "train: loss=%6.4f "%(train_info["batch_loss"]), 
            "dice=", np.round(train_info["dice"] *100 , 2), 
            "\nval: "
            "loss=%6.4f"%(test_info["avg_loss"]), 
            "dice=", np.round(test_info["dice"] *100 , 2), 
            "avg dice=", np.round(test_info["avg_dice"]*100, 2),
            "\n",
            now.strftime("%H:%M:%S"),
            "\n"
    ]
    

    return message 


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

        
def main():
    global_best_dice = 50.0     
    avg_best_dice = 50.0

    for k,v in sorted(vars(args).items()):
        print_to_terminal_and_txt([k,'=', v], args.log_txt)
    print_to_terminal_and_txt("\n\n", args.log_txt)

    print_info = ["python file name:", os.path.basename(sys.argv[0]), "\n:",
            args.best_model, "\n:%s\n" % (args.information),
            "#train data:%d, #validataion data: %d" % (len(train_dataset), len(val_loader))]
    print_info = "".join(print_info)
    print_to_terminal_and_txt(print_info, args.log_txt)

    
    val_results = {"avg_loss": [], "dice":[], "avg_dice":[]}
    train_results = {"batch_loss": [], "dice":[]}
    
    delta_epoch = 50


    for epoch in range(args.epoch, args.epochs):

        train_data, train_metrics =  train(args, model, device, 
            criterion, optimizer, epoch, train_loader)

        val_data, val_metrics = validation_2d_to_3d(args, model, device, 
            criterion, val_loader)
        
        train_results["batch_loss"].append(train_metrics["batch_loss"])
        train_results["dice"].append(train_metrics["dice"])

        val_results["avg_loss"].append(val_metrics["avg_loss"])
        val_results["avg_dice"].append(val_metrics["avg_dice"])
        val_results["dice"].append(val_metrics["dice"])

        message = print_measure(epoch, train_metrics, val_metrics)
        print_to_terminal_and_txt(message, args.log_txt)

        try:
            torch.save(model.state_dict(), "./temp_weight/%s_%s_%s.pth"%(exp_id, 
                    str(epoch), model_name))
        except OSError:
            
            print('save pth file faile')

        global_dice = val_metrics["dice"][-1] * 100

        if global_dice >= global_best_dice: 
            global_best_dice = global_dice
            torch.save(model.state_dict(), args.best_model)
            print_to_terminal_and_txt(["save pth succuss (by all Dice)"], args.log_txt)

        avg_dice = val_metrics["avg_dice"] * 100
        if avg_dice >= avg_best_dice: 
            avg_best_dice = avg_dice
            torch.save(model.state_dict(), args.avg_best_model)
            print_to_terminal_and_txt(["save pth succuss ( by average Dice)"], args.log_txt)

        torch.save(model.state_dict(), args.final_model)
        np.save(args.val_data_dict, val_results)
        np.save(args.train_data_dict, train_results)
        

    print_to_terminal_and_txt(["train prdict result shape: ",
                                train_data["pred"].shape, 
                                "test prdict  result shape: ",
                                val_data["pred"].shape],
                                args.log_txt)
   

    plot_figure(train_results["dice"], "train Dice", args.train_dice_name)
    plot_figure(val_results["dice"], "val Dice",  args.test_dice_name)
    plot_loss_value(train_results["batch_loss"], val_results["avg_loss"], args.loss_name)


def save_test_pred():
    data_txt = ["33.txt", "34.txt", "35.txt", "36.txt", "37.txt", "38.txt", "39.txt", "40.txt"]
    # data_txt = ["33.txt"]

    for i in data_txt:
        dataset = MyDataset(args.root_dir, 
                        "../dataset_2D/txt/" + i,
                        shuffle = False, 
                        )

        loader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1)
        save_to_3d_data(args, model, device, loader, i.split(".")[0])



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.datetime.now()
    os.makedirs("./temp_weight/", exist_ok=True)


    print_to_terminal_and_txt("device: %s"%(device), args.log_txt)

    
    input_c = 1
    model = FASUNet(in_dim=input_c, n_classes=args.num_classes, init_c=args.channels).to(device)    
    
    model = BalancedDataParallel(8, model)
    # model = nn.DataParallel(model)


    optimizer = optim.SGD(model.parameters(), 
                    lr=args.lr, momentum=0.99, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)


    train_dataset = MyDataset(args.root_dir,
                            args.train_txt,
                            shuffle = True,
                            )
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers= 2 )

    val_dataset = MyDataset(args.root_dir, 
                            args.test_txt,
                            shuffle = False, 
                            )
    val_loader = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1)

    ##################################################### 训练
    main()

    ##################################################### 测试

    if not os.path.exists("./predict.txt"):
        os.system(r"touch {}".format("predict.txt"))    #调用系统命令行来创建文件
    

    print_to_terminal_and_txt(["test---------"*4], "predict.txt")
    hist, y, pred = save_2D_predict(args, "./visual_pred_2d/", model, 
                        device, val_loader, color_dict, "predict.txt")



    save_test_pred()
    
    