#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../")

from trainer_patch_chaos import train, validation, save_3D_prediction_to_npy,visual_volume_pred_gt
from fasunet_3d import FASUNet_3D

import torch
import torch.nn as nn
import warnings
import argparse
import os
import datetime
import numpy as np
from data_tool.draw import plot_figure, plot_loss_value
from data_tool.chaos_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"




parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--num-classes", type=int, default=2,
                    help="2 for CHAOS datasets")

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--model-dir", type=str, default="./weight/")
parser.add_argument("--output_dir", type=str, default="./output/")
parser.add_argument("--root_dir", type=str, default=".",
                    help="")

t = str("13_1")
box = str(380)
channels= 32
parser.add_argument("--train-txt", type=str,
                    default="../dataset/overlap_patch_from_380/train_128x128x64_%sv2.txt"%(box))
parser.add_argument("--val-txt", type=str,
                    default="../dataset/overlap_patch_from_380/val_128x128x64_%s.txt"%(box))

parser.add_argument("--epochs", type=int, default=150, help="default: 150")
parser.add_argument("--batch-size", type=int, default=4)


parser.add_argument("--best_model",
                    type=str, default="./weight/%s_overlap_fas_c%d_352_uf_%s_v2.pth"%(t, channels, box))
parser.add_argument("--final-model",
                    type=str, default="./weight/%s_overlap_fas-c%d_352_uf_final_%s_v2.pth"%(t, channels, box))
parser.add_argument("--output_txt",
                    type=str, default="./output/%s_fas-c%d_352_uf_v2.txt"%(t, channels))
parser.add_argument("--train_dice_name",
                    type=str, default="./output/%s_fas-c%d_352_train_dice_v2.png"%(t, channels))
parser.add_argument("--val_dice_name",
                    type=str, default="./output/%s_fas-c%d_352_val_dice_v2.png"%(t, channels))
parser.add_argument("--loss_name",
                    type=str, default="./output/%s_fas-c%d_352_uf_loss_v2.png"%(t, channels))

parser.add_argument("--val_data_dict", 
                    type=str, default = "./output/%s-fas_c%d_352_val_%s_v2.npy"%(t, channels, box))
parser.add_argument("--train_data_dict", 
                    type=str, default = "./output/%s-fas_c%d_352_train_%s_v2.npy"%(t, channels, box))

parser.add_argument("--information", type=str, default="{3,5,2}")
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
np.set_printoptions(formatter={'all': lambda x: "%5.2f" % x})
train_loss, train_dice, test_loss, test_dice = [], [], [], []


color_dict = {
    (0, 0, 0):     0,
    (255, 0, 0): 255
}


def print_measure(epoch, train_info, test_info):


    now = datetime.datetime.now()
    message = ['{}/{}'.format(epoch, args.epochs), 
            "train: loss=%6.4f "%(train_info["batch_loss"]), 
            "dice=", np.round(train_info["batch_dice"] *100 , 2), 
            ", \nval: "
            "loss=%6.4f"%(test_info["avg_loss"]), 
            "dice=", np.round(test_info["dice"] *100 , 2), 
            "\n",
            now.strftime("%H:%M:%S"),
            "\n"
    ]

    return message 
    



def print_to_terminal_and_txt(message, output_txt_file, term = True):
    """
    message: list or string
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
    best_dice = 0

    for k,v in sorted(vars(args).items()):
        print_to_terminal_and_txt([k,'=', v], args.output_txt)
    print_to_terminal_and_txt("\n\n", args.output_txt)

    print_info = [
          "python file name:", os.path.basename(sys.argv[0]), 
          "\nmodel name:", args.best_model, 
          "\n%s\n" % (args.information),
          "train patch:%d, validataion patch: %d" % (len(train_dataset), len(val_loader))
          ]
    print_info = "".join(print_info)
    print_to_terminal_and_txt(print_info, args.output_txt)
  

    val_result = {"avg_loss": [], "dice":[]}
    train_result = {"epoch_loss": [], "batch_loss": [], "batch_dice":[]}
  

    for epoch in range(args.epochs):

        now = datetime.datetime.now()

        train_output, train_metrics =  train(args, model, device, 
              criterion, optimizer, epoch, train_loader)

        val_out_img, val_metrics = validation(args, model, device, criterion, epoch, val_loader)
      
    
        train_result["batch_loss"].append(train_metrics["batch_loss"])
        train_result["batch_dice"].append(train_metrics["batch_dice"])

        val_result["avg_loss"].append(val_metrics["avg_loss"])
        val_result["dice"].append(val_metrics["dice"])

        message = print_measure(epoch, train_metrics, val_metrics)
        print_to_terminal_and_txt(message, args.output_txt)

        try:
            torch.save(model.state_dict(), "./temp_weight/%s_fas_%s.pth"%(t, str(epoch)))
        except OSError:
            print('save pth faile T_T')    

        if val_metrics["dice"][-1] *100 >= best_dice: 
            best_dice = val_metrics["dice"][-1] *100
            torch.save(model.state_dict(), args.best_model)
            print_to_terminal_and_txt(["save pth success..."], args.output_txt)

        torch.save(model.state_dict(), args.final_model)
        np.save(args.val_data_dict, val_result)
        np.save(args.train_data_dict, train_result)

    print_to_terminal_and_txt(["train prdict result shape: ",
                                train_output["pred"].shape, 
                                "test prdict  result shape: ",
                                val_out_img["pred"].shape],
                                args.output_txt)

  

    plot_figure(train_result["batch_dice"], "train dice", args.train_dice_name)
    plot_figure(val_result["dice"], "val dice",  args.val_dice_name)
    plot_loss_value(train_result["batch_loss"], val_result["avg_batch_loss"], args.loss_name)



if __name__ == "__main__":

    start_time = datetime.datetime.now()
    print_to_terminal_and_txt(["run code","\ndevice:--->", device], args.output_txt)


    input_c = 1
    model = FASUNet_3D(input_c, args.num_classes, channels).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_tranforms = None
    val_tranforms = None

    train_dataset = MyDataset(args.root_dir,
                              args.train_txt,
                              shuffle=True,
                              transform=train_tranforms,
                              )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

    val_dataset = MyDataset(args.root_dir,
                            args.val_txt,
                            shuffle=False,
                            transform=val_tranforms)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)

    main()
  



    test_log_file = "./fas_test_log_380.txt"
    post_process_list = [False, True]
    for post_process in post_process_list:
        if post_process:
            post_sign = "post"
        else:
            post_sign = "no"
        npy_save_dir = "./3D_predict_val_%s_%s/"%(t, post_sign)

        print_to_terminal_and_txt("validation ----------------"*3, args.output_txt)

        save_3D_prediction_to_npy(args, npy_save_dir, model, 
                device, val_loader, args.output_txt, post_process=post_process)

        print_to_terminal_and_txt("prediction finish " + "\n"*3, args.output_txt)

    
    visual_volume_pred_gt("./3D_predict_val_1_post/npy/", "fas_visual_val_post", data_type = "npy")
    end_time = datetime.datetime.now()
