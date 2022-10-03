import sys
sys.path.append("../")
import os

from data_tool.my_metrics import (fast_hist, my_calculate_metric,
    calculate_metric_medpy, class_dice)

import numpy as np
import datetime



def calcu_metric(gt, pred, n_class, txt_path, voxelspacing):

    start_time = datetime.datetime.now()


    hist_my = fast_hist(label_true= gt, label_pred = pred, n_class = n_class)
    dict_my = my_calculate_metric(hist_my)

    dict_medpy = calculate_metric_medpy(gt=gt, pred=pred, n_class = n_class, voxelspacing=voxelspacing)

    
    txt_file = open(txt_path, "a")
    print("medpy", file=txt_file)

    for key,value in dict_medpy.copy().items():
        print(key,value,file=txt_file)

    end_time = datetime.datetime.now()
    print("start time", start_time, "\n end time", end_time, "\n\n", file=txt_file)
    
    txt_file.close()
    

    results = dict_medpy
    results["hist"] = dict_my["hist"]


    return results



if __name__ == "__main__":
    
    import glob
    voxelspacing={
                   "05": [0.75, 0.75, 3.2],   # 5
                    "18": [0.68, 0.68, 3.2],      # 18
                    "24": [0.63, 0.63, 2.0],      # 24
                    "29": [0.68, 0.68, 2.0]       # 29
    }

    
    n_class = 2
    exp_id = str(1)        
    data_dir_list = ["./3D_predict_val_%s_no/npy"%(exp_id), "./3D_predict_val_%s_post/npy"%(exp_id)]

    txt_path_list = ["./metric_results_%s_no.txt"%(exp_id), "./metric_results_%s_post.txt"%(exp_id)]

    for data_dir, txt_path in zip(data_dir_list, txt_path_list):
        start_time = datetime.datetime.now()
        data_path = []
    

        gt_path_list = glob.glob(r"%s/*_gt.npy"%(data_dir))
        gt_path_list = sorted(gt_path_list)    
        pred_path_list = [x.replace("_gt","_pred") for x in gt_path_list]
        
        for i in range(len(gt_path_list)):
            data_path.append([gt_path_list[i], pred_path_list[i]])


        if os.path.exists(txt_path):
            # os.remove("./output/metric_results.txt")
            with open(txt_path, "w") as f:
                f.truncate()


        dc_results,  jc_results  = [], []
        hd95_results = []
        pre_results, sen_results = [], []
        assd_results = []


        hist = np.zeros((n_class, n_class))


        for i, path in  enumerate(data_path):
            
            gt_path = path[0]
            pred_path = path[1]
            data_id = gt_path.split("/")[-1].split("_")[0]


            gt = np.array(np.load(gt_path),np.int32).squeeze()
            pred = np.array(np.load(pred_path),np.int32).squeeze()

            with open(txt_path, "a") as f:
                print(path, file= f)
            results = calcu_metric(gt=gt, pred=pred, n_class = n_class, 
                    txt_path=txt_path,voxelspacing=voxelspacing[data_id])

            dc_results.append(results['dice'])
            sen_results.append(results['sen'])
            hd95_results.append(results['hd95'])
            jc_results.append(results["jc"])
            pre_results.append(results["precision"])
            assd_results.append(results["assd"])


            hist += np.array(results['hist'])

            print(gt_path)
            

        end_time = datetime.datetime.now()


        dc_results = np.array(dc_results)
        sen_results = np.array(sen_results)
        hd95_results = np.array(hd95_results)
        jc_results = np.array(jc_results)
        pre_results = np.array(pre_results)
        assd_results = np.array(assd_results)


        all_data_dice = class_dice(hist)

        
        ############################################### 
        mean_dice_sample = np.mean(dc_results, axis=0)
        std_dice_sample = np.std(dc_results, axis=0)
        mean_dice_avg = mean_dice_sample[1]
        mean_dice_sample = np.round(mean_dice_sample * 100, 2)
        std_dice_sample = np.round(std_dice_sample * 100, 2)
        mean_dice_avg = np.round(mean_dice_avg *100, 2)

        mean_sen_sample = np.mean(sen_results, axis=0)
        std_sen_sample = np.std(sen_results, axis=0)
        mean_sen_avg = mean_sen_sample[1]
        mean_sen_sample = np.round(mean_sen_sample * 100, 2)
        std_sen_sample = np.round(std_sen_sample *100, 2)
        mean_sen_avg = np.round(mean_sen_avg *100, 2)

        mean_pre_sample = np.mean(pre_results, axis=0)
        std_pre_sample = np.std(pre_results, axis=0)
        mean_pre_avg = mean_pre_sample[1]
        mean_pre_sample = np.round(mean_pre_sample * 100, 2)
        std_pre_sample = np.round(std_pre_sample * 100, 2)
        mean_pre_avg = np.round(mean_pre_avg * 100, 2)

        mean_jc_sample = np.mean(jc_results, axis=0)
        std_jc_sample = np.std(jc_results, axis=0)
        mean_jc_avg = mean_jc_sample[1]
        mean_jc_sample = np.round(mean_jc_sample*100, 2)
        std_jc_sample = np.round(std_jc_sample*100, 2)
        mean_jc_avg = np.round(mean_jc_avg*100, 2)


        mean_hd95_sample = np.mean(hd95_results, axis=0)
        std_hd95_sample = np.std(hd95_results, axis=0)
        mean_hd95_avg = mean_hd95_sample[1]
        mean_hd95_sample = np.round(mean_hd95_sample, 2)
        std_hd95_sample = np.round(std_hd95_sample, 2)
        mean_hd95_avg = np.round(mean_hd95_avg, 2)


        mean_assd_sample = np.mean(assd_results, axis=0)
        std_assd_sample = np.std(assd_results, axis=0)
        avg_assd = mean_assd_sample[1]
        mean_assd_sample = np.round(mean_assd_sample, 2)
        std_assd_sample = np.round(std_assd_sample, 2)
        avg_assd = np.round(avg_assd, 2)

            


        txt_file = open(txt_path, "a")
        print("\n\n\n",
            "==========================================================================\n",
            "\n all dice: ", all_data_dice,
            "\n\n", 
            "\n\n\n1 mean Dice :\n", 
            mean_dice_sample[1], "$\pm$", std_dice_sample[1],"\n",
            mean_dice_avg,"\n",

            "\n\n2 mean Sen:\n", 
            mean_sen_sample[1], "$\pm$", std_sen_sample[1], "\n",
            mean_sen_avg, "\n",

            "\n\n3 mean Pre:\n", 
            mean_pre_sample[1], "$\pm$", std_pre_sample[1],"\n",
            mean_pre_avg,


            "\n\n4 95HD: \n", 
            mean_hd95_sample[1], "$\pm$", std_hd95_sample[1],"\n",
            mean_hd95_avg,


            "\n\n5 ASSD:\n", 
            mean_assd_sample[1], "$\pm$", std_assd_sample[1],"\n",
            avg_assd,

            "\n\n6 jc:\n", 
            mean_jc_sample[1], "$\pm$", std_jc_sample[1],"\n",
            mean_jc_avg,

            file=txt_file
        )

        txt_file.close()



