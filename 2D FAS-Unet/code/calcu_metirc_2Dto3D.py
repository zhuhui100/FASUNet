import sys
sys.path.append("../")
import os

from tool.my_metric import (fast_hist,my_calculate_metric,
    calculate_metric_medpy, get_metrics_sitk, class_dice)

import numpy as np
import datetime


def calcu_metric(gt, pred, n_class, txt_path, voxelspacing):


    start_time = datetime.datetime.now()

    hist_my = fast_hist(label_true= gt, label_pred = pred, n_class = n_class)
    dict_my = my_calculate_metric(hist_my)

    dict_medpy = calculate_metric_medpy(gt=gt, pred=pred, n_class = n_class, voxelspacing=voxelspacing)
    txt_file = open(txt_path, "a")


    print("\n\nresult", file=txt_file)

    for key,value in dict_medpy.copy().items():
        print(key,value,file=txt_file)

    end_time = datetime.datetime.now()
    print("start time", start_time, "\nend time", end_time, file=txt_file)
    
    
    txt_file.close()

    results = dict_medpy
    results["hist"] = dict_my["hist"]

    return results




if __name__ == "__main__":
    
    import glob

    start_time = datetime.datetime.now()
    n_class = 5
    data_dir = "./person_pred_3d-64"
    txt_path = "./metric_results_16.txt"

    voxelspacing=[
                    [0.9765625, 0.9765625, 2.0],
                    [0.976562, 0.976562, 2.5],
                    [1.3671875, 1.3671875, 2.0],
                    [0.9765625, 0.9765625, 2.0],
                    [0.976562, 0.976562, 2.5],
                    [0.976562, 0.976562, 2.5],
                    [0.9765625, 0.9765625, 2.0],
                    [0.976562, 0.976562, 2.5],
                    ]

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



    dc_results, sen_results, hd95_results = [], [], []
    jc_results = []
    pre_results = []
    assd_results = []

    hist = np.zeros((n_class, n_class))


    for i, path in  enumerate(data_path):
    
        gt_path = path[0]
        pred_path = path[1]

        gt = np.array(np.load(gt_path),np.int32).squeeze()
        pred = np.array(np.load(pred_path),np.int32).squeeze()


        results = calcu_metric(gt=gt, pred=pred, n_class = n_class, 
                txt_path=txt_path, voxelspacing=voxelspacing[i])

        dc_results.append(results['dice'])
        sen_results.append(results['sen'])
        hd95_results.append(results['hd95'])
        jc_results.append(results["jc"])
        pre_results.append(results["precision"])
        assd_results.append(results["assd"])

        hist += np.array(results['hist'])

        print(gt_path)
        


    dc_results = np.array(dc_results)
    sen_results = np.array(sen_results)
    hd95_results = np.array(hd95_results)
    assd_results = np.array(assd_results)
    pre_results = np.array(pre_results)
    jc_results = np.array(jc_results)

    all_data_dice = class_dice(hist)

    end_time = datetime.datetime.now()


    txt_file = open(txt_path, "a")
    

    mean_dice_sample = np.mean(dc_results, axis=0)
    std_dice_sample = np.std(dc_results, axis=0)

    mean_dice_avg = np.mean(mean_dice_sample[1:])
    mean_dice_sample = np.round(mean_dice_sample * 100, 2)
    std_dice_sample = np.round(std_dice_sample * 100, 2)
    mean_dice_avg = np.round(mean_dice_avg *100, 2)

    mean_sen_sample = np.mean(sen_results, axis=0)
    std_sen_sample = np.std(sen_results, axis=0)
    mean_sen_avg = np.mean(mean_sen_sample[1:]) 
    mean_sen_sample = np.round(mean_sen_sample * 100, 2)
    std_sen_sample = np.round(std_sen_sample *100, 2)
    mean_sen_avg = np.round(mean_sen_avg *100, 2)

    mean_jc_sample = np.mean(jc_results, axis=0)
    std_jc_sample = np.std(jc_results, axis=0)
    mean_jc_avg = np.mean(mean_jc_sample[1:])
    mean_jc_sample = np.round(mean_jc_sample*100, 2)
    std_jc_sample = np.round(std_jc_sample*100, 2)
    mean_jc_avg = np.round(mean_jc_avg*100, 2)

    mean_pre_sample = np.mean(pre_results, axis=0)
    std_pre_sample = np.std(pre_results, axis=0)
    mean_pre_avg = np.mean(mean_pre_sample[1:])
    mean_pre_sample = np.round(mean_pre_sample * 100, 2)
    std_pre_sample = np.round(std_pre_sample * 100, 2)
    mean_pre_avg = np.round(mean_pre_avg * 100, 2)

    mean_hd95_sample = np.mean(hd95_results, axis=0)
    std_hd95_sample = np.std(hd95_results, axis=0)
    mean_hd95_avg = np.mean(mean_hd95_sample[1:])
    mean_hd95_sample = np.round(mean_hd95_sample, 2)
    std_hd95_sample = np.round(std_hd95_sample, 2)
    mean_hd95_avg = np.round(mean_hd95_avg, 2)

    mean_assd_sample = np.mean(assd_results, axis=0)
    std_assd_sample = np.std(assd_results, axis=0)
    avg_hd95 = np.mean(mean_assd_sample[1:])
    mean_assd_sample = np.round(mean_assd_sample, 2)
    std_assd_sample = np.round(std_assd_sample, 2)
    avg_assd = np.round(avg_hd95, 2)



    ##################################################################

    print("\n\n\n",
        "==========================================================================\n",
        "\nwhole dice: ", all_data_dice,
        "\n\n", # 
        "\n\n\n1 Dice \nmean:\n", 
        mean_dice_sample[1], "$\pm$", std_dice_sample[1],"\n",
        mean_dice_sample[2], "$\pm$", std_dice_sample[2],"\n",
        mean_dice_sample[3], "$\pm$", std_dice_sample[3],"\n",
        mean_dice_sample[4], "$\pm$", std_dice_sample[4],"\n",
        mean_dice_avg,"\n",

        "\n\n2 (sen): :\n", 
        mean_sen_sample[1], "$\pm$", std_sen_sample[1], "\n",
        mean_sen_sample[2], "$\pm$", std_sen_sample[2], "\n",
        mean_sen_sample[3], "$\pm$", std_sen_sample[3], "\n",
        mean_sen_sample[4], "$\pm$", std_sen_sample[4], "\n",
        mean_sen_avg, "\n",

        "\n\n3 95HD: :\n", 
        mean_hd95_sample[1], "$\pm$", std_hd95_sample[1],"\n",
        mean_hd95_sample[2], "$\pm$", std_hd95_sample[2],"\n",
        mean_hd95_sample[3], "$\pm$", std_hd95_sample[3],"\n",
        mean_hd95_sample[4], "$\pm$", std_hd95_sample[4],"\n",
        mean_hd95_avg,

        "\n\n4 ASSD:\n", 
        mean_assd_sample[1], "$\pm$", std_assd_sample[1],"\n",
        mean_assd_sample[2], "$\pm$", std_assd_sample[2],"\n",
        mean_assd_sample[3], "$\pm$", std_assd_sample[3],"\n",
        mean_assd_sample[4], "$\pm$", std_assd_sample[4],"\n",
        avg_assd,

        "\n\n5 Preci:\n", 
        mean_pre_sample[1], "$\pm$", std_pre_sample[1],"\n",
        mean_pre_sample[2], "$\pm$", std_pre_sample[2],"\n",
        mean_pre_sample[3], "$\pm$", std_pre_sample[3],"\n",
        mean_pre_sample[4], "$\pm$", std_pre_sample[4],"\n",
        mean_pre_avg,

        "\n\n6 jc::\n", 
        mean_jc_sample[1], "$\pm$", std_jc_sample[1],"\n",
        mean_jc_sample[2], "$\pm$", std_jc_sample[2],"\n",
        mean_jc_sample[3], "$\pm$", std_jc_sample[3],"\n",
        mean_jc_sample[4], "$\pm$", std_jc_sample[4],"\n",
        mean_jc_avg,

        file=txt_file
    )

    txt_file.close()
    



