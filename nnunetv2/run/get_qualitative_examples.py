from sklearn.metrics import f1_score
from skimage import io
from tqdm import tqdm
import os.path

import pandas as pd
import matplotlib.pyplot as plt
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from glob import glob


ege_prediction_dir = "/afs/crc.nd.edu/user/y/ypeng4/EGE-UNet/" \
                     "results/670_egeunet_isic17/" \
                     "outputs/prediction"

my_pred_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/" \
              "trained_models/Dataset122_ISIC2017/" \
              "ISICTrainer__nnUNetPlans__2d/" \
              "456324_my_unet_FusedMBConv_16/" \
              "fold_0"

unet_pred_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/" \
                "trained_models/Dataset122_ISIC2017/" \
                "ISICTrainer__nnUNetPlans__2d/" \
                "458818_my_unet_FusedMBConv_16/fold_0"

unetplus_pred_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/" \
                    "trained_models/Dataset122_ISIC2017/" \
                    "ISICTrainer__nnUNetPlans__2d/" \
                    "458819_my_unet_FusedMBConv_16/" \
                    "fold_0"

gt_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/" \
         "preprocessed_data/Dataset122_ISIC2017/" \
         "gt_segmentations"

# io.imread('/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset122_ISIC2017/gt_segmentations/ISIC_0009995.png')

my_pred_files = subfiles(join(my_pred_dir, "validation"), suffix=".png")
unet_pred_files = subfiles(join(unet_pred_dir, "validation"), suffix=".png")
unetplus_pred_files = subfiles(join(unetplus_pred_dir, "validation"), suffix=".png")
ege_prediction_files = subfiles(join(ege_prediction_dir), suffix=".jpg")
gt_files = subfiles(gt_dir, suffix=".png")

# for gt in gt_files:
#     io.imread(gt)

cases = [os.path.basename(x)[:-len(".jpg")] for x in my_pred_files]

print(len(my_pred_files), len(unetplus_pred_files),
      len(unet_pred_files), len(ege_prediction_files),
      len(gt_files))

results = {"filename": [], "my_socre": [], "unet_score": [],
           "unetplus_score": [], "ege_score": []}
import numpy as np

for case in tqdm(cases):
    my_pred_arr = io.imread(join(my_pred_dir, "validation",
                                 f"{case}.png"))

    unet_pred_arr = io.imread(join(unet_pred_dir, "validation",
                                   f"{case}.png"))

    unetplus_pred_arr = io.imread(join(unetplus_pred_dir, "validation",
                                       f"{case}.png"))

    egeunet_pred_arr = io.imread(join(ege_prediction_dir,
                                      f"{case}.jpg"))
    if egeunet_pred_arr.max() > 1:
        egeunet_pred_arr = egeunet_pred_arr / 255
    msk_pred = egeunet_pred_arr > 0.5

    gt_arr = io.imread(join(gt_dir, f"{case}.png"))

    # print("my", np.unique(my_pred_arr))
    # print("gt", np.unique(gt_arr))
    # print("unetplus", np.unique(unetplus_pred_arr))
    # print("egeunet", np.unique(egeunet_pred_arr))
    my_socre = f1_score(y_true=gt_arr.flatten(), y_pred=my_pred_arr.flatten())

    unet_score = f1_score(y_true=gt_arr.flatten(), y_pred=unet_pred_arr.flatten())

    unetplus_score = f1_score(y_true=gt_arr.flatten(), y_pred=unetplus_pred_arr.flatten())

    ege_score = f1_score(y_true=gt_arr.flatten(), y_pred=msk_pred.flatten())

    results['filename'].append(case)
    results['my_socre'].append(my_socre)
    results['unet_score'].append(unet_score)
    results['unetplus_score'].append(unetplus_score)
    results['ege_score'].append(ege_score)


pd.DataFrame(results).to_csv("result.csv", index=False)

    # results = {"filename": [], "my_socre": [], "unet_score": [],
    #            "unetplus_score": [], "ege_score": []}





# best_pth_file = join(my_pred_dir, "checkpoint_best.pth")
# latest_pth_file = join(my_pred_dir, "checkpoint_final.pth")
#
# best_pth = torch.load(best_pth_file)
# print(best_pth.keys())
# print(best_pth['current_epoch'])

