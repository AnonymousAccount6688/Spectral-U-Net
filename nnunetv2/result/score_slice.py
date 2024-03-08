import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *

gt_dir = "/scratch365/ypeng4/data/raw_data/Dataset003_Cirrus/labelsTr"

pred_base_dir = "/scratch365/ypeng4/data/result/Dataset003_Cirrus"

methods = ['88.84', '89.18', '89.38', '92.42']

cases = subfiles(join(pred_base_dir, methods[0]), suffix=".nii.gz", join=False)

name = []
method1_irf = []
method2_irf = []
method3_irf = []
method4_irf = []

method1_srf = []
method2_srf = []
method3_srf = []
method4_srf = []

method1_ped = []
method2_ped = []
method3_ped = []
method4_ped = []

srf = []
ped = []


def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.0

    return dc

for i, m in enumerate(methods):

    for case in cases:
        gt_file = join(gt_dir, case)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

        # plt.imshow(gt_arr[20], cmap="gray"); plt.show()
        dc_scores = []

        pred_file = join(pred_base_dir, m, case)
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))

        for s_idx in range(pred_arr.shape[0]):
            gt_slice = gt_arr[s_idx]
            pred_slice = pred_arr[s_idx]

            if i == 0:
                name.append(f"{case.split('.')[0]}_{s_idx:0>3}")

            for j in range(1, 4):
                dc_score = dc(pred_slice==j, gt_slice==j)
                if i == 0:
                    if j == 1:
                        method1_irf.append(dc_score)
                    elif j == 2:
                        method1_srf.append(dc_score)
                    elif j == 3:
                        method1_ped.append(dc_score)
                elif i == 1:
                    if j == 1:
                        method2_irf.append(dc_score)
                    elif j == 2:
                        method2_srf.append(dc_score)
                    elif j == 3:
                        method2_ped.append(dc_score)

                elif i == 2:
                    if j == 1:
                        method3_irf.append(dc_score)
                    elif j == 2:
                        method3_srf.append(dc_score)
                    elif j == 3:
                        method3_ped.append(dc_score)
                elif i == 3:
                    if j == 1:
                        method4_irf.append(dc_score)
                    elif j == 2:
                        method4_srf.append(dc_score)
                    elif j == 3:
                        method4_ped.append(dc_score)


method1_mean = []
for i in range(len(method1_irf)):
    method1_mean.append(np.mean([method1_irf[i], method1_srf[i], method1_ped[i]]))

method2_mean = []
for i in range(len(method2_irf)):
    method2_mean.append(np.mean([method2_irf[i], method2_srf[i], method2_ped[i]]))

method3_mean = []
for i in range(len(method3_irf)):
    method3_mean.append(np.mean([method3_irf[i], method3_srf[i], method3_ped[i]]))

method4_mean = []
for i in range(len(method4_irf)):
    method4_mean.append(np.mean([method4_irf[i], method4_srf[i], method4_ped[i]]))

df = pd.DataFrame({"name": name,
              "method1_irf":method1_irf, "method2_irf":method2_irf,
              "method3_irf":method3_irf, "method4_irf":method4_irf,
              "method1_srf": method1_srf, "method2_srf":method2_srf,
              "method3_srf":method3_srf, "method4_srf":method4_srf,
              "method1_ped":method1_ped, "method2_ped":method2_ped,
              "method3_ped":method3_ped, "method4_ped":method4_ped,
                   "method1_mean": method1_mean, "method2_mean": method2_mean,
                   "method3_mean": method3_mean, "method4_mean": method4_mean
              })

df.to_csv("slice_result.csv", index=False)


