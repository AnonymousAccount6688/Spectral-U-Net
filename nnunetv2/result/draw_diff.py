from einops import repeat
import numpy as np
import pandas as pd

df = pd.read_csv("slices.csv", header=0)

img_dir = "/scratch365/ypeng4/data/raw_data/Dataset003_Cirrus/imagesTr"
gt_dir = "/scratch365/ypeng4/data/raw_data/Dataset003_Cirrus/labelsTr"

pred_base_dir = "/scratch365/ypeng4/data/result/Dataset003_Cirrus"

methods = ['88.84', '89.18', '89.38', '92.42']

from batchgenerators.utilities.file_and_folder_operations import *

import SimpleITK as sitk
from skimage import io

for i in range(len(df)):
    name = df.loc[i, "name"]
    slice_idx = int(name[-3:])
    case = name[:-4]

    img_file = join(img_dir, f"{case}_0000.nii.gz")
    gt_file = join(gt_dir, f"{case}.nii.gz")

    pred1 = join(pred_base_dir, str(methods[0]), f"{case}.nii.gz")
    pred2 = join(pred_base_dir, str(methods[1]), f"{case}.nii.gz")
    pred3 = join(pred_base_dir, str(methods[2]), f"{case}.nii.gz")
    pred4 = join(pred_base_dir, str(methods[3]), f"{case}.nii.gz")

    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

    pred1_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred1))
    pred2_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred2))
    pred3_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred3))
    pred4_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred4))

    res = np.zeros((256, 256*6))

    res[:, 256*0:256*1] = img_arr[slice_idx]
    res[:, 256*1:256 * 2] = (gt_arr[slice_idx] != 0) * 255

    res[:, 256*2:256 * 3] = (pred1_arr[slice_idx] != 0) * 255
    res[:, 256*3:256 * 4] = (pred2_arr[slice_idx] != 0) * 255
    res[:, 256*4:256 * 5] = (pred3_arr[slice_idx] != 0) * 255
    res[:, 256*5:256 * 6] = (pred4_arr[slice_idx] != 0) * 255

    io.imsave(f"imgs/{name}.png", repeat(res, 'h w->h w c', c=3))