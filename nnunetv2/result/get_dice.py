import torch.nn.functional as F
import torch
import numpy as np
from medpy.metric.binary import hd95
import SimpleITK as sitk
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def dice_score_slice(y_pred, y_true, num_classes):
    # print(type(y_pred), type(y_true), y_pred.dtype, y_true.dtype)
    y_pred = F.one_hot(y_pred, num_classes=num_classes).to(torch.uint8)
    y_true = F.one_hot(y_true, num_classes=num_classes).to(torch.uint8)

    eps = 1e-4
    # tp = torch.sum(y_pred & y_true, dim=(1, 2))
    # pred_sum = torch.sum(y_pred, dim=(1, 2))
    # true_sum = torch.sum(y_true, dim=(1, 2))

    # precision = (tp + eps) / (pred_sum + eps)
    # recall = (tp + eps) / (true_sum + eps)

    FN = torch.sum((1 - y_pred) * y_true, dim=(1, 2))  # [0,0,0,0]
    FP = torch.sum((1 - y_true) * y_pred, dim=(1, 2))  # [0,0,0,1]
    Pred = y_pred
    GT = y_true
    inter = torch.sum(GT * Pred, dim=(1, 2))  # 0

    union = torch.sum(GT, dim=(1, 2)) + torch.sum(Pred, dim=(1, 2))  # 1
    dice = (2 * inter + eps) / (union + eps)

    # return 2 * (precision * recall) / (precision + recall)
    return dice


def dice_score_vol(y_pred, y_true, num_classes, voxel_spacing):
    y_pred = F.one_hot(y_pred, num_classes=num_classes).to(torch.uint8)
    y_true = F.one_hot(y_true, num_classes=num_classes).to(torch.uint8)
    eps = 1e-4

    # tp = torch.sum(y_pred & y_true, dim=(0, 1, 2))
    # pred_sum = torch.sum(y_pred, dim=(0, 1, 2))
    # true_sum = torch.sum(y_true, dim=(0, 1, 2))
    #
    # precision = (tp + eps) / (pred_sum + eps)
    # recall = (tp + eps) / (true_sum + eps)
    # return 2 * (precision * recall) / (precision + recall)

    FN = torch.sum((1 - y_pred) * y_true, dim=(0, 1, 2))  # [0,0,0,0]
    FP = torch.sum((1 - y_true) * y_pred, dim=(0, 1, 2))  # [0,0,0,1]
    Pred = y_pred
    GT = y_true
    inter = torch.sum(GT * Pred, dim=(0, 1, 2))  # 0

    union = torch.sum(GT, dim=(0, 1, 2)) + torch.sum(Pred, dim=(0, 1, 2))  # 1
    dice = (2 * inter + eps) / (union + eps)

    hd95s = []
    pred_npy = Pred.detach().cpu().numpy()
    gt_npy = GT.detach().cpu().numpy()
    for i in range(1, 4):
        if pred_npy[..., i].max() > 0 and gt_npy[..., i].max() > 0:
            hd95s.append(hd95(pred_npy[..., i], gt_npy[..., i],
                              voxelspacing=voxel_spacing))
        else:
            hd95s.append(np.nan)

    return dice, hd95s


def get_score(file_ref, file_pred, ref_reader, pred_reader):
    # print(file_ref+('+++++++++++'.center(20, "="))+file_pred)
    # print("++++".center(50, "="))
    seg_ref, seg_ref_dict = ref_reader.read_seg(seg_fname=file_ref)
    seg_pred, seg_pred_dict = pred_reader.read_seg(file_pred)

    seg_ref, seg_pred = seg_ref.squeeze(axis=0), seg_pred.squeeze(axis=0)

    # (C, H, W)
    assert seg_ref.shape == seg_pred.shape, f"invalid shape, seg: {seg_pred.shape}, ref: {seg_ref.shape}"

    # case_vol_avd = {1: [], 2: [], 3: []}
    # case_bacc = {1: [], 2: [], 3: []}

    seg_ref = torch.tensor(seg_ref, dtype=torch.int64)  # .cuda(1)  # (C, H, W)
    seg_pred = torch.tensor(seg_pred, dtype=torch.int64)  # .cuda(1)  # (C, H, W)

    # shape: (C, num_classes)
    # print(type(seg_ref), type(seg_pred))
    case_slice_dice = dice_score_slice(seg_pred, seg_ref, num_classes=4)

    # shape: (num_classes,)
    case_vol_dice, case_hd95 = dice_score_vol(seg_pred, seg_ref, num_classes=4,
                                              voxel_spacing=seg_ref_dict['spacing'])

    return case_slice_dice.detach().cpu().numpy(), \
        case_vol_dice.detach().cpu().numpy(), case_hd95


from batchgenerators.utilities.file_and_folder_operations import *

pred_dir = "/scratch365/ypeng4/data/result/Dataset003_Cirrus/RetinaTrainer__WaveUNet_nnUNetPlans__2d/fold_0/validation"
gt_dir = "/scratch365/ypeng4/data/raw_data/Dataset003_Cirrus/labelsTr"

# gts = join(pred_dir, "*.nii.gz")
preds = join(pred_dir, "*.nii.gz")
keys = subfiles(pred_dir, suffix=".nii.gz", join=False)


reader_gt = SimpleITKIO()
reader_pred = SimpleITKIO()

case_slices = []
from tqdm import tqdm
for k in tqdm(keys):
    case_slice_dc, case_vol_dc, hd95_score = get_score(
        join(gt_dir, k),
        join(pred_dir, k),
        reader_gt,
        reader_pred
    )

    case_slices.append(case_slice_dc)

print(np.mean(case_slices))

