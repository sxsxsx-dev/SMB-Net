import os
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.misc
import medpy.io as medio
from torch.cuda.amp import autocast

cudnn.benchmark = True

path = os.path.dirname(__file__)

patch_size = 128


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
        o4 = o3 * 0.0
    else:
        o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = (
        torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    )
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = (
        torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    )
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (
            torch.unsqueeze(ncr_net_dice, 1),
            torch.unsqueeze(edema_dice, 1),
            torch.unsqueeze(enhancing_dice, 1),
        ),
        dim=1,
    )
    dice_evaluate = torch.cat(
        (
            torch.unsqueeze(dice_whole, 1),
            torch.unsqueeze(dice_core, 1),
            torch.unsqueeze(enhancing_dice, 1),
            torch.unsqueeze(enhancing_dice_postpro, 1),
        ),
        dim=1,
    )

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1, 2, 3)) + eps
    denominator5 = torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t5, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = (
        torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    )
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = (
        torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    )
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (
            torch.unsqueeze(necrosis_dice, 1),
            torch.unsqueeze(edema_dice, 1),
            torch.unsqueeze(non_enhancing_dice, 1),
            torch.unsqueeze(enhancing_dice, 1),
        ),
        dim=1,
    )
    dice_evaluate = torch.cat(
        (
            torch.unsqueeze(dice_whole, 1),
            torch.unsqueeze(dice_core, 1),
            torch.unsqueeze(enhancing_dice, 1),
            torch.unsqueeze(enhancing_dice_postpro, 1),
        ),
        dim=1,
    )

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def test_softmax(
    test_loader, model, dataname="BRATS2020", feature_mask=None, mask_name=None
):
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()
    step = patch_size // 2  # 明确步长为整数

    if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
        num_cls = 4
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "ncr_net", "edema", "enhancing"
    elif dataname == "BRATS2015":
        num_cls = 5
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "necrosis", "edema", "non_enhancing", "enhancing"

    with torch.no_grad():  # 禁用梯度计算
        for i, data in enumerate(test_loader):
            target = data[1].cuda()
            x = data[0].cuda()
            names = data[-1]
            if feature_mask is not None:
                mask = torch.from_numpy(np.array(feature_mask))
                mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            else:
                mask = data[2]
            mask = mask.cuda()
            _, _, H, W, Z = x.size()

            # 生成滑动窗口索引
            def generate_indices(length):
                cnt = int(np.ceil((length - patch_size) / step))
                indices = [i * step for i in range(cnt)]
                indices.append(length - patch_size)
                return indices

            h_idx_list = generate_indices(H)
            w_idx_list = generate_indices(W)
            z_idx_list = generate_indices(Z)

            # 计算权重
            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            # 预测
            pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
            model.module.is_training = False
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                        # 使用autocast启用混合精度计算
                        with autocast():
                            pred_part = model(x_input, mask)
                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
            pred = pred / weight
            pred = pred[:, :, :H, :W, :T]
            pred = torch.argmax(pred, dim=1)

            # 计算 Dice 分数
            if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
                scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            elif dataname == "BRATS2015":
                scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

            # 记录结果
            for k, name in enumerate(names):
                msg = f"Subject {i+1}/{len(test_loader)}, {k+1}/{len(names)}, {name}"
                vals_separate.update(scores_separate[k])
                vals_evaluation.update(scores_evaluation[k])
                msg += ", " + ", ".join(f"{k}: {v:.4f}" for k, v in zip(class_evaluation, scores_evaluation[k]))
                logging.info(msg)

    # 输出平均结果
    msg = "Average scores: " + ", ".join(f"{k}: {v:.4f}" for k, v in zip(class_evaluation, vals_evaluation.avg))
    print(msg)
    model.train()
    return vals_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
