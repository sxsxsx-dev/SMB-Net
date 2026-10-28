import os
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.misc
import medpy.io as medio
import matplotlib.pyplot as plt
from matplotlib import colors
import concurrent.futures

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


def _add_zoom_inset(ax, img_slice, mask_slice, cmap, norm, bbox):
    """
    辅助函数：向给定的子图(ax)添加一个右上角的放大图，
    并在主图上用红色虚线框标记放大区域。
    """
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = bbox

    # 创建放大图的子区域
    axins = zoomed_inset_axes(ax, zoom=4, loc='upper right')

    # 在放大图中显示图像和分割蒙版
    axins.imshow(np.zeros_like(img_slice), cmap='gray', interpolation='none')
    axins.imshow(mask_slice, cmap=cmap, norm=norm, alpha=1.0, interpolation='none')

    # 设置放大区域的坐标范围
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)

    # 隐藏放大图的坐标刻度
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)
    axins.tick_params(axis='both', which='both', length=0)

    # 在主图(ax)上绘制红色虚线框以标记放大位置
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=1.2,
                     edgecolor='red',
                     facecolor='none',
                     linestyle='--')
    ax.add_patch(rect)


def visualize_results(input_tensor, label_tensor, pred_tensor, case_name, mask_name, save_dir='visual_results_2d'):
    """
    (此函数被异步调用)
    可视化单个病例的结果。
    为原始图像、真实标签和预测结果分别保存为独立的图像文件。
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors

    base_dir = os.path.join(save_dir, mask_name, case_name)
    os.makedirs(base_dir, exist_ok=True)

    input_np = input_tensor.cpu().float().numpy()
    label_np = label_tensor.cpu().numpy()
    pred_np = pred_tensor.cpu().numpy()

    # 色彩映射
    # 标签 1 (坏死/非增强): 红色
    # 标签 2 (水肿): 绿色
    # 标签 3/4 (增强): 黄色
    cmap = colors.ListedColormap(['black', 'red', 'green', 'yellow', 'yellow'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    slices = {
        'axial': (input_np.shape[2] // 2, 2),
        'sagittal': (input_np.shape[3] // 2, 1),
        'coronal': (input_np.shape[1] // 2, 0)
    }

    available_mods = [i for i in range(input_np.shape[0]) if input_np[i].max() > 0]
    available_mods = available_mods if available_mods else [0]

    for mod_idx in available_mods:
        mod_name = ['FLAIR', 'T1', 'T1ce', 'T2'][mod_idx]

        for plane, (slice_idx, dim) in slices.items():
            if dim == 0:
                img_slice = np.rot90(input_np[mod_idx, slice_idx, :, :])
                label_slice = np.rot90(label_np[slice_idx, :, :])
                pred_slice = np.rot90(pred_np[slice_idx, :, :])
            elif dim == 1:
                img_slice = np.rot90(input_np[mod_idx, :, slice_idx, :])
                label_slice = np.rot90(label_np[:, slice_idx, :])
                pred_slice = np.rot90(pred_np[:, slice_idx, :])
            else:
                img_slice = np.rot90(input_np[mod_idx, :, :, slice_idx])
                label_slice = np.rot90(label_np[:, :, slice_idx])
                pred_slice = np.rot90(pred_np[:, :, slice_idx])

            # --- 1. 保存原始图像 ---
            fig_raw, ax_raw = plt.subplots(figsize=(6, 6), facecolor='black')
            ax_raw.imshow(img_slice, cmap='gray')
            ax_raw.axis('off')
            plt.tight_layout(pad=0)
            save_path_raw = os.path.join(base_dir, f'{mod_name}_{plane}_raw.png')
            plt.savefig(save_path_raw, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_raw)

            # --- 2. 保存真实标签 ---
            masked_label = np.ma.masked_where(label_slice == 0, label_slice)
            fig_label, ax_label = plt.subplots(figsize=(6, 6), facecolor='black')
            ax_label.imshow(np.zeros_like(img_slice), cmap='gray') # 黑色背景
            ax_label.imshow(masked_label, cmap=cmap, norm=norm, alpha=1.0)
            ax_label.axis('off')
            plt.tight_layout(pad=0)
            save_path_label = os.path.join(base_dir, f'{mod_name}_{plane}_groundtruth.png')
            plt.savefig(save_path_label, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_label)

            # --- 3. 保存预测结果 ---
            masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            fig_pred, ax_pred = plt.subplots(figsize=(6, 6), facecolor='black')
            ax_pred.imshow(np.zeros_like(img_slice), cmap='gray') # 黑色背景
            ax_pred.imshow(masked_pred, cmap=cmap, norm=norm, alpha=1.0)
            ax_pred.axis('off')
            plt.tight_layout(pad=0)
            save_path_pred = os.path.join(base_dir, f'{mod_name}_{plane}_prediction.png')
            plt.savefig(save_path_pred, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_pred)

def test_softmax(
    test_loader, model, dataname="BRATS2018", feature_mask=None, mask_name=None
):
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
        num_cls = 4
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "ncr_net", "edema", "enhancing"
    elif dataname == "BRATS2015":
        num_cls = 5
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "necrosis", "edema", "non_enhancing", "enhancing"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i, data in enumerate(test_loader):
            target = data[1].cuda()
            x = data[0].cuda()
            names = data[-1]
            raw_input = data[0].cpu()

            if feature_mask is not None:
                mask = torch.from_numpy(np.array(feature_mask))
                mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            else:
                mask = data[2]
            mask = mask.cuda()
            _, _, H, W, Z = x.size()

            h_cnt = int(np.ceil((H - patch_size) / (patch_size * 0.5)))
            h_idx_list = [h_idx * int(patch_size * 0.5) for h_idx in range(h_cnt)]
            h_idx_list.append(H - patch_size)

            w_cnt = int(np.ceil((W - patch_size) / (patch_size * 0.5)))
            w_idx_list = [w_idx * int(patch_size * 0.5) for w_idx in range(w_cnt)]
            w_idx_list.append(W - patch_size)

            z_cnt = int(np.ceil((Z - patch_size) / (patch_size * 0.5)))
            z_idx_list = [z_idx * int(patch_size * 0.5) for z_idx in range(z_cnt)]
            z_idx_list.append(Z - patch_size)

            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
            model.module.is_training = False
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                        pred_part = model(x_input, mask)
                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
            pred = pred / weight
            pred = pred[:, :, :H, :W, :T]
            pred = torch.argmax(pred, dim=1)

            for k in range(len(names)):
                executor.submit(
                    visualize_results,
                    input_tensor=raw_input[k],
                    label_tensor=target[k].cpu(),
                    pred_tensor=pred[k].detach().cpu(),
                    case_name=names[k],
                    mask_name=mask_name or 'full',
                    save_dir=os.path.join('./out/vis', 'visual_results_2d')
                )

            if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
                scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            elif dataname == "BRATS2015":
                scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

            for k, name in enumerate(names):
                msg = f"Subject {i+1}/{len(test_loader)}, {k+1}/{len(names)}"
                msg += f"{name:>20}, "
                vals_separate.update(scores_separate[k])
                vals_evaluation.update(scores_evaluation[k])
                msg += ", ".join([f"{k}: {v:.4f}" for k, v in zip(class_evaluation, scores_evaluation[k])])
                logging.info(msg)

    msg = "Average scores:"
    msg += ", ".join([f"{k}: {v:.4f}" for k, v in zip(class_evaluation, vals_evaluation.avg)])
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