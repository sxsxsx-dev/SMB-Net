import concurrent.futures
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import colors

cudnn.benchmark = True

path = os.path.dirname(__file__)
patch_size = 128


def softmax_output_dice_class4(output, target):
    eps = 1e-8
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

    o4 = o3 * 0.0 if torch.sum(o3) < 500 else o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
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

    o5 = o4 * 0 if torch.sum(o4) < 500 else o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1, 2, 3)) + eps
    denominator5 = torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t5, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
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


def visualize_results(input_tensor, label_tensor, pred_tensor, case_name, mask_name, save_dir="visual_results_2d"):
    base_dir = os.path.join(save_dir, mask_name, case_name)
    os.makedirs(base_dir, exist_ok=True)

    input_np = input_tensor.cpu().float().numpy()
    label_np = label_tensor.cpu().numpy()
    pred_np = pred_tensor.cpu().numpy()

    cmap = colors.ListedColormap(["black", "red", "green", "yellow", "yellow"])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    slices = {
        "axial": (input_np.shape[2] // 2, 2),
        "sagittal": (input_np.shape[3] // 2, 1),
        "coronal": (input_np.shape[1] // 2, 0),
    }

    available_mods = [i for i in range(input_np.shape[0]) if input_np[i].max() > 0]
    available_mods = available_mods if available_mods else [0]

    for mod_idx in available_mods:
        mod_name = ["FLAIR", "T1", "T1ce", "T2"][mod_idx]
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

            fig_raw, ax_raw = plt.subplots(figsize=(6, 6), facecolor="black")
            ax_raw.imshow(img_slice, cmap="gray")
            ax_raw.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(
                os.path.join(base_dir, f"{mod_name}_{plane}_raw.png"),
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
            )
            plt.close(fig_raw)

            masked_label = np.ma.masked_where(label_slice == 0, label_slice)
            fig_label, ax_label = plt.subplots(figsize=(6, 6), facecolor="black")
            ax_label.imshow(np.zeros_like(img_slice), cmap="gray")
            ax_label.imshow(masked_label, cmap=cmap, norm=norm, alpha=1.0)
            ax_label.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(
                os.path.join(base_dir, f"{mod_name}_{plane}_groundtruth.png"),
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
            )
            plt.close(fig_label)

            masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            fig_pred, ax_pred = plt.subplots(figsize=(6, 6), facecolor="black")
            ax_pred.imshow(np.zeros_like(img_slice), cmap="gray")
            ax_pred.imshow(masked_pred, cmap=cmap, norm=norm, alpha=1.0)
            ax_pred.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(
                os.path.join(base_dir, f"{mod_name}_{plane}_prediction.png"),
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
            )
            plt.close(fig_pred)


def _extract_prediction_tensor(model_output):
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        pred = model_output[0]
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"Expected first model output to be a Tensor, got {type(pred)}.")
        return pred
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def test_softmax(test_loader, model, dataname="BRATS2018", feature_mask=None, mask_name=None):
    t_limit = 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size, dtype=torch.float32).cuda()

    if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
        num_cls = 4
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "ncr_net", "edema", "enhancing"
    elif dataname == "BRATS2015":
        num_cls = 5
        class_evaluation = "whole", "core", "enhancing", "enhancing_postpro"
        class_separate = "necrosis", "edema", "non_enhancing", "enhancing"
    else:
        raise ValueError(f"Unsupported dataname: {dataname}")

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
            mask = mask.cuda().bool()

            _, _, h_total, w_total, z_total = x.size()

            h_cnt = int(np.ceil((h_total - patch_size) / (patch_size * 0.5)))
            h_idx_list = [h_idx * int(patch_size * 0.5) for h_idx in range(h_cnt)]
            h_idx_list.append(h_total - patch_size)

            w_cnt = int(np.ceil((w_total - patch_size) / (patch_size * 0.5)))
            w_idx_list = [w_idx * int(patch_size * 0.5) for w_idx in range(w_cnt)]
            w_idx_list.append(w_total - patch_size)

            z_cnt = int(np.ceil((z_total - patch_size) / (patch_size * 0.5)))
            z_idx_list = [z_idx * int(patch_size * 0.5) for z_idx in range(z_cnt)]
            z_idx_list.append(z_total - patch_size)

            weight1 = torch.zeros(1, 1, h_total, w_total, z_total, dtype=torch.float32).cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h : h + patch_size, w : w + patch_size, z : z + patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            pred = torch.zeros(len(names), num_cls, h_total, w_total, z_total, dtype=torch.float32).cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h : h + patch_size, w : w + patch_size, z : z + patch_size]
                        model_output = model(x_input, mask)
                        pred_part = _extract_prediction_tensor(model_output)
                        if pred_part.shape != (len(names), num_cls, patch_size, patch_size, patch_size):
                            raise ValueError(
                                f"Prediction patch shape mismatch. Expected {(len(names), num_cls, patch_size, patch_size, patch_size)}, "
                                f"got {tuple(pred_part.shape)}."
                            )
                        pred[:, :, h : h + patch_size, w : w + patch_size, z : z + patch_size] += pred_part

            pred = pred / weight
            pred = pred[:, :, :h_total, :w_total, :t_limit]
            pred = torch.argmax(pred, dim=1)

            for k in range(len(names)):
                executor.submit(
                    visualize_results,
                    input_tensor=raw_input[k],
                    label_tensor=target[k].cpu(),
                    pred_tensor=pred[k].detach().cpu(),
                    case_name=names[k],
                    mask_name=mask_name or "full",
                    save_dir=os.path.join("./out/vis", "visual_results_2d"),
                )

            if dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
                scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            else:
                scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

            for k, name in enumerate(names):
                msg = f"Subject {i + 1}/{len(test_loader)}, {k + 1}/{len(names)}"
                msg += f"{name:>20}, "
                vals_separate.update(scores_separate[k])
                vals_evaluation.update(scores_evaluation[k])
                msg += ", ".join([f"{cls}: {score:.4f}" for cls, score in zip(class_evaluation, scores_evaluation[k])])
                logging.info(msg)

    msg = "Average scores:"
    msg += ", ".join([f"{cls}: {score:.4f}" for cls, score in zip(class_evaluation, vals_evaluation.avg)])
    print(msg)
    model.train()
    return vals_evaluation.avg


class AverageMeter(object):
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
