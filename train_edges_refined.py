# coding=utf-8
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from data.data_utils import init_fn
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.transforms import *
from model_swint_mamba_edges_refined import Model
from predict_vis_refined import AverageMeter, test_softmax
from utils import Parser, criterions as criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", "--batch_size", default=1, type=int, help="Batch size")
parser.add_argument(
    "--datapath",
    default=None,
    type=str,
    help="Path to the training data directory",
)
parser.add_argument("--dataname", default="BRATS2018", type=str)
parser.add_argument(
    "--savepath",
    default="./out/swint-mamba-refined",
    type=str,
    help="Directory to save checkpoints",
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="Checkpoint path to resume from",
)
parser.add_argument("--pretrain", default=None, type=str)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--num_epochs", default=0, type=int)
parser.add_argument("--iter_per_epoch", default=150, type=int)
parser.add_argument("--region_fusion_start_epoch", default=0, type=int)
parser.add_argument("--seed", default=1024, type=int)
parser.add_argument("--res", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str, help="Optional CUDA_VISIBLE_DEVICES override")

args = parser.parse_args()
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print(args)
setup(args, "training")
args.train_transforms = (
    "Compose([RandCrop3D((128,128,128)), RandomRotion(10), "
    "RandomIntensityChange((0.1,0.1)), RandomFlip(0), "
    "NumpyType((np.float32, np.int64)),])"
)
args.test_transforms = "Compose([NumpyType((np.float32, np.int64)),])"

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

train_log_dir = "summary/train/"
writer = SummaryWriter(os.path.join(args.savepath, train_log_dir))

masks = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],
    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],
    [True, True, True, True],
]

masks_torch = torch.from_numpy(np.array(masks))
mask_name = [
    "t2",
    "t1ce",
    "t1",
    "flair",
    "t1cet2",
    "t1cet1",
    "flairt1",
    "t1t2",
    "flairt2",
    "flairt1ce",
    "flairt1cet1",
    "flairt1t2",
    "flairt1cet2",
    "t1cet1t2",
    "flairt1cet1t2",
]
print(masks_torch.int())


def get_boundary_3d(mask, kernel_size=3):
    if mask.sum() == 0:
        return mask
    padding = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool3d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated - eroded


def generate_multilabel_boundaries(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)

    wt_mask = (mask > 0).float()
    tc_mask = ((mask == 2) | (mask == 3)).float()
    et_mask = (mask == 3).float()

    wt_edge = get_boundary_3d(wt_mask)
    tc_edge = get_boundary_3d(tc_mask)
    et_edge = get_boundary_3d(et_mask)
    return torch.cat([et_edge, tc_edge, wt_edge], dim=1)


def assert_batch_shapes(x, target, mask, num_cls):
    if x.dim() != 5 or x.shape[1] != 4:
        raise ValueError(f"Expected x to be [B,4,D,H,W], got {tuple(x.shape)}.")
    if mask.dim() != 2 or mask.shape[0] != x.shape[0] or mask.shape[1] != 4:
        raise ValueError(f"Expected mask to be [B,4], got {tuple(mask.shape)}.")
    if target.dim() != 5:
        raise ValueError(f"Expected target to be 5D, got {tuple(target.shape)}.")
    if target.shape[0] != x.shape[0]:
        raise ValueError("Batch size mismatch between x and target.")
    if target.shape[1] != num_cls:
        raise ValueError(f"Expected target channel count {num_cls}, got {target.shape[1]}.")


def build_edge_targets(target):
    edge_targets = []
    scales = [1, 2, 4, 8, 16]
    labels_idx_map = torch.argmax(target, dim=1).float()

    for scale in scales:
        if scale == 1:
            curr_label_idx = labels_idx_map
        else:
            curr_label_idx = F.interpolate(
                labels_idx_map.unsqueeze(1),
                scale_factor=1 / scale,
                mode="nearest",
            ).squeeze(1)

        with torch.no_grad():
            curr_edge_gt = generate_multilabel_boundaries(curr_label_idx).to(target.device)
        edge_targets.append(curr_edge_gt)

    return edge_targets


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.num_epochs < 0:
        raise ValueError(f"num_epochs must be > 0, got {args.num_epochs}.")

    if args.dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
        num_cls = 4
    elif args.dataname == "BRATS2015":
        num_cls = 5
    else:
        raise ValueError(f"Unsupported dataset: {args.dataname}")

    model = Model(num_cls=num_cls)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=True,
    )

    if args.dataname in ["BRATS2020", "BRATS2015", "BRATS2018"]:
        train_file = "train.txt"
        test_file = "test.txt"
    else:
        raise ValueError(f"Unsupported dataset split config for {args.dataname}")

    logging.info(str(args))
    train_set = Brats_loadall_nii(
        transforms=args.train_transforms,
        root=args.datapath,
        num_cls=num_cls,
        train_file=train_file,
    )
    test_set = Brats_loadall_test_nii(
        transforms=args.test_transforms,
        root=args.datapath,
        test_file=test_file,
    )
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
    )
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    start_epoch = -1
    if args.resume is not None and os.path.isfile(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location="cpu")

        start_epoch = int(checkpoint["epoch"])
        logging.info("Resume from epoch: {}".format(checkpoint["epoch"]))
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optim_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_dict"])
    elif args.resume:
        logging.warning("Resume checkpoint not found: %s", args.resume)

    start = time.time()
    logging.info("#############training############")
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    test_average_filename = os.path.join(args.savepath, "test_average.txt")

    for epoch in range(start_epoch + 1, args.num_epochs):
        model.train()
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar("lr", step_lr, global_step=(epoch + 1))
        epoch_start = time.time()

        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True).bool()

            assert_batch_shapes(x, target, mask, num_cls)

            fuse_logits, prm_logits, edge_preds = model(
                x,
                mask,
                return_edges=True,
                apply_softmax=False,
            )

            if fuse_logits.shape[1] != num_cls:
                raise ValueError(f"fuse_logits channel mismatch: {tuple(fuse_logits.shape)}")
            if len(prm_logits) != 4:
                raise ValueError(f"Expected 4 deep supervision outputs, got {len(prm_logits)}")
            if len(edge_preds) != 5:
                raise ValueError(f"Expected 5 edge predictions, got {len(edge_preds)}")
            for edge_pred in edge_preds:
                if edge_pred.dim() != 5 or edge_pred.shape[1] != 3:
                    raise ValueError(f"Edge prediction shape invalid: {tuple(edge_pred.shape)}")

            fuse_prob = torch.softmax(fuse_logits, dim=1)
            prm_probs = tuple(torch.softmax(pred, dim=1) for pred in prm_logits)

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_prob, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_prob, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1, device=x.device).float()
            prm_dice_loss = torch.zeros(1, device=x.device).float()
            for prm_pred in prm_probs:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            edge_targets = build_edge_targets(target)
            edge_loss = criterions.edges_loss(edge_preds, edge_targets)

            total_loss = fuse_loss + prm_loss + edge_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar("total_loss", total_loss.item(), global_step=step)
            writer.add_scalar("fuse_loss", fuse_loss.item(), global_step=step)
            writer.add_scalar("prm_loss", prm_loss.item(), global_step=step)
            writer.add_scalar("edge_loss", edge_loss.item(), global_step=step)

            msg = (
                "Epoch {}/{}, Iter {}/{}, Total Loss {:.4f}, fuse_loss:{:.4f}, "
                "prm_loss:{:.4f}, edge_loss:{:.4f}"
            ).format(
                epoch + 1,
                args.num_epochs,
                i + 1,
                iter_per_epoch,
                total_loss.item(),
                fuse_loss.item(),
                prm_loss.item(),
                edge_loss.item(),
            )
            logging.info(msg)

        logging.info("train time per epoch: {} s".format(time.time() - epoch_start))

        state_dict = model.state_dict()
        file_name = os.path.join(ckpts, "model_last.pth")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": state_dict,
                "optim_dict": optimizer.state_dict(),
            },
            file_name,
        )

        if (epoch + 1) % 50 == 0 or epoch >= (args.num_epochs - 10):
            file_name = os.path.join(ckpts, "model_{}.pth".format(epoch + 1))
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                file_name,
            )

        if (epoch + 1) % 100 == 0 and (epoch + 1) != args.num_epochs:
            model.eval()
            test_score = AverageMeter()
            with torch.no_grad():
                logging.info("###########test set wi/wo postprocess###########")
                with open(test_average_filename, mode="a+", encoding="utf-8") as f:
                    f.write("epoch:{}\n\n".format(epoch + 1))
                for idx, feature_mask in enumerate(masks):
                    logging.info("{}".format(mask_name[idx]))
                    dice_score = test_softmax(
                        test_loader,
                        model,
                        dataname=args.dataname,
                        feature_mask=feature_mask,
                    )
                    test_score.update(dice_score)
                    with open(test_average_filename, mode="a+", encoding="utf-8") as f:
                        f.write("{},  {}\n".format(mask_name[idx], dice_score))
                logging.info("Avg scores: {}".format(test_score.avg))
                with open(test_average_filename, mode="a+", encoding="utf-8") as f:
                    f.write("{}\n\n".format(test_score.avg))

    logging.info("total time: {:.4f} hours".format((time.time() - start) / 3600))

    model.eval()
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info("###########test set wi/wo postprocess###########")
        with open(test_average_filename, mode="a+", encoding="utf-8") as f:
            f.write("epoch:{}\n\n".format(args.num_epochs))
        for idx, feature_mask in enumerate(masks):
            logging.info("{}".format(mask_name[idx]))
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=feature_mask,
            )
            test_score.update(dice_score)
            with open(test_average_filename, mode="a+", encoding="utf-8") as f:
                f.write("{},  {}\n".format(mask_name[idx], dice_score))
        logging.info("Avg scores: {}".format(test_score.avg))
        with open(test_average_filename, mode="a+", encoding="utf-8") as f:
            f.write("{}\n\n".format(test_score.avg))


if __name__ == "__main__":
    main()
