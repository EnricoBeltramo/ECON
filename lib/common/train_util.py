# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import yaml
import os.path as osp
import torch
import numpy as np
from ..dataset.mesh_util import *
from ..net.geometry import orthogonal
import cv2, PIL
from tqdm import tqdm
import os
from termcolor import colored
import pytorch_lightning as pl


def init_loss():

    losses = {
        # Cloth: Normal_recon - Normal_pred
        "cloth": {
            "weight": 1e1,
            "value": 0.0
        },
        # Cloth: [RT]_v1 - [RT]_v2 (v1-edge-v2)
        "stiffness": {
            "weight": 1e5,
            "value": 0.0
        },
        # Cloth: det(R) = 1
        "rigid": {
            "weight": 1e5,
            "value": 0.0
        },
        # Cloth: edge length
        "edge": {
            "weight": 0,
            "value": 0.0
        },
        # Cloth: normal consistency
        "nc": {
            "weight": 0,
            "value": 0.0
        },
        # Cloth: laplacian smoonth
        "laplacian": {
            "weight": 1e2,
            "value": 0.0
        },
        # Body: Normal_pred - Normal_smpl
        "normal": {
            "weight": 1e0,
            "value": 0.0
        },
        # Body: Silhouette_pred - Silhouette_smpl
        "silhouette": {
            "weight": 1e0,
            "value": 0.0
        },
        # Joint: reprojected joints difference
        "joint": {
            "weight": 5e0,
            "value": 0.0
        },
    }

    return losses


class SubTrainer(pl.Trainer):

    def save_checkpoint(self, filepath, weights_only=False):
        """Save model/training states as a checkpoint file through state-dump and file-write.
        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        _checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)

        del_keys = []
        for key in _checkpoint["state_dict"].keys():
            for ignore_key in ["normal_filter", "voxelization", "reconEngine"]:
                if ignore_key in key:
                    del_keys.append(key)
        for key in del_keys:
            del _checkpoint["state_dict"][key]

        pl.utilities.cloud_io.atomic_save(_checkpoint, filepath)


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def load_normal_networks(model, normal_path):
    
    pretrained_dict = torch.load(
            normal_path,
            map_location=model.device)["state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    # # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    del pretrained_dict
    del model_dict

    print(colored(f"Resume Normal weights from {normal_path}", "green"))


def load_networks(model, mlp_path, normal_path=None):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}
    
    # MLP part loading
    if os.path.exists(mlp_path) and mlp_path.endswith("ckpt"):
        main_dict = torch.load(
            mlp_path,
            map_location=model.device)["state_dict"]

        main_dict = {
            k: v
            for k, v in main_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and (
                "reconEngine" not in k) and ("normal_filter" not in k) and (
                    "voxelization" not in k)
        }
        print(colored(f"Resume MLP weights from {mlp_path}", "green"))

    # normal network part loading
    if normal_path is not None and os.path.exists(normal_path) and normal_path.endswith("ckpt"):
        normal_dict = torch.load(
            normal_path,
            map_location=model.device)["state_dict"]

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key,
                                 key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {normal_path}", "green"))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    # clean unused GPU memory
    del main_dict
    del normal_dict
    del model_dict
    torch.cuda.empty_cache()


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3],
    )
    return sample_tensor


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    """
    return:
        IOU, precision, and recall
    """
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data["img"].to(device=cuda)
            calib_tensor = data["calib"].to(device=cuda)
            sample_tensor = data["samples"].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor,
                                                      opt.num_views)
            label_tensor = data["labels"].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor,
                                     sample_tensor,
                                     calib_tensor,
                                     labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return (
        np.average(erorr_arr),
        np.average(IOU_arr),
        np.average(prec_arr),
        np.average(recall_arr),
    )


def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data["img"].to(device=cuda)
            calib_tensor = data["calib"].to(device=cuda)
            color_sample_tensor = data["color_samples"].to(
                device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(
                    color_sample_tensor, opt.num_views)

            rgb_tensor = data["rgbs"].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(
                image_tensor,
                netG.get_im_feat(),
                color_sample_tensor,
                calib_tensor,
                labels=rgb_tensor,
            )

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)


# pytorch lightning training related fucntions


def query_func(opt, netG, features, points, proj_matrix=None):
    """
        - points: size of (bz, N, 3)
        - proj_matrix: size of (bz, 4, 4)
    return: size of (bz, 1, N)
    """
    assert len(points) == 1
    samples = points.repeat(opt.num_views, 1, 1)
    samples = samples.permute(0, 2, 1)  # [bz, 3, N]

    # view specific query
    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = torch.stack([torch.eye(4).float()], dim=0).type_as(samples)

    preds = netG.query(
        features=features,
        points=samples,
        calibs=calib_tensor,
        regressor=netG.if_regressor,
    )

    if type(preds) is list:
        preds = preds[0]

    return preds

def query_func_IF(batch, netG, points):
    """
        - points: size of (bz, N, 3)
    return: size of (bz, 1, N)
    """
    
    batch["samples_geo"] = points
    batch["calib"] = torch.stack([torch.eye(4).float()], dim=0).type_as(points)
    
    preds = netG(batch)

    return preds.unsqueeze(1)


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def in1d(ar1, ar2):
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1, ), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]

def batch_mean(res, key):
    return torch.stack([
        x[key] if torch.is_tensor(x[key]) else torch.as_tensor(x[key])
        for x in res
    ]).mean()


def tf_log_convert(log_dict):
    new_log_dict = log_dict.copy()
    for k, v in log_dict.items():
        new_log_dict[k.replace("_", "/")] = v
        del new_log_dict[k]

    return new_log_dict


def bar_log_convert(log_dict, name=None, rot=None):
    from decimal import Decimal

    new_log_dict = {}

    if name is not None:
        new_log_dict["name"] = name[0]
    if rot is not None:
        new_log_dict["rot"] = rot[0]

    for k, v in log_dict.items():
        color = "yellow"
        if "loss" in k:
            color = "red"
            k = k.replace("loss", "L")
        elif "acc" in k:
            color = "green"
            k = k.replace("acc", "A")
        elif "iou" in k:
            color = "green"
            k = k.replace("iou", "I")
        elif "prec" in k:
            color = "green"
            k = k.replace("prec", "P")
        elif "recall" in k:
            color = "green"
            k = k.replace("recall", "R")

        if "lr" not in k:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{v:.3f}", color)
        else:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{Decimal(str(v)):.1E}",
                                                   color)

    if "loss" in new_log_dict.keys():
        del new_log_dict["loss"]

    return new_log_dict


def accumulate(outputs, rot_num, split):

    hparam_log_dict = {}

    metrics = outputs[0].keys()
    datasets = split.keys()

    for dataset in datasets:
        for metric in metrics:
            keyword = f"{dataset}/{metric}"
            if keyword not in hparam_log_dict.keys():
                hparam_log_dict[keyword] = 0
            for idx in range(split[dataset][0] * rot_num,
                             split[dataset][1] * rot_num):
                hparam_log_dict[keyword] += outputs[idx][metric].item()
            hparam_log_dict[keyword] /= (split[dataset][1] -
                                         split[dataset][0]) * rot_num

    print(colored(hparam_log_dict, "green"))

    return hparam_log_dict


def calc_error_N(outputs, targets):
    """calculate the error of normal (IGR)

    Args:
        outputs (torch.tensor): [B, 3, N]
        target (torch.tensor): [B, N, 3]

    # manifold loss and grad_loss in IGR paper
    grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()

    Returns:
        torch.tensor: error of valid normals on the surface
    """
    # outputs = torch.tanh(-outputs.permute(0,2,1).reshape(-1,3))
    outputs = -outputs.permute(0, 2, 1).reshape(-1, 1)
    targets = targets.reshape(-1, 3)[:, 2:3]
    with_normals = targets.sum(dim=1).abs() > 0.0

    # eikonal loss
    grad_loss = ((outputs[with_normals].norm(2, dim=-1) - 1)**2).mean()
    # normals loss
    normal_loss = (outputs - targets)[with_normals].abs().norm(2, dim=1).mean()

    return grad_loss * 0.0 + normal_loss


def calc_knn_acc(preds, carn_verts, labels, pick_num):
    """calculate knn accuracy

    Args:
        preds (torch.tensor): [B, 3, N]
        carn_verts (torch.tensor): [SMPLX_V_num, 3]
        labels (torch.tensor): [B, N_knn, N]
    """
    N_knn_full = labels.shape[1]
    preds = preds.permute(0, 2, 1).reshape(-1, 3)
    labels = labels.permute(0, 2, 1).reshape(-1, N_knn_full)  # [BxN, num_knn]
    labels = labels[:, :pick_num]

    dist = torch.cdist(preds, carn_verts, p=2)  # [BxN, SMPL_V_num]
    knn = dist.topk(k=pick_num, dim=1, largest=False)[1]  # [BxN, num_knn]
    cat_mat = torch.sort(torch.cat((knn, labels), dim=1))[0]
    bool_col = torch.zeros_like(cat_mat)[:, 0]
    for i in range(pick_num * 2 - 1):
        bool_col += cat_mat[:, i] == cat_mat[:, i + 1]
    acc = (bool_col > 0).sum() / len(bool_col)

    return acc


def calc_acc_seg(output, target, num_multiseg):
    from pytorch_lightning.metrics import Accuracy

    return Accuracy()(output.reshape(-1, num_multiseg).cpu(),
                      target.flatten().cpu())


def add_watermark(imgs, titles):

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (350, 50)
    bottomRightCornerOfText = (800, 50)
    fontScale = 1
    fontColor = (1.0, 1.0, 1.0)
    lineType = 2

    for i in range(len(imgs)):

        title = titles[i + 1]
        cv2.putText(imgs[i], title, bottomLeftCornerOfText, font, fontScale,
                    fontColor, lineType)

        if i == 0:
            cv2.putText(
                imgs[i],
                str(titles[i][0]),
                bottomRightCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )

    result = np.concatenate(imgs, axis=0).transpose(2, 0, 1)

    return result


def make_test_gif(img_dir):

    if img_dir is not None and len(os.listdir(img_dir)) > 0:
        for dataset in os.listdir(img_dir):
            for subject in sorted(os.listdir(osp.join(img_dir, dataset))):
                img_lst = []
                im1 = None
                for file in sorted(
                        os.listdir(osp.join(img_dir, dataset, subject))):
                    if file[-3:] not in ["obj", "gif"]:
                        img_path = os.path.join(img_dir, dataset, subject,
                                                file)
                        if im1 == None:
                            im1 = PIL.Image.open(img_path)
                        else:
                            img_lst.append(PIL.Image.open(img_path))

                print(os.path.join(img_dir, dataset, subject, "out.gif"))
                im1.save(
                    os.path.join(img_dir, dataset, subject, "out.gif"),
                    save_all=True,
                    append_images=img_lst,
                    duration=500,
                    loop=0,
                )


def export_cfg(logger, dir, cfg):

    cfg_export_file = osp.join(dir, f"cfg_{logger.version}.yaml")

    if not osp.exists(cfg_export_file):
        os.makedirs(osp.dirname(cfg_export_file), exist_ok=True)
        with open(cfg_export_file, "w+") as file:
            _ = yaml.dump(cfg, file)


from yacs.config import CfgNode

_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".
                format(".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict
