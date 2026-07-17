import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import cv2
import time
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import multi_archs
import multi_losses
from multi_dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold

import shutil

try:
    from thop import profile
except Exception:
    profile = None


ARCH_NAMES = ['BPNet']
LOSS_NAMES = multi_losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='ATfirst',
                        help='model name')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='mini-batch size')

    parser.add_argument('--dataseed', default=42, type=int,
                        help='random seed for data split and training')

    parser.add_argument('--arch', '-a', default='BPNet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES)

    parser.add_argument('--dataset', default='ATLDSD',
                        help='dataset name')
    parser.add_argument('--data_dir', default='/home/liyuanyuan/BPNet',
                        help='dataset root dir')
    parser.add_argument('--output_dir', default='outputs',
                        help='output dir')

    parser.add_argument('--img_ext', default='.jpg',
                        help='image extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask extension')

    parser.add_argument('--random_split', default=True, type=str2bool,
                        help='True means split from one images and masks folder')
    parser.add_argument('--n_splits', default=5, type=int,
                        help='number of folds')
    parser.add_argument('--fold', default=0, type=int,
                        help='fold index used as validation set, from 0 to n_splits minus 1')

    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)

    parser.add_argument('--scheduler', default='ConstantLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', default=False, action='store_true')
    parser.add_argument('--disease_classes', default=4, type=int,
                        help='number of lesion categories, excluding healthy class')
    parser.add_argument('--healthy_keywords', default='healthy,normal',
                        help='folder name keywords treated as healthy class')
    parser.add_argument('--mask_folder_candidates', default='masks,mask,label,labels',
                        help='candidate mask folder names under each class folder')

    config = parser.parse_args()
    return config


def save_feats_mean(x, size=(256, 256)):
    b, c, h, w = x.shape
    heatmaps = []
    with torch.no_grad():
        x = x.detach().cpu().numpy()
        for i in range(b):
            xi = np.transpose(x[i], (1, 2, 0))
            xi = np.mean(xi, axis=-1)

            maxv = np.max(xi)
            if maxv > 0:
                xi = xi / maxv
            else:
                xi = np.zeros_like(xi)

            xi = xi * 255.0
            xi = xi.astype(np.uint8)

            if (w, h) != size:
                xi = cv2.resize(xi, size)

            xi = cv2.applyColorMap(xi, cv2.COLORMAP_JET)
            xi = np.array(xi, dtype=np.uint8)
            heatmaps.append(xi)

    return heatmaps


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter()
    }

    if config['deep_supervision']:
        raise NotImplementedError("当前版本不支持 deep_supervision=True")

    model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, meta in train_loader:
        input = input.cuda()
        target = target.cuda().float()
        class_id = meta['class_id'].cuda().long()

        logits, edge_pred, feat = model(input)

        loss, loss_dict = criterion(
            logits,
            edge_pred,
            feat,
            target,
            class_id,
            model.prototype_lesion,
            model.prototype_bg
        )

        iou, dice, _ = iou_score(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg)
    ])

def get_binary_confusion(logits, target, threshold=0.5):
    """
    计算一个 batch 的二分类像素级 TP、FP、FN、TN。
    logits: [B, 1, H, W]，模型原始输出
    target: [B, 1, H, W]，取值为 0/1
    """
    pred = torch.sigmoid(logits) > threshold
    gt = target > 0.5

    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, torch.logical_not(gt)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), gt).sum().item()
    tn = torch.logical_and(torch.logical_not(pred), torch.logical_not(gt)).sum().item()

    return tp, fp, fn, tn


def calculate_binary_metrics(tp, fp, fn, tn=0):
    """
    用整个验证集累计的 TP、FP、FN、TN 统一计算指标。
    这里的 Dice 与 F1 严格使用同一口径。
    """
    eps = 1e-8

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)

    f1 = dice

    return iou, dice, precision, recall, specificity, f1

def validate(config, val_loader, model, criterion):
    avg_meters = {
        'loss': AverageMeter(),
    }

    if config['deep_supervision']:
        raise NotImplementedError("当前版本不支持 deep_supervision=True")

    model.eval()

    heatmap_dir = os.path.join(config['output_dir'], config['name'], 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    # 在整个验证集范围内累计混淆矩阵
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))

        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda().float()
            class_id = meta['class_id'].cuda().long()

            logits, edge_pred, feat = model(input)

            loss, _ = criterion(
                logits,
                edge_pred,
                feat,
                target,
                class_id,
                model.prototype_lesion,
                model.prototype_bg
            )

            heatmaps = save_feats_mean(
                feat,
                size=(config['input_w'], config['input_h'])
            )

            for hmap, img_id in zip(heatmaps, meta['img_id']):
                cv2.imwrite(os.path.join(heatmap_dir, f'{img_id}.jpg'), hmap)

            # 当前 batch 的 TP / FP / FN / TN，累计到全验证集
            tp, fp, fn, tn = get_binary_confusion(
                logits=logits,
                target=target,
                threshold=0.5
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            avg_meters['loss'].update(loss.item(), input.size(0))

            cur_iou, cur_dice, cur_precision, cur_recall, _, cur_f1 = \
                calculate_binary_metrics(
                    total_tp, total_fp, total_fn, total_tn
                )

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', cur_iou),
                ('dice', cur_dice),
                ('prec', cur_precision),
                ('rec', cur_recall),
                ('f1', cur_f1),
            ])

            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()


    iou, dice, precision, recall, specificity, f1 = calculate_binary_metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    print(
        f'[Validation counts] '
        f'TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}'
    )

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', iou),
        ('dice', dice),
        ('precision', precision),
        ('recall', recall),
        ('specificity', specificity),
        ('f1', f1),
    ])


def save_pr_roc_by_threshold_csv(config, val_loader, model, pr_csv_path, roc_csv_path,
                                 thresholds=None, thr_chunk=16):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    thresholds = np.asarray(thresholds, dtype=np.float32)
    T = len(thresholds)

    tp = torch.zeros(T, dtype=torch.long)
    fp = torch.zeros(T, dtype=torch.long)
    tn = torch.zeros(T, dtype=torch.long)
    fn = torch.zeros(T, dtype=torch.long)

    model.eval()
    thr_tensor_full = torch.tensor(thresholds, device='cuda')

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda().float()

            out = model(input)

            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out

            if logits.dim() == 4 and logits.size(1) == 1:
                prob = torch.sigmoid(logits[:, 0])
            elif logits.dim() == 3:
                prob = torch.sigmoid(logits)
            else:
                raise ValueError(f"Unexpected logits shape for binary: {tuple(logits.shape)}")

            if target.dim() == 4 and target.size(1) == 1:
                gt = target[:, 0] > 0.5
            else:
                gt = target > 0.5

            prob_flat = prob.reshape(-1)
            gt_flat = gt.reshape(-1).bool()

            for s in range(0, T, thr_chunk):
                e = min(s + thr_chunk, T)
                thr = thr_tensor_full[s:e].view(1, -1)

                pred = prob_flat.view(-1, 1) >= thr
                yb = gt_flat.view(-1, 1)

                tp[s:e] += (pred & yb).sum(dim=0).to('cpu', dtype=torch.long)
                fp[s:e] += (pred & (~yb)).sum(dim=0).to('cpu', dtype=torch.long)
                tn[s:e] += ((~pred) & (~yb)).sum(dim=0).to('cpu', dtype=torch.long)
                fn[s:e] += ((~pred) & yb).sum(dim=0).to('cpu', dtype=torch.long)

            pbar.update(1)

        pbar.close()

    tp_np = tp.numpy().astype(np.float64)
    fp_np = fp.numpy().astype(np.float64)
    tn_np = tn.numpy().astype(np.float64)
    fn_np = fn.numpy().astype(np.float64)

    precision = np.divide(tp_np, tp_np + fp_np, out=np.full_like(tp_np, np.nan), where=(tp_np + fp_np) > 0)
    recall = np.divide(tp_np, tp_np + fn_np, out=np.full_like(tp_np, np.nan), where=(tp_np + fn_np) > 0)
    tpr = recall.copy()
    fpr = np.divide(fp_np, fp_np + tn_np, out=np.full_like(tp_np, np.nan), where=(fp_np + tn_np) > 0)

    pr_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision,
        'recall': recall,
        'tp': tp_np,
        'fp': fp_np,
        'tn': tn_np,
        'fn': fn_np
    })

    roc_df = pd.DataFrame({
        'threshold': thresholds,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp_np,
        'fp': fp_np,
        'tn': tn_np,
        'fn': fn_np
    })

    pr_df.to_csv(pr_csv_path, index=False)
    roc_df.to_csv(roc_csv_path, index=False)


def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return total_params, trainable_params, non_trainable_params


def calculate_flops_params(model, input_size):
    if profile is None:
        print("Warning: thop is not installed. FLOPs cannot be calculated.")
        return None, None

    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"Error calculating FLOPs/Params: {e}")
        flops, params = None, None

    if was_training:
        model.train()

    return flops, params


def measure_inference_speed(model, input_size, warmup=50, repeat=200):
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    was_training = model.training
    model.eval()

    if device.type == 'cuda':
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(repeat):
                _ = model(dummy_input)
            end_event.record()

            torch.cuda.synchronize()

            total_time_ms = start_event.elapsed_time(end_event)
            latency_ms = total_time_ms / repeat
            fps = 1000.0 / latency_ms

    else:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

            start_time = time.time()
            for _ in range(repeat):
                _ = model(dummy_input)
            end_time = time.time()

            total_time_s = end_time - start_time
            latency_ms = total_time_s * 1000.0 / repeat
            fps = 1000.0 / latency_ms

    if was_training:
        model.train()

    return latency_ms, fps


def print_model_info(model, config, input_size=None):
    if input_size is None:
        input_size = (
            1,
            config['input_channels'],
            config['input_h'],
            config['input_w']
        )

    print("\n" + "=" * 60)
    print("Model Complexity and Inference Speed")
    print("=" * 60)
    print(f"Input size: {input_size}")

    total_params, trainable_params, non_trainable_params = count_model_params(model)

    print(f"Total Params      : {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"Trainable Params  : {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    print(f"Non-trainable     : {non_trainable_params:,}")

    flops, thop_params = calculate_flops_params(model, input_size)

    if flops is not None:
        print(f"FLOPs             : {flops:,}")
        print(f"GFLOPs            : {flops / 1e9:.3f} G")
    else:
        print("FLOPs             : None")

    latency_ms, fps = measure_inference_speed(
        model,
        input_size=input_size,
        warmup=50,
        repeat=200
    )

    print(f"Inference Time    : {latency_ms:.3f} ms / image")
    print(f"Inference Speed   : {fps:.2f} FPS")
    print("=" * 60 + "\n")

    info = {
        'input_h': config['input_h'],
        'input_w': config['input_w'],
        'input_channels': config['input_channels'],
        'total_params': total_params,
        'total_params_M': total_params / 1e6,
        'trainable_params': trainable_params,
        'trainable_params_M': trainable_params / 1e6,
        'non_trainable_params': non_trainable_params,
        'flops': flops,
        'gflops': flops / 1e9 if flops is not None else None,
        'thop_params': thop_params,
        'latency_ms': latency_ms,
        'fps': fps
    }

    return info


def save_model_info_to_file(info_dict, txt_path, csv_path=None):
    with open(txt_path, 'w') as f:
        f.write("Model Complexity and Inference Speed\n")
        f.write("=" * 60 + "\n")
        for key, value in info_dict.items():
            if value is None:
                f.write(f"{key}: None\n")
            elif isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("=" * 60 + "\n")

    if csv_path is not None:
        pd.DataFrame([info_dict]).to_csv(csv_path, index=False)


def seed_torch(seed=1029):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed

def _is_healthy_folder(folder_name, healthy_keywords):
    name = folder_name.lower()
    keywords = [k.strip().lower() for k in healthy_keywords.split(',') if k.strip()]
    return any(k in name for k in keywords)


def _find_mask_dir(class_dir, candidates):
    for name in candidates.split(','):
        name = name.strip()
        if not name:
            continue

        p = os.path.join(class_dir, name)

        if os.path.isdir(p):
            return p

    return None


def build_split(config):
    dataset_root = os.path.join(config['data_dir'], config['dataset'])
    img_ext = config['img_ext']
    mask_ext = config['mask_ext']

    if not os.path.isdir(dataset_root):
        raise RuntimeError(f"Dataset root does not exist, {dataset_root}")

    class_names = []
    for name in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, name)
        image_dir = os.path.join(class_dir, 'images')

        if os.path.isdir(class_dir) and os.path.isdir(image_dir):
            class_names.append(name)

    if len(class_names) == 0:
        raise RuntimeError(
            f"No class folders found under {dataset_root}. "
            f"Expected structure, dataset/class_name/images and dataset/class_name/mask."
        )

    disease_names = [
        name for name in class_names
        if not _is_healthy_folder(name, config['healthy_keywords'])
    ]

    disease_to_id = {name: i for i, name in enumerate(disease_names)}

    if len(disease_names) != int(config['disease_classes']):
        raise RuntimeError(
            f"disease_classes mismatch, config says {config['disease_classes']}, "
            f"but detected {len(disease_names)}, disease folders are {disease_names}"
        )

    all_samples = []
    y = []

    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        image_dir = os.path.join(class_dir, 'images')
        mask_dir = _find_mask_dir(class_dir, config['mask_folder_candidates'])

        is_healthy = _is_healthy_folder(class_name, config['healthy_keywords'])
        class_id = -1 if is_healthy else disease_to_id[class_name]

        if mask_dir is None and not is_healthy:
            raise RuntimeError(f"No mask folder found for disease class, {class_name}")

        img_paths = sorted(glob(os.path.join(image_dir, '*' + img_ext)))

        if len(img_paths) == 0:
            print(f"Warning, no images found in {image_dir} with extension {img_ext}")
            continue

        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]

            mask_path = ''
            if mask_dir is not None:
                candidate = os.path.join(mask_dir, stem + mask_ext)
                if os.path.exists(candidate):
                    mask_path = candidate

            if class_id >= 0 and not mask_path:
                raise RuntimeError(
                    f"Mask not found for disease image, image={img_path}, "
                    f"expected mask={os.path.join(mask_dir, stem + mask_ext)}"
                )

            safe_img_id = f"{class_name.replace(' ', '_')}__{stem}"

            all_samples.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'img_id': safe_img_id,
                'class_id': class_id,
                'class_name': class_name
            })

            y.append(class_id)

    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found in {dataset_root}")

    n_splits = int(config['n_splits'])
    fold = int(config['fold'])

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, but got {n_splits}")

    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], but got {fold}")

    y = np.asarray(y)

    unique_labels, label_counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_labels) > 1 and label_counts.min() >= n_splits

    if can_stratify:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(config['dataseed'])
        )
        splits = list(splitter.split(np.arange(len(all_samples)), y))
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(config['dataseed'])
        )
        splits = list(splitter.split(np.arange(len(all_samples))))

    train_index, val_index = splits[fold]

    train_samples = [all_samples[i] for i in train_index]
    val_samples = [all_samples[i] for i in val_index]

    print(f"Dataset root: {dataset_root}")
    print(f"Class folders: {class_names}")
    print(f"Disease folders: {disease_names}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Current fold: {fold + 1}/{n_splits}")

    return train_samples, val_samples, disease_names


def save_split_file(config, train_samples, val_samples):
    split_path = os.path.join(config['output_dir'], config['name'], 'split.csv')

    rows = []

    for s in train_samples:
        rows.append({
            'img_id': s['img_id'],
            'image_path': s['image_path'],
            'mask_path': s['mask_path'],
            'class_id': s['class_id'],
            'class_name': s['class_name'],
            'split': 'train'
        })

    for s in val_samples:
        rows.append({
            'img_id': s['img_id'],
            'image_path': s['image_path'],
            'mask_path': s['mask_path'],
            'class_id': s['class_id'],
            'class_name': s['class_name'],
            'split': 'val'
        })

    pd.DataFrame(rows).to_csv(split_path, index=False)

    print(f"=> saved split file: {split_path}")


def main():
    config = vars(parse_args())

    seed_torch(config['dataseed'])

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    exp_name = config['name']
    output_dir = config['output_dir']

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    criterion = multi_losses.TotalLoss(lambda_edge=0.5, lambda_proto=0.05).cuda()

    cudnn.benchmark = False

    model = multi_archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        disease_classes=config['disease_classes']
    )

    model = model.cuda()

    input_size = (
        1,
        config['input_channels'],
        config['input_h'],
        config['input_w']
    )

    model_info = print_model_info(
        model=model,
        config=config,
        input_size=input_size
    )

    save_model_info_to_file(
        model_info,
        txt_path=f'{output_dir}/{exp_name}/model_complexity.txt',
        csv_path=f'{output_dir}/{exp_name}/model_complexity.csv'
    )

    my_writer.add_scalar('model/total_params_M', model_info['total_params_M'], 0)
    my_writer.add_scalar('model/trainable_params_M', model_info['trainable_params_M'], 0)

    if model_info['gflops'] is not None:
        my_writer.add_scalar('model/gflops', model_info['gflops'], 0)

    my_writer.add_scalar('model/latency_ms', model_info['latency_ms'], 0)
    my_writer.add_scalar('model/fps', model_info['fps'], 0)

    model.train()

    param_groups = []

    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            param_groups.append({
                'params': param,
                'lr': config['kan_lr'],
                'weight_decay': config['kan_weight_decay']
            })
        else:
            param_groups.append({
                'params': param,
                'lr': config['lr'],
                'weight_decay': config['weight_decay']
            })

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)

    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=config['lr'],
            momentum=config['momentum'],
            nesterov=config['nesterov'],
            weight_decay=config['weight_decay']
        )

    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )

    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['factor'],
            patience=config['patience'],
            verbose=1,
            min_lr=config['min_lr']
        )

    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['gamma']
        )

    elif config['scheduler'] == 'ConstantLR':
        scheduler = None

    else:
        raise NotImplementedError

    try:
        shutil.copy2('multi_train.py', f'{output_dir}/{exp_name}/')
    except Exception as e:
        print(f"=> failed to copy train.py: {e}")

    try:
        shutil.copy2('multi_archs.py', f'{output_dir}/{exp_name}/')
    except Exception as e:
        print(f"=> failed to copy archs.py: {e}")

    img_ext = config['img_ext']
    mask_ext = config['mask_ext']

    train_samples, val_samples, disease_names = build_split(config)

    save_split_file(config, train_samples, val_samples)

    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_samples,
        img_dir=None,
        mask_dir=None,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform
    )

    val_dataset = Dataset(
        img_ids=val_samples,
        img_dir=None,
        mask_dir=None,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_precision', []),
        ('val_recall', []),
        ('val_f1', []),
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        elif config['scheduler'] == 'MultiStepLR':
            scheduler.step()


        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - '
              'val_dice %.4f - val_prec %.4f - val_rec %.4f - val_f1 %.4f'
              % (train_log['loss'], train_log['iou'],
                 val_log['loss'], val_log['iou'],
                 val_log['dice'],
                 val_log['precision'], val_log['recall'], val_log['f1']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_precision'].append(val_log['precision'])
        log['val_recall'].append(val_log['recall'])
        log['val_f1'].append(val_log['f1'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/precision', val_log['precision'], global_step=epoch)
        my_writer.add_scalar('val/recall', val_log['recall'], global_step=epoch)
        my_writer.add_scalar('val/f1', val_log['f1'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']

            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            print('Precision: %.4f' % val_log['precision'])
            print('Recall: %.4f' % val_log['recall'])
            print('F1: %.4f' % val_log['f1'])

            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    try:
        best_model_path = f'{output_dir}/{exp_name}/model.pth'

        if os.path.exists(best_model_path):
            state = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(state)
            model = model.cuda()

        pr_csv_path = f'{output_dir}/{exp_name}/p_r_by_threshold.csv'
        roc_csv_path = f'{output_dir}/{exp_name}/tpr_fpr_by_threshold.csv'

        save_pr_roc_by_threshold_csv(
            config=config,
            val_loader=val_loader,
            model=model,
            pr_csv_path=pr_csv_path,
            roc_csv_path=roc_csv_path,
            thresholds=np.linspace(0.0, 1.0, 101, dtype=np.float32),
            thr_chunk=16
        )

        print(f"=> saved threshold PR CSV  : {pr_csv_path}")
        print(f"=> saved threshold ROC CSV : {roc_csv_path}")

    except Exception as e:
        print(f"=> failed to save threshold PR/ROC CSVs: {e}")

    my_writer.close()


if __name__ == '__main__':
    main()