import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def iou_score(output, target):
    smooth = 1e-5

    # output 是 logits，这里做 sigmoid
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    # hd95 仅作参考，已做异常保护
    try:
        hd95_ = hd95(output_, target_)
    except Exception:
        hd95_ = 0.0

    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def indicators(output, target):
    """
    output: logits, tensor [B,1,H,W] 或 [B,H,W]
    target: tensor [B,1,H,W] 或 [B,H,W]，0/1
    返回: iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
    """
    # 转 numpy + sigmoid
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    # Jaccard / Dice / Recall / Specificity / Precision 用 medpy 即可
    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    # ==== 关键：hd / hd95 做前景检查 + try/except ====
    if np.any(output_) and np.any(target_):
        try:
            hd_ = hd(output_, target_)
        except Exception:
            hd_ = 0.0

        try:
            hd95_ = hd95(output_, target_)
        except Exception:
            hd95_ = 0.0
    else:
        # 任意一方没有前景，对 hd 无意义，直接给 0 或者 np.nan
        hd_, hd95_ = 0.0, 0.0

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_





# import numpy as np
# import torch
# import torch.nn.functional as F
#
# from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
#
#
#
# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2* iou) / (iou+1)
#
#     try:
#         hd95_ = hd95(output_, target_)
#     except:
#         hd95_ = 0
#
#     return iou, dice, hd95_
#
#
# def dice_coef(output, target):
#     smooth = 1e-5
#
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#
#     return (2. * intersection + smooth) / \
#         (output.sum() + target.sum() + smooth)
#
# def indicators(output, target):
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#
#     iou_ = jc(output_, target_)
#     dice_ = dc(output_, target_)
#     hd_ = hd(output_, target_)
#     hd95_ = hd95(output_, target_)
#     recall_ = recall(output_, target_)
#     specificity_ = specificity(output_, target_)
#     precision_ = precision(output_, target_)
#
#     return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
