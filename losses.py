import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'TotalLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice



def mask_to_edge(mask):
    """
    mask: [B,1,H,W], float 0/1
    返回: edge: [B,1,H,W], 0/1 边界 map
    使用 max_pool2d 近似形态学膨胀/腐蚀
    """

    dilate = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    erode = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (dilate - erode).clamp(0, 1)
    return edge


def prototype_loss(feat, mask, proto_lesion, proto_bg, eps=1e-6):
    """
    feat:  [B, C, H, W]  解码特征
    mask:  [B, 1, H, W]  0/1 病斑标签
    proto_lesion: [C]
    proto_bg:     [C]
    返回: 标量 loss
    """
    B, C, H, W = feat.shape
    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
    mask_flat = mask.reshape(-1)                          # [B*H*W]

    lesion_mask = mask_flat > 0.5
    bg_mask     = ~lesion_mask


    p_les = proto_lesion / (proto_lesion.norm() + eps)
    p_bg  = proto_bg     / (proto_bg.norm() + eps)

    loss = 0.0
    count = 0

    if lesion_mask.any():
        f_les = feat_flat[lesion_mask]                    # [N_les, C]
        f_les_n = f_les / (f_les.norm(dim=1, keepdim=True) + eps)

        cos_les_ples = (f_les_n @ p_les)
        cos_les_pbg  = (f_les_n @ p_bg)

        loss_les_pull = (1.0 - cos_les_ples).mean()
        loss_les_push = F.relu(cos_les_pbg).mean()

        loss = loss + loss_les_pull + loss_les_push
        count += 1

    if bg_mask.any():
        f_bg = feat_flat[bg_mask]
        f_bg_n = f_bg / (f_bg.norm(dim=1, keepdim=True) + eps)

        cos_bg_pbg  = (f_bg_n @ p_bg)
        cos_bg_ples = (f_bg_n @ p_les)

        loss_bg_pull = (1.0 - cos_bg_pbg).mean()
        loss_bg_push = F.relu(cos_bg_ples).mean()

        loss = loss + loss_bg_pull + loss_bg_push
        count += 1

    if count > 0:
        loss = loss / count

    return loss


class TotalLoss(nn.Module):
    def __init__(self, lambda_edge=0.5, lambda_proto=0.05):
        super().__init__()
        self.seg_loss_fn = BCEDiceLoss()
        self.lambda_edge = lambda_edge
        self.lambda_proto = lambda_proto

    def forward(self, logits, edge_pred, feat, mask,
                proto_lesion, proto_bg):
        """
        logits:     [B,1,H,W] 主分割 logits
        edge_pred:  [B,1,H,W] 边界 logits
        feat:       [B,C,H,W] 用于原型约束的特征
        mask:       [B,1,H,W] 0/1
        proto_*:    [C]
        """

        loss_seg = self.seg_loss_fn(logits, mask)

        edge_gt = mask_to_edge(mask)                      # [B,1,H,W]
        loss_edge = F.binary_cross_entropy_with_logits(edge_pred, edge_gt)

        loss_proto = prototype_loss(feat, mask, proto_lesion, proto_bg)
        loss = loss_seg + self.lambda_edge * loss_edge + self.lambda_proto * loss_proto

        return loss, {
            'loss_seg': loss_seg.detach(),
            'loss_edge': loss_edge.detach(),
            'loss_proto': loss_proto.detach()
        }

