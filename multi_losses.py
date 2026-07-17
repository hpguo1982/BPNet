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
    # 膨胀
    dilate = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    # 腐蚀：对 -mask 做 max_pool 再取负
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

    # 归一化原型
    p_les = proto_lesion / (proto_lesion.norm() + eps)
    p_bg  = proto_bg     / (proto_bg.norm() + eps)

    loss = 0.0
    count = 0

    if lesion_mask.any():
        f_les = feat_flat[lesion_mask]                    # [N_les, C]
        f_les_n = f_les / (f_les.norm(dim=1, keepdim=True) + eps)

        cos_les_ples = (f_les_n @ p_les)                  # 越接近 1 越好
        cos_les_pbg  = (f_les_n @ p_bg)                   # 希望越小越好

        loss_les_pull = (1.0 - cos_les_ples).mean()       # 拉近病斑→p_les
        loss_les_push = F.relu(cos_les_pbg).mean()        # 推远病斑→p_bg

        loss = loss + loss_les_pull + loss_les_push
        count += 1

    if bg_mask.any():
        f_bg = feat_flat[bg_mask]
        f_bg_n = f_bg / (f_bg.norm(dim=1, keepdim=True) + eps)

        cos_bg_pbg  = (f_bg_n @ p_bg)
        cos_bg_ples = (f_bg_n @ p_les)

        loss_bg_pull = (1.0 - cos_bg_pbg).mean()          # 拉近背景→p_bg
        loss_bg_push = F.relu(cos_bg_ples).mean()         # 推远背景→p_les

        loss = loss + loss_bg_pull + loss_bg_push
        count += 1

    if count > 0:
        loss = loss / count

    return loss


def multi_prototype_loss(feat, mask, class_id, proto_lesion, proto_bg, eps=1e-6):
    """
    feat:          [B,C,H,W]
    mask:          [B,1,H,W], binary mask, 0 background, 1 lesion
    class_id:      [B], disease class id, healthy image uses -1
    proto_lesion:  [D,C], D disease prototypes
    proto_bg:      [C], background prototype
    """
    B, C, H, W = feat.shape
    D = proto_lesion.shape[0]

    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.float(), size=(H, W), mode='nearest')

    class_id = class_id.view(-1).long().to(feat.device)

    proto_lesion_n = F.normalize(proto_lesion, dim=1, eps=eps)
    proto_bg_n = F.normalize(proto_bg, dim=0, eps=eps)

    losses = []

    for b in range(B):
        f = feat[b].permute(1, 2, 0).reshape(-1, C)
        f = F.normalize(f, dim=1, eps=eps)

        m = mask[b, 0].reshape(-1) > 0.5
        bg = ~m

        cid = int(class_id[b].item())

        if cid >= 0 and cid < D and m.any():
            f_les = f[m]

            sim_all_lesion = f_les @ proto_lesion_n.t()
            sim_bg = f_les @ proto_bg_n

            pull_target = 1.0 - sim_all_lesion[:, cid]

            other_mask = torch.ones(D, dtype=torch.bool, device=feat.device)
            other_mask[cid] = False

            if other_mask.any():
                push_other = F.relu(sim_all_lesion[:, other_mask]).mean()
            else:
                push_other = torch.tensor(0.0, device=feat.device)

            push_bg = F.relu(sim_bg).mean()

            losses.append(pull_target.mean() + push_other + push_bg)

        if bg.any():
            f_bg = f[bg]

            sim_bg = f_bg @ proto_bg_n
            sim_lesion = f_bg @ proto_lesion_n.t()

            pull_bg = 1.0 - sim_bg
            push_lesion = F.relu(sim_lesion).mean()

            losses.append(pull_bg.mean() + push_lesion)

    if len(losses) == 0:
        return feat.sum() * 0.0

    return torch.stack(losses).mean()


class TotalLoss(nn.Module):
    def __init__(self, lambda_edge=0.5, lambda_proto=0.05):
        super().__init__()
        self.seg_loss_fn = BCEDiceLoss()
        self.lambda_edge = lambda_edge
        self.lambda_proto = lambda_proto

    def forward(self, logits, edge_pred, feat, mask, class_id,
                proto_lesion, proto_bg):
        """
        logits:        [B,1,H,W]
        edge_pred:     [B,1,H,W]
        feat:          [B,C,H,W]
        mask:          [B,1,H,W]
        class_id:      [B]
        proto_lesion:  [D,C]
        proto_bg:      [C]
        """
        loss_seg = self.seg_loss_fn(logits, mask)

        edge_gt = mask_to_edge(mask)
        loss_edge = F.binary_cross_entropy_with_logits(edge_pred, edge_gt)

        loss_proto = multi_prototype_loss(
            feat=feat,
            mask=mask,
            class_id=class_id,
            proto_lesion=proto_lesion,
            proto_bg=proto_bg
        )

        loss = loss_seg + self.lambda_edge * loss_edge + self.lambda_proto * loss_proto

        return loss, {
            'loss_seg': loss_seg.detach(),
            'loss_edge': loss_edge.detach(),
            'loss_proto': loss_proto.detach()
        }