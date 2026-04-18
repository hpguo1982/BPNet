#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from tqdm import tqdm

import archs
from dataset import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# -------------------------
# Grad-CAM (segmentation)
# -------------------------
class GradCAMSeg:
    """
    Grad-CAM for segmentation logits.
    target_layer must output feature maps [B,C,h,w].
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(m, inp, out):
            self._acts = out

        def bwd_hook(m, grad_in, grad_out):
            self._grads = grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def _norm01(self, cam):
        B = cam.size(0)
        v = cam.view(B, -1)
        mn = v.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        mx = v.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        return (cam - mn) / (mx - mn + 1e-6)

    def __call__(self, x, scalar, use_amp=True):
        """
        x: [B,3,H,W]
        scalar: a scalar tensor to backprop from
        returns cam: [B,1,H,W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        # backward
        scalar.backward(retain_graph=False)

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM got no activations/grads. Choose another target_layer.")

        w = grads.mean(dim=(2, 3), keepdim=True)               # [B,C,1,1]
        cam = (w * acts).sum(dim=1, keepdim=True)             # [B,1,h,w]
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = self._norm01(cam)
        return cam


def find_default_cam_layer(model):
    # pick last Conv2d with out_channels > 1 (avoid 1-channel head conv)
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and getattr(m, "out_channels", 0) > 1:
            last = m
    return last


# -------------------------
# Region control (where to generate CAM)
# -------------------------
def build_region_mask(region, target, pred_mask, H, W, box=None, point=None):
    """
    returns roi_mask: [B,1,H,W] float 0/1 or None (for 'all')
    region:
      - 'all'  : no mask, use full map
      - 'gt'   : use target as ROI
      - 'pred' : use predicted mask as ROI
      - 'box'  : use box ROI (x1,y1,x2,y2) in resized space
      - 'point': use a single point (x,y) -> small square ROI
    """
    if region == "all":
        return None

    if region == "gt":
        if target.ndim == 3:
            roi = target.unsqueeze(1).float()
        else:
            roi = target.float()
        roi = (roi > 0.5).float()
        return roi

    if region == "pred":
        return pred_mask.float()

    if region == "box":
        assert box is not None and len(box) == 4
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(W - 1, x1)))
        x2 = int(max(0, min(W, x2)))
        y1 = int(max(0, min(H - 1, y1)))
        y2 = int(max(0, min(H, y2)))
        roi = torch.zeros((target.shape[0], 1, H, W), device=target.device, dtype=torch.float32)
        roi[:, :, y1:y2, x1:x2] = 1.0
        return roi

    if region == "point":
        assert point is not None and len(point) == 2
        x, y = point
        x = int(max(0, min(W - 1, x)))
        y = int(max(0, min(H - 1, y)))
        r = 8  # point radius -> (2r+1)x(2r+1)
        x1, x2 = max(0, x - r), min(W, x + r + 1)
        y1, y2 = max(0, y - r), min(H, y + r + 1)
        roi = torch.zeros((target.shape[0], 1, H, W), device=target.device, dtype=torch.float32)
        roi[:, :, y1:y2, x1:x2] = 1.0
        return roi

    raise ValueError(f"Unknown region: {region}")


def scalar_from_logits(logits, roi_mask=None):
    """
    logits: [B,1,H,W]
    roi_mask: [B,1,H,W] or None
    returns a scalar to backprop
    """
    if roi_mask is None:
        return logits.mean()
    denom = roi_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)      # [B]
    num = (logits * roi_mask).sum(dim=(1, 2, 3))            # [B]
    return (num / denom).mean()


# -------------------------
# Visualization
# -------------------------
def overlay_cam_on_bgr(img_bgr, cam01, alpha=0.45):
    """
    img_bgr: [H,W,3] uint8
    cam01:   [H,W] float in [0,1]
    returns heat_bgr, overlay_bgr
    """
    heat_u8 = (np.clip(cam01, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)
    return heat_bgr, overlay_bgr


# -------------------------
# Boilerplate
# -------------------------
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='model name')
    p.add_argument('--output_dir', default='outputs', help='output dir')

    # CAM output
    p.add_argument('--save_dir', default=None, help='where to save CAM images. default: outputs/<name>/gradcam')
    p.add_argument('--cam_layer', default='decoder5', help='preferred layer attribute name, e.g., decoder5')
    p.add_argument('--alpha', type=float, default=0.45, help='overlay alpha')
    p.add_argument('--use_amp', action='store_true', help='use AMP to reduce memory')

    # region control
    p.add_argument('--region', default='all', choices=['all', 'gt', 'pred', 'box', 'point'],
                   help="where to generate CAM: all/gt/pred/box/point")
    p.add_argument('--box', nargs=4, type=int, default=None, metavar=('x1', 'y1', 'x2', 'y2'),
                   help='ROI box in resized space (after Resize(input_h,input_w))')
    p.add_argument('--point', nargs=2, type=int, default=None, metavar=('x', 'y'),
                   help='ROI point in resized space (after Resize)')
    p.add_argument('--pred_thr', type=float, default=0.5, help='threshold for pred mask if region=pred')

    return p.parse_args()


def main():
    seed_torch()
    args = parse_args()

    # load config
    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cudnn.benchmark = True

    # build model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list']
    ).cuda()

    # load ckpt
    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth', map_location='cpu')
    try:
        model.load_state_dict(ckpt)
    except:
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=False)
    model.eval()

    # dataset paths
    dataset_name = config['dataset']
    img_ext = '.png'
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    else:
        mask_ext = '.png'

    dataset_root = config['data_dir']
    val_img_dir = os.path.join(dataset_root, config['dataset'], 'val', 'images')
    val_mask_dir = os.path.join(dataset_root, config['dataset'], 'val', 'masks')

    val_img_ids = sorted(glob(os.path.join(val_img_dir, '*' + img_ext)))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # CAM layer
    target_layer = getattr(model, args.cam_layer, None)
    if target_layer is None:
        target_layer = find_default_cam_layer(model)
        if target_layer is None:
            raise RuntimeError("Cannot find a suitable Conv2d layer for CAM.")
    cam_engine = GradCAMSeg(model, target_layer=target_layer)

    # save dir
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(args.output_dir, args.name, 'gradcam')
    os.makedirs(save_dir, exist_ok=True)

    H_in, W_in = config['input_h'], config['input_w']

    # ONLY: generate heatmaps (and overlays)
    for inp, target, meta in tqdm(val_loader, total=len(val_loader)):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward (needs grad for CAM)
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                out = model(inp)

                # handle deep supervision outputs (tuple/list/dict)
                if isinstance(out, (tuple, list)):
                    picked = None
                    for o in out:
                        if torch.is_tensor(o) and o.ndim == 4 and o.shape[-2:] == target.shape[-2:]:
                            picked = o
                            break
                    out = picked if picked is not None else out[0]
                elif isinstance(out, dict):
                    if 'out' in out:
                        out = out['out']
                    else:
                        for v in out.values():
                            if torch.is_tensor(v):
                                out = v
                                break

                logits = out  # expected [B,1,H,W]

                # if region=pred, build pred mask first (no_grad OK but keep it simple)
                if args.region == "pred":
                    pred = (torch.sigmoid(logits) > args.pred_thr).float()  # [B,1,H,W]
                else:
                    pred = None

                roi = build_region_mask(
                    region=args.region,
                    target=target,
                    pred_mask=pred if pred is not None else torch.zeros_like(logits),
                    H=logits.shape[-2],
                    W=logits.shape[-1],
                    box=args.box,
                    point=args.point
                )

                scalar = scalar_from_logits(logits, roi_mask=roi)

            cam = cam_engine(inp, scalar=scalar, use_amp=args.use_amp)  # [B,1,H,W]

        cam_np = cam.detach().cpu().numpy()

        # overlay on resized original image (same space as model input)
        for b, img_id in enumerate(meta['img_id']):
            img_path = os.path.join(val_img_dir, img_id + img_ext)
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (W_in, H_in), interpolation=cv2.INTER_LINEAR)

            cam01 = cam_np[b, 0].astype(np.float32)
            heat_bgr, overlay_bgr = overlay_cam_on_bgr(img_bgr, cam01, alpha=args.alpha)

            cv2.imwrite(os.path.join(save_dir, f"{img_id}_cam_heat.png"), heat_bgr)
            cv2.imwrite(os.path.join(save_dir, f"{img_id}_cam_overlay.png"), overlay_bgr)

    print(f"[DONE] Saved CAM to: {save_dir}")


if __name__ == '__main__':
    main()
