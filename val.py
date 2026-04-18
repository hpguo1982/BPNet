import argparse
import os
from glob import glob
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import Resize

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    args = parser.parse_args()
    return args


def seed_torch(seed=None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unwrap_output(output, target):
    """
    处理 deep supervision / tuple / dict 输出
    最终返回用于分割评估的主输出 logits
    """
    if isinstance(output, (tuple, list)):
        picked = None
        for o in output:
            if torch.is_tensor(o) and o.shape[-2:] == target.shape[-2:]:
                picked = o
                break
        output = picked if picked is not None else output[0]
    elif isinstance(output, dict):
        if 'out' in output:
            output = output['out']
        else:
            for v in output.values():
                if torch.is_tensor(v):
                    output = v
                    break
    return output


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list']
    )
    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'new-corn':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    else:
        mask_ext = '.png'

    dataset_root = config['data_dir']
    val_img_dir = os.path.join(dataset_root, config['dataset'], 'val', 'images')
    val_mask_dir = os.path.join(dataset_root, config['dataset'], 'val', 'masks')

    val_img_ids = sorted(glob(os.path.join(val_img_dir, '*' + img_ext)))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth', map_location='cpu')

    try:
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)

    model.eval()

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

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()

    mask_save_dir = os.path.join(args.output_dir, args.name, 'out_val_old')
    os.makedirs(mask_save_dir, exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            raw_output = model(input)
            output = unwrap_output(raw_output, target)

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            prob = torch.sigmoid(output).cpu().numpy()   # [B,1,H,W]
            output_bin = (prob >= 0.5).astype(np.uint8)

            for pred_bin, img_id in zip(output_bin, meta['img_id']):
                pred_mask = pred_bin[0]
                pred_np = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_np, 'L').save(
                    os.path.join(mask_save_dir, f'{img_id}.jpg')
                )

    print(args.name)
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)


if __name__ == '__main__':
    main()