import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir=None, mask_dir=None,
                 img_ext='.jpg', mask_ext='.png',
                 num_classes=1, transform=None,
                 color_tolerance=10):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.color_tolerance = int(color_tolerance)

        self.bg_color_rgb = np.array([0, 0, 0], dtype=np.int16)
        self.leaf_color_rgb = np.array([128, 0, 0], dtype=np.int16)

        self.use_sample_dict = False
        if len(img_ids) > 0 and isinstance(img_ids[0], dict):
            self.use_sample_dict = True

    def __len__(self):
        return len(self.img_ids)

    def _color_mask_to_binary_lesion(self, mask_bgr):
        """
        彩色 mask 转二值病斑 mask。

        黑色 RGB 0,0,0 表示背景。
        红色 RGB 128,0,0 表示叶片主体。
        其他所有颜色表示病斑。
        """
        if mask_bgr is None:
            raise RuntimeError("mask_bgr is None")

        if mask_bgr.ndim == 2:
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        mask_rgb = mask_rgb.astype(np.int16)

        bg_diff = np.abs(mask_rgb - self.bg_color_rgb.reshape(1, 1, 3))
        leaf_diff = np.abs(mask_rgb - self.leaf_color_rgb.reshape(1, 1, 3))

        is_bg = np.all(bg_diff <= self.color_tolerance, axis=2)
        is_leaf = np.all(leaf_diff <= self.color_tolerance, axis=2)

        lesion = (~is_bg) & (~is_leaf)

        binary_mask = lesion.astype(np.uint8) * 255

        return binary_mask

    def _read_mask_as_binary(self, mask_path, image_shape):
        if mask_path and os.path.exists(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_COLOR)

            if mask_raw is None:
                raise RuntimeError(f"Failed to read mask, {mask_path}")

            mask = self._color_mask_to_binary_lesion(mask_raw)
        else:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)

        return mask[..., None]

    def _read_from_sample_dict(self, idx):
        sample = self.img_ids[idx]

        image_path = sample['image_path']
        mask_path = sample.get('mask_path', '')
        img_id = sample.get('img_id', os.path.splitext(os.path.basename(image_path))[0])
        class_id = int(sample.get('class_id', -1))
        class_name = sample.get('class_name', '')

        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(f"Failed to read image, {image_path}")

        mask = self._read_mask_as_binary(mask_path, img.shape)

        return img, mask, img_id, class_id, class_name

    def _read_from_old_format(self, idx):
        img_id = self.img_ids[idx]

        image_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(f"Failed to read image, {image_path}")

        mask_list = []

        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)

            if not os.path.exists(mask_path):
                raise RuntimeError(f"Mask not found, {mask_path}")

            mask_i = self._read_mask_as_binary(mask_path, img.shape)
            mask_list.append(mask_i)

        mask = np.dstack(mask_list)

        return img, mask, img_id, 0, ''

    def __getitem__(self, idx):
        if self.use_sample_dict:
            img, mask, img_id, class_id, class_name = self._read_from_sample_dict(idx)
        else:
            img, mask, img_id, class_id, class_name = self._read_from_old_format(idx)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask[..., None]

        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)

        mask = (mask > 0).astype('float32')
        mask = mask.transpose(2, 0, 1)

        return img, mask, {
            'img_id': img_id,
            'class_id': class_id,
            'class_name': class_name
        }