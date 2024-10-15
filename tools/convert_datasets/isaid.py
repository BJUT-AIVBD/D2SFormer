# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv
import numpy as np
from PIL import Image

iSAID_palette = \
    {
        0: (0, 0, 0),
        1: (0, 0, 63),
        2: (0, 63, 63),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 191, 127),
        13: (0, 127, 191),
        14: (0, 127, 255),
        15: (0, 100, 155)
    }

iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}
PALETTE = [(0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0), (0, 63, 127),
           (0, 63, 191), (0, 63, 255), (0, 127, 63), (0, 127, 127),
           (0, 0, 127), (0, 0, 191), (0, 0, 255), (0, 191, 127),
           (0, 127, 191), (0, 127, 255), (0, 100, 155),(255, 255, 255)]


def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        a = np.array(c).reshape(1, 1, 3)
        m = np.all(arr_3d == a, axis=2)
        arr_2d[m] = i

    return arr_2d


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap):
    img = np.asarray(Image.open(src_path).convert('RGB'))

    img_H, img_W, _ = img.shape

    if img_H < patch_H and img_W > patch_W:

        img = mmcv.impad(img, shape=(patch_H, img_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H > patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(img_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H < patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(patch_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            img_patch = img[y_str:y_end, x_str:x_end, :]
            img_patch = Image.fromarray(img_patch.astype(np.uint8))
            image = osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            # print(image)
            save_path_image = osp.join(out_dir,  mode, 'images', str(image))
            img_patch.save(save_path_image)


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap):
    label = mmcv.imread(src_path, channel_order='rgb')
    label = iSAID_convert_from_color(label)

    img_H, img_W = label.shape

    if img_H < patch_H and img_W > patch_W:

        label = mmcv.impad(label, shape=(patch_H, img_W), pad_val=255)

        img_H = patch_H

    elif img_H > patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(img_H, patch_W), pad_val=255)

        img_W = patch_W

    elif img_H < patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(patch_H, patch_W), pad_val=255)

        img_H = patch_H
        img_W = patch_W

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            lab_patch = label[y_str:y_end, x_str:x_end]

            a = lab_patch.astype(np.uint8)
            lab_patch = Image.fromarray(a, mode='P')
            palette_bytes = bytes(i for color in PALETTE for i in color)
            lab_patch.putpalette(palette_bytes)
            image = osp.basename(src_path).split('.')[0].split(
                '_')[0] + '_' + str(y_str) + '_' + str(y_end) + '_' + str(
                    x_str) + '_' + str(x_end) + '_instance_color_RGB' + '.png'
            lab_patch.save(osp.join(out_dir,  mode, 'labels', str(image)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation format')
    parser.add_argument('--dataset_path',
                        default='/media/bdaksh/SSD256/iSAID',
                        help='iSAID folder path')
    # parser.add_argument('--tmp_dir',
    #                     help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir',
                        default='/media/bdaksh/SSD256/iSAID/896(200)',
                        help='output path')
    parser.add_argument(
        '--patch_width',
        default=896,
        type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height',
        default=896,
        type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=200, type=int, help='Overlap area')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    # image patch width and height
    patch_H, patch_W = args.patch_width, args.patch_height

    overlap = args.overlap_area  # overlap area

    if args.out_dir is None:
        out_dir = osp.join('data', 'iSAID')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'train', 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'val', 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'test', 'images'))

    mmcv.mkdir_or_exist(osp.join(out_dir, 'train', 'labels'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'val', 'labels'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'test', 'labels'))

    assert os.path.exists(os.path.join(dataset_path, 'train')), \
        'train is not in {}'.format(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, 'val')), \
        'val is not in {}'.format(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, 'test')), \
        'test is not in {}'.format(dataset_path)

    for dataset_mode in ['train', 'val']:
        src_path_img = os.path.join(dataset_path, dataset_mode, 'images')
        src_path_lab = os.path.join(dataset_path, dataset_mode, 'labels')
        for name in os.listdir(src_path_img):
            img_path = os.path.join(src_path_img,name)
            lab_path = os.path.join(src_path_lab,name.replace('.png','_instance_color_RGB.png'))
            slide_crop_image(img_path, out_dir, dataset_mode, patch_H, patch_W, overlap)
            slide_crop_label(lab_path, out_dir, dataset_mode, patch_H,patch_W, overlap)
            print(name)
    print('Done!')


if __name__ == '__main__':
    main()
