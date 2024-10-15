import argparse
import os
import os.path as osp
import shutil
import mmcv
import numpy as np
from PIL import Image

loveda_palette = \
    {
        0: (255, 255, 255), #'background'，白
        1: (255, 0, 0),  #'building'
        2: (255, 255, 0), #road
        3: (0, 0, 255), # water
        4: (159, 129, 183), #barren
        5: (0, 255, 0), #forest
        6: (255, 195, 128) #agricultural
    }

loveda_invert_palette = {v: k for k, v in loveda_palette.items()}
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

# loveda_palette = \
#     {
#         0: (0, 0, 0), #忽略，黑
#         1: (255, 255, 255), #'background'，白
#         2: (255, 0, 0),  #'building'
#         3: (255, 255, 0), #road
#         4: (0, 0, 255), # water
#         5: (159, 129, 183), #barren
#         6: (0, 255, 0), #forest
#         7: (255, 195, 128) #agricultural
#     }

# loveda_invert_palette = {v: k for k, v in loveda_palette.items()}
# PALETTE = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
#             [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert loveda dataset to mmsegmentation format')
    # crop
    # parser.add_argument('--dataset_path',
    #                     default='/media/bdaksh/SSD256/2021LoveDA/yuan',
    #                     help='loveda folder path')
    # parser.add_argument('-o', '--out_dir',
    #                     default='/media/bdaksh/SSD256/2021LoveDA/new_512',
    #                     help='output path')

    #recover
    parser.add_argument('--names_path',
                        default='/media/bdaksh/SSD256/2021LoveDA/yuan/test',
                        help='loveda folder path')
    parser.add_argument('--dataset_path',
                        default='/home/bdaksh/Documents/yy/mmsegmentation-0.x/result/loveda/ftunetformer/test/image',
                        help='loveda folder path')
    parser.add_argument('-o', '--out_dir',
                        default='/home/bdaksh/Documents/yy/mmsegmentation-0.x/result/loveda/ftunetformer/test/recover_img',
                        help='output path')
    parser.add_argument(
        '--val',
        default=False,
        help='生成论文中主观图时用')
    parser.add_argument(
        '--crop',
        default=False,
        help='Width of the cropped image patch')

    parser.add_argument(
        '--patch_width',
        default=512,
        type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height',
        default=512,
        type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=0, type=int, help='Overlap area')
    args = parser.parse_args()
    return args


def loveda_convert_from_color(arr_3d, palette=loveda_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        a = np.array(c).reshape(1, 1, 3)
        m = np.all(arr_3d == a, axis=2)
        arr_2d[m] = i

    return arr_2d


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap, dataset_list):
    img = np.asarray(Image.open(src_path).convert('RGB'))

    img_H, img_W = img.shape[0], img.shape[1]

    if img_H < patch_H and img_W > patch_W:

        img = mmcv.impad(img, shape=(patch_H, img_W), pad_val=0)

        img_H, img_W = img.shape[0], img.shape[1]

    elif img_H > patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(img_H, patch_W), pad_val=0)

        img_H, img_W = img.shape[0], img.shape[1]

    elif img_H < patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(patch_H, patch_W), pad_val=0)

        img_H, img_W = img.shape[0], img.shape[1]

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
            if len(img.shape) == 3:
                img_patch = img[y_str:y_end, x_str:x_end, :]
            else:
                img_patch = img[y_str:y_end, x_str:x_end]
            img_patch = Image.fromarray(img_patch.astype(np.uint8))
            image = dataset_list + '_' + osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            # print(image)
            save_path_image = osp.join(out_dir,  mode, 'images', str(image))
            img_patch.save(save_path_image)


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap, dataset_list):
    # label = mmcv.imread(src_path, channel_order='rgb')
    label = np.asarray(Image.open(src_path))
    # label = loveda_convert_from_color(label)

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

            lab_patch = Image.fromarray(lab_patch.astype(np.uint8))
            # lab_patch = Image.fromarray(lab_patch, mode='P')
            # palette_bytes = bytes(i for color in PALETTE for i in color)
            # lab_patch.putpalette(palette_bytes)

            image = dataset_list + '_' +osp.basename(src_path).split('.')[0].split(
                '_')[0] + '_' + str(y_str) + '_' + str(y_end) + '_' + str(
                    x_str) + '_' + str(x_end) + '.png'
            lab_patch.save(osp.join(out_dir,  mode, 'labels', str(image)))


def stitch_patches(src_path, img_name, out_dir, name, val):
    # 找出所有匹配的文件
    patch_files = sorted([f for f in os.listdir(src_path) if img_name in f])

    # 假设我们知道原始图像的尺寸（仅为示例）
    stitched_img = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # 遍历所有小图像块并拼接到大图像中
    for patch_file in patch_files:
        # 解析文件名以获取位置信息（这里需要根据实际文件名格式进行解析）
        # 例如：'dataset_image_0_100_0_100.png' -> y_start=0, y_end=100, x_start=0, x_end=100
        parts = os.path.basename(patch_file).split('_')
        y_start = int(parts[-4])
        y_end = int(parts[-3])
        x_start = int(parts[-2])
        x_end = int(parts[-1].split('.')[0])

        # 读取小图像块并放置到大图像中
        patch_img = np.asarray(Image.open(os.path.join(src_path, patch_file)).convert('RGB'))
        stitched_img[y_start:y_end, x_start:x_end, :] = patch_img

    if val:
        stitched_img = Image.fromarray(stitched_img.astype(np.uint8))
    else:
        new_img = np.zeros((1024, 1024), dtype=np.uint8)
        for h in range(1024):
            for w in range(1024):
                index = stitched_img[h, w, :]  # 这里假设index是整数，并且位于调色板范围内
                # 检查索引是否在调色板中
                index = tuple(index.tolist())
                if index in loveda_invert_palette:
                    rgb_value = loveda_invert_palette[index]
                    new_img[h, w] = rgb_value
        # 将NumPy数组转换回PIL图像（如果需要）
        stitched_img = Image.fromarray(new_img.astype(np.uint8)).convert('L')
    if 'Rural' in img_name:
        save_path_image = osp.join(out_dir, 'Rural', str(name))
    else:
        save_path_image = osp.join(out_dir, 'Urban', str(name))
    stitched_img.save(save_path_image)


def crop(out_dir,dataset_path,patch_H, patch_W, overlap):
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

    for dataset_mode in ['train', 'val', 'test']:
        path_img = os.path.join(dataset_path, dataset_mode)
        path_lab = os.path.join(dataset_path, dataset_mode)
        for dataset_list in ['Rural', 'Urban']:
            src_path_img = os.path.join(path_img, dataset_list, 'images')
            src_path_lab = os.path.join(path_lab, dataset_list, 'masks')
            for name in os.listdir(src_path_img):
                img_path = os.path.join(src_path_img,name)
                slide_crop_image(img_path, out_dir, dataset_mode, patch_H, patch_W, overlap, dataset_list)
                if dataset_mode != 'test':
                    lab_path = os.path.join(src_path_lab, name)
                    slide_crop_label(lab_path, out_dir, dataset_mode, patch_H, patch_W, overlap,dataset_list)
                print(name)
    print('Crop Done!')


def recover(out_dir, dataset_path, names_path, val):
    '''
    out_dir: 输出文件夹
    dataset_path：输入文件夹
    '''
    print('Making directories...')

    mmcv.mkdir_or_exist(os.path.join(out_dir, 'Rural'))
    mmcv.mkdir_or_exist(os.path.join(out_dir, 'Urban'))
    for dataset_list in ['Rural', 'Urban']:
        src_path_img = os.path.join(names_path, dataset_list, 'images')
        for name in os.listdir(src_path_img):
            new_name = dataset_list + '_' + name
            stitch_patches(dataset_path, new_name.replace('.png',''), out_dir, name,val)
            print(new_name)
    print('recover Done!')


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    val = args.val
    if args.out_dir is None:
        out_dir = osp.join('data', 'loveda')
    else:
        out_dir = args.out_dir
    if args.crop:
        # image patch width and height
        patch_H, patch_W = args.patch_width, args.patch_height
        overlap = args.overlap_area  # overlap area
        crop(out_dir, dataset_path, patch_H, patch_W, overlap)
    else:
        names_path = args.names_path
        recover(out_dir, dataset_path, names_path, val)





if __name__ == '__main__':
    main()
