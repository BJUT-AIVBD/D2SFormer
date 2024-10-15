import os
import tifffile
import numpy as np
import cv2
import shutil

train_name = [7, 8, 9, 10, 11, 12]
test_name = [13, 14, 15]
val_name = ['2_10']


def imsave(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path[:-4] + '.png', img)


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def crop_overlap(img, label, vis, name, img_path, label_path, vis_path, size=256, stride=128):
    h, w, c = img.shape
    new_h, new_w = (h // size + 1) * size, (w // size + 1) * size
    num_h, num_w = (new_h // stride) - 1, (new_w // stride) - 1

    new_img = np.zeros([new_h, new_w, c]).astype(np.uint8)
    new_img[:h, :w, :] = img

    new_label = 255 * np.ones([new_h, new_w, 3]).astype(np.uint8)
    new_label[:h, :w, :] = label

    new_vis = np.zeros([new_h, new_w, 3]).astype(np.uint8)
    new_vis[:h, :w, :] = vis

    count = 0

    for i in range(num_h):
        for j in range(num_w):
            out = new_img[i * stride:i * stride + size, j * stride:j * stride + size, :]
            gt = new_label[i * stride:i * stride + size, j * stride:j * stride + size, :]
            v = new_vis[i * stride:i * stride + size, j * stride:j * stride + size, :]
            assert v.shape == (256, 256, 3), print(v.shape)

            tifffile.imsave(img_path + '/' + str(name) + '_' + str(count) + '.tif', out)
            tifffile.imsave(label_path + '/' + str(name) + '_' + str(count) + '.tif', gt)
            tifffile.imsave(vis_path + '/' + str(name) + '_' + str(count) + '.tif', v)

            count += 1


def crop(img, label, name, img_path, label_path, size=600):
    height, width, _ = img.shape
    h_size = height // size
    w_size = width // size

    count = 0

    for i in range(h_size):
        for j in range(w_size):
            out = img[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            gt = label[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            # v = vis[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            # assert v.shape == (512, 512, 3)

            imsave(img_path + '/' + str(name) + '_' + str(count) + '.jpg', out)
            imsave(label_path + '/' + str(name) + '_' + str(count) + '.png', gt)
            # imsave(vis_path + '/' + str(name) + '_' + str(count) + '.png', v)

            count += 1


def save(img, label, name, save_path, flag='val'):
    img_path = save_path + '/image'
    label_path = save_path + '/mask_pseudo'
    # v_path = save_path + '/label_vis'
    if check_dir(img_path):
        os.mkdir(img_path)
    if check_dir(label_path):
        os.mkdir(label_path)
    crop(img, label, name, img_path, label_path)


def run(root_path):
    img_path = root_path + '/images'
    label_path = root_path + '/labels'
    # vis_path = root_path + '/annotations/labels'

    list = os.listdir(label_path)
    for i in list:
        img = cv2.imread(os.path.join(img_path, i.replace('.png','.tif')))
        label = cv2.imread(os.path.join(label_path, i))
        # vis = cv2.imread(os.path.join(vis_path, i))

        name = i[:-4]
        num = name.split('_')[3]
        num2 = name.split('_')[2:]

        if num2 in val_name:
            print(name, 'val')
            val_path = root_path + '/val'
            check_dir(val_path)
            save(img, label, name, val_path, flag='val')

        elif int(num) in test_name:
            print(name, 'test')
            test_path = root_path+'/test'
            check_dir(test_path)
            save(img, label, name, test_path, flag='test')

        else:
            print(name, 'train')
            train_path = root_path+'/train'
            check_dir(train_path)
            save(img, label, name, train_path, flag='train')


def move(name, old, save):
    imgpath = os.path.join(old,'image')
    labelpath = os.path.join(old,'mask_pseudo')
    imgnames = os.listdir(imgpath)
    labelnames = os.listdir(labelpath)
    os.mkdir(os.path.join(save,'image'))
    os.mkdir(os.path.join(save, 'mask_pseudo'))
    for i in imgnames:
        if name in i:
            shutil.move(os.path.join(imgpath,i), os.path.join(save,'image',i))
            print(i)
    for i in labelnames:
        if name in i:
            shutil.move(os.path.join(labelpath, i), os.path.join(save, 'mask_pseudo', i))

if __name__ == '__main__':
    name = '2_10_'
    root_path = '/media/bdaksh/SSD256/seg_datasets/old_potsdam'
    old = '/media/bdaksh/SSD256/seg_datasets/old_potsdam/train'
    new = '/media/bdaksh/SSD256/seg_datasets/old_potsdam/val'
    if check_dir(new):
        os.mkdir(new)

    move(name,old,new)
    # run(root_path)