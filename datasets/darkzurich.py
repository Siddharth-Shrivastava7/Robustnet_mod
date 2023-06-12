"""
Darkzurich Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms
from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import datasets.cityscapes_labels as cityscapes_labels

from config import cfg

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
num_classes = 19
ignore_label = 255
root = cfg.DATASET.DARKZURICH_DIR 
img_postfix = '_rgb_anon.png'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(items, img_path, mask_path, mask_postfix):


    c_items = [name.split(img_postfix)[0] for name in
                os.listdir(os.path.join(img_path))]
    for it in c_items:
        item = (os.path.join(img_path, it + img_postfix),
                os.path.join(mask_path, it + mask_postfix))
        ########################################################
        ###### dataset augmentation ############################
        ########################################################
        items.append(item)



def make_dataset(mode):
    
    items = []
    
    assert mode in ['train', 'val', 'test', 'trainval']
    img_dir_name = 'rgb_anon'
    img_path = os.path.join(root, img_dir_name, 'val/night/GOPR0356')
    mask_path = os.path.join(root, 'gt', 'val/night/GOPR0356')
    mask_postfix = '_gt_labelTrainIds.png'
    add_items(items, img_path, mask_path,
                        mask_postfix)
    
    logging.info('Darkzurich-{}: {} images'.format(mode, len(items)))
    return items



class DarkZurich(data.Dataset):

    def __init__(self, mode,  joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None, target_aux_transform=None, dump_images=False,
                 eval_mode=False,
                 eval_scales=None, eval_flip=False, image_in=False, extract_feature=False):
        
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        self.image_in = image_in
        self.extract_feature = extract_feature  

        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(final_tensor)
            return_imgs.append(imgs)
        return return_imgs, mask

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)

        if self.eval_mode == 'pooling':
            return [transforms.ToTensor()(img)], self._eval_get_item(img, mask,
                                                                     self.eval_scales,
                                                                     self.eval_flip), img_name

        mask = Image.fromarray(mask.astype(np.uint8))

        # Image Transformations
        if self.extract_feature is not True:
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        rgb_mean_std_gt = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_gt = transforms.Normalize(*rgb_mean_std_gt)(img)

        if not self.eval_mode:
            rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if self.image_in:
                eps = 1e-5
                rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                        [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
            img = transforms.Normalize(*rgb_mean_std)(img)

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        return img, mask, img_name, mask_aux

    def __len__(self):
        return len(self.imgs)


