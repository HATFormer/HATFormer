import os
import numpy as np
import tifffile as tiff
import skimage
import imageio as iio
import cv2

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu


def center_crop(imm, size, imtype='image'):
    h = int(size[0] / 2)
    w = int(size[1] / 2)
    ch = int(imm.shape[0] / 2)
    cw = int(imm.shape[1] / 2)
    if imtype == 'image':
        return imm[ch - h:ch + h, cw - w:cw + w, :]
    else:
        return imm[ch - h:ch + h, cw - w:cw + w]


def extract_names(root, sets):
    if len(sets) == 5:
        t1_images_dir = os.path.join(root, sets[0])
        t2_images_dir = os.path.join(root, sets[1])
        masks2d_dir = os.path.join(root, sets[2])
        masks3d_dir = os.path.join(root, sets[3])
        height_dir = os.path.join(root, sets[4])
    elif len(sets) == 4:
        t1_images_dir = os.path.join(root, sets[0])
        t2_images_dir = os.path.join(root, sets[1])
        masks2d_dir = os.path.join(root, sets[2])
        masks3d_dir = os.path.join(root, sets[3])

    ids = os.listdir(t1_images_dir)

    # important!
    ids = [x for x in ids if x.endswith('.tif')]

    t1_images_fps = [os.path.join(t1_images_dir, image_id) for image_id in ids]
    t2_images_fps = [os.path.join(t2_images_dir, image_id) for image_id in ids]
    masks2d_fps = [os.path.join(masks2d_dir, image_id) for image_id in ids]
    masks3d_fps = [os.path.join(masks3d_dir, image_id) for image_id in ids]

    if len(sets) == 5:
        height_fps = [os.path.join(height_dir, image_id) for image_id in ids]
        return t1_images_fps, t2_images_fps, masks2d_fps, masks3d_fps, height_fps
    else:
        return t1_images_fps, t2_images_fps, masks2d_fps, masks3d_fps


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    """

    def __init__(
            self,
            roots,
            sets=[],
            augmentation=False,
            return_crs=False
    ):
        if type(roots) != list:
            roots = [roots]
        self.t1_images_fps = []
        self.t2_images_fps = []
        self.masks2d_fps = []
        self.masks3d_fps = []
        self.height_fps = []
        self.custom_lut = self.genColormap()
        for root in roots:
            if len(sets)==5:
                r1, r2, r3, r4, r5 = extract_names(root, sets)
                self.t1_images_fps += r1
                self.t2_images_fps += r2
                self.masks2d_fps += r3
                self.masks3d_fps += r4
                self.height_fps += r5
            else:
                r1, r2, r3, r4 = extract_names(root, sets)
                self.t1_images_fps += r1
                self.t2_images_fps += r2
                self.masks2d_fps += r3
                self.masks3d_fps += r4


        self.augmentation = augmentation
        self.return_crs = return_crs
        self.sets_len = len(sets)

    def norm(self, x, M, m):
        return (x - m) / (M - m+1e-6)

    def rescale(self, img):
        
        def norm(x, M, m):
            return (x-m)/(M-m)
        img[img>10**8]=0
        M = 99
        m = -10

        img = norm(img,M,m)
        img = img * 255.0

        return img

    def genColormap(self):
        custom_lut = []  # np.zeros((256, 1, 3), dtype=np.uint8)

        def convert_line(line):
            return [int(x) for x in line.split(',')[1:4]]

        lines = open('./utils/colormap.txt', 'r').readlines()
        for idx in range(len(lines) - 1):
            lcolor = convert_line(lines[idx])
            rcolor = convert_line(lines[idx + 1])
            if idx == 0:
                custom_lut.append(lcolor)

            R = np.linspace(lcolor[0], rcolor[0], 6, dtype=int)[1:]
            G = np.linspace(lcolor[1], rcolor[1], 6, dtype=int)[1:]
            B = np.linspace(lcolor[2], rcolor[2], 6, dtype=int)[1:]

            for r, g, b in zip(R, G, B):
                custom_lut.append([r, g, b])

        # import pdb;pdb.set_trace()
        return np.array(custom_lut, dtype=np.uint8).reshape(256, 1, 3)

    def vis_3d(self, img, custom_lut, mM=None):

        if mM is None:
            m = img.min()#27.29
            M = img.max()#83.26
        else:
            m, M = mM

        img_gray = np.uint8(255 * self.norm(img, M, m))
        img_gray = np.stack([img_gray, img_gray, img_gray], axis=2)

        img_color = cv2.LUT(img_gray, custom_lut)
        return img_color

    def save_dsm(self, x, save_dir):
        color_img = self.vis_3d(x,self.custom_lut)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir, cv2.resize(color_img,(512,512)))

    def __getitem__(self, i):

        # read data with tifffile because of 3d mask int16
        t1 = iio.imread(self.t1_images_fps[i])[:, :, :3]  # .transpose([2,0,1])
        t2 = iio.imread(self.t2_images_fps[i])  # [:,:,:3]#.transpose([2,0,1])
        t2[t2 > 10 ** 5] = 0

        t1_save_dir = r'/home/hc/lby/HATFormer/results/CD_hatformer_baseline_and_BHE_FME_AFA_smars/image'
        t2_save_dir = r'/home/hc/lby/HATFormer/results/CD_hatformer_baseline_and_BHE_FME_AFA_smars/dsm'
        os.makedirs(t1_save_dir,exist_ok=True)
        os.makedirs(t2_save_dir,exist_ok=True)
        name = self.t1_images_fps[i].split('/')[-1].replace('.tif','.png')
        #cv2.imwrite(os.path.join(t1_save_dir,name), cv2.cvtColor(cv2.resize(t1,(512,512)), cv2.COLOR_RGB2BGR))


        dsm_mask = t2.copy()
        dsm_mask[dsm_mask > 10 ** 5] = 0

        
        mask2d = iio.imread(self.masks2d_fps[i])
        mask3d = tiff.imread(self.masks3d_fps[i])
        #if 'SParis_dsm2img_08_08' in self.t2_images_fps[i]:

        #self.save_dsm(t2, os.path.join(t2_save_dir,name))

        t2 = self.rescale(t2)
        if t2.shape[0]!=3:
            t2 = np.stack([t2, t2, t2], axis=2)



        mask2d[mask2d == 3] = 2
        #dsm_mask = dsm_mask*(mask2d==0)
        dsm_mask += mask3d.copy()

        if self.sets_len == 5:
            height = tiff.imread(self.height_fps[i])
        
        # apply augmentations
        if self.augmentation:
            t1 = np.uint8(t1)
            if len(t2.shape)==3:
                t2[np.isinf(t2)]=0
                t2 = np.uint8(t2)
            #import pdb;pdb.set_trace()
            if self.sets_len == 5:
                sample = self.augmentation(image=t1, t2=t2, mask=mask2d, mask3d=mask3d, height=height)
                
                t1, t2, mask2d, mask3d, height = sample['image'], sample['t2'], sample['mask'], sample['mask3d'], sample['height']
 
            else:
                sample = self.augmentation(image=t1, t2=t2, mask=mask2d, mask3d=mask3d, dsm_mask=dsm_mask)
                t1, t2, mask2d, mask3d, dsm_mask = sample['image'], sample['t2'],\
                                                   sample['mask'], sample['mask3d'], sample['dsm_mask']


        if self.return_crs:
            if self.sets_len == 5:
                return t1, t2, mask2d, mask3d.float(), height.float(), self.t1_images_fps[i]
            else:
                return t1, t2, mask2d, mask3d.float(), dsm_mask, self.t1_images_fps[i]
        else:
            if self.sets_len == 5:
                return t1, t2, mask2d, mask3d.float(), height.float()
            else:
                return t1, t2, mask2d, mask3d.float(), dsm_mask

    def __len__(self):
        return len(self.t1_images_fps)
