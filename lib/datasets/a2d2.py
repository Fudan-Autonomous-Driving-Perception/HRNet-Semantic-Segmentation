import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class A2D2(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(A2D2, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {(255, 0, 0): 0,
                              (200, 0, 0): 0,
                              (150, 0, 0): 0,
                              (128, 0, 0): 0,
                              (182, 89, 6): 1,
                              (150, 50, 4): 1,
                              (90, 30, 1):  1,
                              (90, 30, 30): 1,
                              (204, 153, 255): 2,
                              (189, 73, 155):  2,
                              (239, 89, 191):  2,
                              (255, 128, 0): 3,
                              (200, 128, 0): 3,
                              (150, 128, 0): 3,
                              (0, 255, 0): 4,
                              (0, 200, 0): 4,
                              (0, 150, 0): 4,
                              (0, 128, 255): 5,
                              (30, 28, 158): 5,
                              (60, 28, 100): 5,
                              (0, 255, 255):  6,
                              (30, 220, 220): 6,
                              (60, 157, 199): 6,
                              (255, 255, 0):   7,
                              (255, 255, 200): 7,
                              (233, 100, 0): 8,
                              (110, 110, 0): 9,
                              (128, 128, 0): 10,
                              (255, 193, 37): 11,
                              (64, 0, 64): 12,
                              (185, 122, 87): 13,
                              (0, 0, 100): 14,
                              (139, 99, 108): 15,
                              (210, 50, 115): 16,
                              (255, 0, 128): 17,
                              (255, 246, 143): 18,
                              (150, 0, 150): 19,
                              (204, 255, 153): 20,
                              (238, 162, 173): 21,
                              (33, 44, 177): 22,
                              (180, 50, 180): 23,
                              (255, 70, 185): 24,
                              (238, 233, 191): 25,
                              (147, 253, 194): 26,
                              (150, 150, 200): 27,
                              (180, 150, 200): 28,
                              (72, 209, 204): 29,
                              (200, 125, 210): 30,
                              (159, 121, 238): 31,
                              (128, 0, 255): 32,
                              (255, 0, 255): 33,
                              (135, 206, 255): 34,
                              (241, 230, 255): 35,
                              (96, 69, 143): 36,
                              (53, 46, 82): 37}
        self.label_mapping_inv = {v:k for k,v in self.label_mapping.items()}
        self.class_weights = torch.FloatTensor([1., 1., 1., 1., 1., 1., 1.5, 1., 1., 1.5, 
                                                1., 1., 1., 1., 1., 1., 1., 1., 1.5, 1., 
                                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
                                                1.5, 1., 1., 1., 1., 1., 1., 1.]).cuda()
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        if inverse:
            label_inv = np.zeros_like(label)
            for l in range(38):
                label_inv[label == l] = self.label_mapping_inv[l]
            label = label_inv
        else:
            W = np.power(256, [[0],[1],[2]])
            lb_id = label.dot(W).squeeze(-1)
            values = np.unique(lb_id)
            mask = np.zeros(lb_id.shape)
            for c in values:
                mask[lb_id==c] = self.label_mapping[tuple(label[lb_id==c][0])] 
            label = mask
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        image = cv2.imread(item["img"],
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
        #                    cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item["label"],
                           cv2.IMREAD_COLOR)
        label = self.convert_label(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
