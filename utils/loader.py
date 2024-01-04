import os
import cv2
import random
from utils import util
# import util
import numpy as np
from math import floor
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as T


class Dataset(data.Dataset):
    def __init__(self, args, params, task_type='train', transform=True):
        self.args = args
        self.params = params
        self.transform = transform
        self.task_type = task_type
        indices_file = os.path.join(self.params['data_dir'], 'images/indices.txt')
        self.mean_indices = util.compute_indices(indices_file, params['num_nb'])[0]
        self.samples = self.get_label(os.path.join(params['data_dir'], 'images/'), self.task_type)

        self.resize = T.Resize((self.args.input_size, self.args.input_size))
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = T.Compose([self.resize, T.ToTensor(), self.norm])

        self.transforms = (RandomTranslate(),
                           RandomBlur(),
                           RandomOcclusion(),
                           RandomFlip(params),
                           RandomRotate())

    def __getitem__(self, item):
        img_name, target = self.samples[item]
        image = Image.open(os.path.join(self.params['data_dir'], 'images', self.task_type, img_name)).convert('RGB')

        if self.transform:
            for tf in self.transforms:
                image, target = tf(image, target)

        image = self.normalize(image)
        if self.task_type == 'train':
            target = self.init_target(target, self.params['num_lms'], self.args.input_size, self.params['stride'],
                                      self.params['num_nb'], self.mean_indices)
        return image, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def init_target(target, num_lms, input_size, stride, num_nb, indices):
        size = int(input_size / stride)
        target_map = np.zeros((num_lms, size, size))
        target_local_x = np.zeros((num_lms, size, size))
        target_local_y = np.zeros((num_lms, size, size))
        target_nb_x = np.zeros((num_nb * num_lms, size, size))
        target_nb_y = np.zeros((num_nb * num_lms, size, size))

        map_ch, map_h, map_w = target_map.shape
        target = target.reshape(-1, 2)
        assert map_ch == target.shape[0]

        for i in range(map_ch):
            mu_x = int(floor(target[i][0] * map_w))
            mu_y = int(floor(target[i][1] * map_h))
            mu_x = max(0, mu_x)
            mu_y = max(0, mu_y)
            mu_x = min(mu_x, map_w - 1)
            mu_y = min(mu_y, map_h - 1)
            target_map[i, mu_y, mu_x] = 1
            shift_x = target[i][0] * map_w - mu_x
            shift_y = target[i][1] * map_h - mu_y
            target_local_x[i, mu_y, mu_x] = shift_x
            target_local_y[i, mu_y, mu_x] = shift_y

            for j in range(num_nb):
                nb_x = target[indices[i][j]][0] * map_w - mu_x
                nb_y = target[indices[i][j]][1] * map_h - mu_y
                target_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
                target_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y

        target_map = torch.from_numpy(target_map).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()
        targets = (target_map, target_local_x, target_local_y, target_nb_x, target_nb_y)

        return targets

    @staticmethod
    def get_label(data_dir, task_type='train'):
        label_path = os.path.join(data_dir, f'{task_type}.txt')
        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [x.strip().split() for x in labels]
        if len(labels[0]) == 1:
            return labels

        labels_new = []
        for label in labels:
            image_name = label[0]
            target = label[1:]
            target = np.array([float(x) for x in target])
            labels_new.append([image_name, target])

        return labels_new


# ------------------------------------ Image Transforms ---------------------------------------------------
# if you want to add or remove some augmentations change below transforms


class RandomTranslate:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            image_height, image_width = image.size
            a = 1
            b = 0
            c = int((random.random() - 0.5) * 60)
            d = 0
            e = 1
            f = int((random.random() - 0.5) * 60)
            image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
            label_translate = label.copy()
            label_translate = label_translate.reshape(-1, 2)
            label_translate[:, 0] -= 1. * c / image_width
            label_translate[:, 1] -= 1. * f / image_height
            label_translate = label_translate.flatten()
            label_translate[label_translate < 0] = 0
            label_translate[label_translate > 1] = 1
            return image, label_translate
        else:
            return image, label


class RandomBlur:
    def __init__(self, p=0.7):
        super().__init__()
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            blurring = ImageFilter.GaussianBlur(random.random() * self.p)
            image = image.filter(blurring)
        return image, label


class RandomOcclusion:
    def __init__(self, p=0.5, factor=0.4):
        super().__init__()
        self.p = p
        self.factor = factor

    def __call__(self, image, label):
        if random.random() > self.p:
            image_np = np.array(image).astype(np.uint8)
            image_np = image_np[:, :, ::-1]
            height, width, _ = image_np.shape
            occ_height = int(height * self.factor * random.random())
            occ_width = int(width * self.factor * random.random())
            occ_xmin = int((width - occ_width - 10) * random.random())
            occ_ymin = int((height - occ_height - 10) * random.random())
            image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 0] = int(random.random() * 255)
            image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 1] = int(random.random() * 255)
            image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 2] = int(random.random() * 255)
            image_pil = Image.fromarray(image_np[:, :, ::-1].astype('uint8'), 'RGB')
            return image_pil, label
        else:
            return image, label


class RandomFlip:
    def __init__(self, params, p=0.5):
        super().__init__()
        self.p = p
        self.points_flip = (np.array(params['points_id']) - 1).tolist()

    def __call__(self, image, label):
        if random.random() > self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label).reshape(-1, 2)
            label = label[self.points_flip, :]
            label[:, 0] = 1 - label[:, 0]
            label = label.flatten()
            return image, label
        else:
            return image, label


class RandomRotate:
    def __init__(self, p=0.5, angle=30):
        super().__init__()
        self.p = p
        self.angle = angle

    def __call__(self, image, label):
        if random.random() > self.p:
            center_x = 0.5
            center_y = 0.5
            landmark_num = int(len(label) / 2)
            label_center = np.array(label) - np.array([center_x, center_y] * landmark_num)
            label_center = label_center.reshape(landmark_num, 2)
            theta_max = np.radians(self.angle)
            theta = random.uniform(-theta_max, theta_max)
            angle = np.degrees(theta)
            image = image.rotate(angle)

            c, s = np.cos(theta), np.sin(theta)
            rot = np.array(((c, -s), (s, c)))
            label_center_rot = np.matmul(label_center, rot)
            label_rot = label_center_rot.reshape(landmark_num * 2) + np.array([center_x, center_y] * landmark_num)
            return image, label_rot
        else:
            return image, label

