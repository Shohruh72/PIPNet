import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as T
from math import floor
from utils import util
from utils.transforms import *


class Dataset(data.Dataset):
    def __init__(self, args, params, data_dir, task_type='train', transform=True):
        self.data_dir = data_dir
        self.task_type = task_type
        self.transform = transform
        self.stride = params['stride']
        self.num_nb = params['num_nb']
        self.num_lms = params['num_lms']
        self.input_size = args.input_size
        self.mean_indices, _, _, _ = util.get_meanface(os.path.join(self.data_dir, 'meanface.txt'), self.num_nb)

        self.images = self.get_label(params['data_dir'], self.task_type)

        self.resize = T.Resize((self.input_size, self.input_size))
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = T.Compose([self.resize, T.ToTensor(), self.norm])

        self.transforms = (RandomTranslate(),
                           RandomBlur(),
                           RandomOcclusion(),
                           RandomFlip(params),
                           RandomRotate())

    def __getitem__(self, index):

        img_name, target = self.images[index]
        image = self.load_image(img_name)

        if self.transform:
            for tf in self.transforms:
                image, target = tf(image, target)

        image = self.normalize(image)
        labels = self.initialize_target_maps()
        targets = self.gen_target_pip(target, self.mean_indices, labels)

        return image, targets

    def load_image(self, img_name):
        return Image.open(os.path.join(self.data_dir, 'images', self.task_type, img_name)).convert('RGB')

    def initialize_target_maps(self):
        map_size = int(self.input_size / self.stride)
        lb_map = np.zeros((self.num_lms, map_size, map_size))
        lb_x = np.zeros((self.num_lms, map_size, map_size))
        lb_y = np.zeros((self.num_lms, map_size, map_size))
        lb_nb_x = np.zeros((self.num_nb * self.num_lms, map_size, map_size))
        lb_nb_y = np.zeros((self.num_nb * self.num_lms, map_size, map_size))
        labels = {"lb_map": lb_map, "lb_x": lb_x, "lb_y": lb_y, "lb_nb_x": lb_nb_x, "lb_nb_y": lb_nb_y}
        return labels

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_label(data_dir, task_type='train'):
        label_path = os.path.join(data_dir, task_type + '.txt')
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

    @staticmethod
    def gen_target_pip(target, meanface_indices, labels):
        lb_map = labels['lb_map']
        lb_x = labels['lb_x']
        lb_y = labels['lb_y']
        lb_nb_x = labels['lb_nb_x']
        lb_nb_y = labels['lb_nb_y']

        num_nb = len(meanface_indices[0])
        map_channel, map_height, map_width = lb_map.shape
        target = target.reshape(-1, 2)
        assert map_channel == target.shape[0]

        for i in range(map_channel):
            mu_x = int(floor(target[i][0] * map_width))
            mu_y = int(floor(target[i][1] * map_height))
            mu_x = max(0, mu_x)
            mu_y = max(0, mu_y)
            mu_x = min(mu_x, map_width - 1)
            mu_y = min(mu_y, map_height - 1)
            lb_map[i, mu_y, mu_x] = 1
            shift_x = target[i][0] * map_width - mu_x
            shift_y = target[i][1] * map_height - mu_y
            lb_x[i, mu_y, mu_x] = shift_x
            lb_y[i, mu_y, mu_x] = shift_y

            for j in range(num_nb):
                nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
                nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
                lb_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
                lb_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y
        lb_map = torch.from_numpy(lb_map).float()
        lb_x = torch.from_numpy(lb_x).float()
        lb_y = torch.from_numpy(lb_y).float()
        lb_nb_x = torch.from_numpy(lb_nb_x).float()
        lb_nb_y = torch.from_numpy(lb_nb_y).float()

        return {'lb_map': lb_map, 'lb_x': lb_x, 'lb_y': lb_y, 'lb_nb_x': lb_nb_x, 'lb_nb_y': lb_nb_y}
        # return lb_map, lb_x, lb_y, lb_nb_x, lb_nb_y
