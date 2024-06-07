from math import floor
from os.path import dirname, join

from torch.utils import data
import torchvision.transforms as T

from utils.util import *


class Dataset(data.Dataset):
    def __init__(self,
                 params,
                 data_dir,
                 augment=True):

        self.params = params
        self.augment = augment
        self.data_dir = data_dir
        self.samples = self.load_label(self.data_dir)
        self.mean_indices = compute_indices(join(dirname(data_dir), 'indices.txt'), params)[0]

        self.resize = T.Resize((self.params['input_size'], self.params['input_size']))
        self.normalize = T.Compose(
            [self.resize, T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transforms = (RandomHSV(),
                           RandomRotate(),
                           RandomCutOut(),
                           RandomRGB2IR(),
                           RandomTranslate(),
                           RandomFlip(params),
                           RandomGaussianBlur())

    def __getitem__(self, item):
        img_name, label = self.samples[item]
        image = Image.open(os.path.join(img_name)).convert('RGB')

        if not self.augment:
            image = self.normalize(image)
            return image, label

        for transform in self.transforms:
            image, label = transform(image, label)

        image = self.normalize(image)

        reduced_size = int(self.params['input_size'] / self.params['stride'])

        target_map = np.zeros((self.params['num_lms'], reduced_size, reduced_size))
        target_local_x = np.zeros((self.params['num_lms'], reduced_size, reduced_size))
        target_local_y = np.zeros((self.params['num_lms'], reduced_size, reduced_size))
        target_nb_x = np.zeros((self.params['num_nb'] * self.params['num_lms'], reduced_size, reduced_size))
        target_nb_y = np.zeros((self.params['num_nb'] * self.params['num_lms'], reduced_size, reduced_size))
        ground_points = target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

        target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = self.init_target(label,
                                                                                                self.mean_indices,
                                                                                                self.params['num_nb'],
                                                                                                ground_points)

        target_map = torch.from_numpy(target_map).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()

        return image, (target_map, target_local_x, target_local_y, target_nb_x, target_nb_y)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_label(data_dir):
        with open(data_dir, 'r') as f:
            labels = f.readlines()
        labels = [x.strip().split() for x in labels]
        if len(labels[0]) == 1:
            return labels

        labels_new = []
        for label in labels:
            image_name, target = label[0], label[1:]
            labels_new.append([image_name, np.array([float(x) for x in target])])
        return labels_new

    @staticmethod
    def init_target(target, indices, num_nb, labels):
        target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = labels

        ch, h, w = target_map.shape
        target = target.reshape(-1, 2)
        assert ch == target.shape[0]

        for i in range(ch):
            mu_x = min(max(0, int(floor(target[i][0] * w))), w - 1)
            mu_y = min(max(0, int(floor(target[i][1] * h))), h - 1)
            target_map[i, mu_y, mu_x] = 1
            target_local_x[i, mu_y, mu_x] = target[i][0] * w - mu_x
            target_local_y[i, mu_y, mu_x] = target[i][1] * h - mu_y

            for j in range(num_nb):
                nb_x = target[indices[i][j]][0] * w - mu_x
                nb_y = target[indices[i][j]][1] * h - mu_y
                target_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
                target_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y

        return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y
