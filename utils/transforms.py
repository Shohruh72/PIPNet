import random
import numpy as np
from PIL import Image, ImageFilter


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
        self.points_flip = (np.array(params['points_flip']) - 1).tolist()

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


