import os
import cv2
import torch
import numpy as np
from scipy.integrate import simps


def get_criterion(criterion_name):
    criteria = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }
    return criteria.get(criterion_name)


class ComputeLoss:
    def __init__(self, params):
        self.num_nb = params['num_nb']
        self.cls_weight = params['cls_weight']
        self.reg_weight = params['reg_weight']
        self.criterion_cls = get_criterion(params['criterion_cls'])
        self.criterion_reg = get_criterion(params['criterion_reg'])

    def __call__(self, outputs, labels):
        output_keys = ['cls_layer', 'x_layer', 'y_layer', 'nb_x_layer', 'nb_y_layer']
        label_keys = ['lb_map', 'lb_x', 'lb_y', 'lb_nb_x', 'lb_nb_y']
        outputs_map, outputs_local_x, outputs_local_y, outputs_nb_x, outputs_nb_y = (outputs[key] for key in
                                                                                     output_keys)
        labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y = (labels[key] for key in label_keys)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
        labels_map = labels_map.view(tmp_batch * tmp_channel, -1)
        labels_max_ids = torch.argmax(labels_map, 1)
        labels_max_ids = labels_max_ids.view(-1, 1)
        labels_max_ids_nb = labels_max_ids.repeat(1, self.num_nb).view(-1, 1)

        outputs_local_x = outputs_local_x.view(tmp_batch * tmp_channel, -1)
        outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
        outputs_local_y = outputs_local_y.view(tmp_batch * tmp_channel, -1)
        outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
        outputs_nb_x = outputs_nb_x.view(tmp_batch * self.num_nb * tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
        outputs_nb_y = outputs_nb_y.view(tmp_batch * self.num_nb * tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

        labels_local_x = labels_x.view(tmp_batch * tmp_channel, -1)
        labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
        labels_local_y = labels_y.view(tmp_batch * tmp_channel, -1)
        labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
        labels_nb_x = labels_nb_x.view(tmp_batch * self.num_nb * tmp_channel, -1)
        labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
        labels_nb_y = labels_nb_y.view(tmp_batch * self.num_nb * tmp_channel, -1)
        labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

        labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
        loss_map = self.criterion_cls(outputs_map, labels_map)
        loss_x = self.criterion_reg(outputs_local_x_select, labels_local_x_select)
        loss_y = self.criterion_reg(outputs_local_y_select, labels_local_y_select)
        loss_nb_x = self.criterion_reg(outputs_nb_x_select, labels_nb_x_select)
        loss_nb_y = self.criterion_reg(outputs_nb_y_select, labels_nb_y_select)
        cls_loss = self.cls_weight * loss_map
        reg_loss = self.reg_weight * (loss_x + loss_y + loss_nb_x + loss_nb_y)
        return cls_loss + reg_loss


def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]

    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len


def forward_pip(params, model, inputs, input_size, reverse_index1, reverse_index2, max_len):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        output_keys = ['cls_layer', 'x_layer', 'y_layer', 'nb_x_layer', 'nb_y_layer']
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = (outputs[key] for key in output_keys)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()

        assert tmp_batch == 1

        outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)
        max_ids = torch.argmax(outputs_cls, 1)
        max_cls = torch.max(outputs_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, params['num_nb']).view(-1, 1)

        outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        outputs_nb_x = outputs_nb_x.view(tmp_batch * params['num_nb'] * tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, params['num_nb'])
        outputs_nb_y = outputs_nb_y.view(tmp_batch * params['num_nb'] * tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, params['num_nb'])

        lms_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)
        lms_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_y_select.view(-1, 1)
        lms_x /= 1.0 * input_size / params['stride']
        lms_y /= 1.0 * input_size / params['stride']

        lms_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select
        lms_nb_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_nb_y_select
        lms_nb_x = lms_nb_x.view(-1, params['num_nb'])
        lms_nb_y = lms_nb_y.view(-1, params['num_nb'])
        lms_nb_x /= 1.0 * input_size / params['stride']
        lms_nb_y /= 1.0 * input_size / params['stride']

        lms_pred = torch.cat((lms_x, lms_y), dim=1).flatten().cpu().numpy()
        tmp_nb_x = lms_nb_x[reverse_index1, reverse_index2].view(params['num_lms'], max_len)
        tmp_nb_y = lms_nb_y[reverse_index1, reverse_index2].view(params['num_lms'], max_len)
        tmp_x = torch.mean(torch.cat((lms_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()

    return lms_pred_merge


def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme


def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc


def process(root_folder, folder_name, image_name, label_name, target_size):
    image_path = os.path.join(root_folder, folder_name, image_name)
    label_path = os.path.join(root_folder, folder_name, label_name)

    with open(label_path, 'r') as ff:
        anno = ff.readlines()
        anno = [x.strip().split() for x in anno]
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]
        anno_y = [x[1] for x in anno]
        bbox_xmin = min(anno_x)
        bbox_ymin = min(anno_y)
        bbox_xmax = max(anno_x)
        bbox_ymax = max(anno_y)
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        scale = 1.1
        bbox_xmin -= int((scale - 1) / 2 * bbox_width)
        bbox_ymin -= int((scale - 1) / 2 * bbox_height)
        bbox_width *= scale
        bbox_height *= scale
        bbox_width = int(bbox_width)
        bbox_height = int(bbox_height)
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_width = min(bbox_width, image_width - bbox_xmin - 1)
        bbox_height = min(bbox_height, image_height - bbox_ymin - 1)
        anno = [[(x - bbox_xmin) / bbox_width, (y - bbox_ymin) / bbox_height] for x, y in anno]

        bbox_xmax = bbox_xmin + bbox_width
        bbox_ymax = bbox_ymin + bbox_height
        image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]
        image_crop = cv2.resize(image_crop, (target_size, target_size))
        return image_crop, anno


def convert(root_folder, target_size):
    if not os.path.exists(os.path.join(root_folder, 'images', 'train')):
        os.makedirs(os.path.join(root_folder, 'images', 'train'))
    if not os.path.exists(os.path.join(root_folder, 'images', 'test')):
        os.makedirs(os.path.join(root_folder, 'images', 'test'))

    folder_names = ['aihub', 'data_voucher', 'dibox', 'nir_face', 'zerone']
    folders_train = [name + '/train' for name in folder_names]
    annos_train = {}
    for folder_train in folders_train:
        all_files = sorted(os.listdir(os.path.join(root_folder, folder_train)))
        image_files = [x for x in all_files if '.pts' not in x]
        label_files = [x for x in all_files if '.pts' in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            print(image_name)
            image_crop, anno = process(os.path.join(root_folder), folder_train, image_name,
                                       label_name, target_size)
            image_crop_name = folder_train.replace('/', '_') + '_' + image_name
            cv2.imwrite(os.path.join(root_folder, 'images/train', image_crop_name), image_crop)
            annos_train[image_crop_name] = anno
    with open(os.path.join(root_folder, 'train.txt'), 'w') as f:
        for image_crop_name, anno in annos_train.items():
            f.write(image_crop_name + ' ')
            for x, y in anno:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    folder_names = ['aihub', 'data_voucher', 'dibox', 'nir_face', 'zerone']
    folders_test = [name + '/test' for name in folder_names]
    annos_test = {}
    for folder_test in folders_test:
        all_files = sorted(os.listdir(os.path.join(root_folder, folder_test)))
        image_files = [x for x in all_files if '.pts' not in x]
        label_files = [x for x in all_files if '.pts' in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            print(image_name)
            image_crop, anno = process(os.path.join(root_folder), folder_test, image_name,
                                       label_name, target_size)
            image_crop_name = folder_test.replace('/', '_') + '_' + image_name
            cv2.imwrite(os.path.join(root_folder, 'images/test', image_crop_name), image_crop)
            annos_test[image_crop_name] = anno
    with open(os.path.join(root_folder, 'test.txt'), 'w') as f:
        for image_crop_name, anno in annos_test.items():
            f.write(image_crop_name + ' ')
            for x, y in anno:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    annos = None
    with open(os.path.join(root_folder, 'test.txt'), 'r') as f:
        annos = f.readlines()
    with open(os.path.join(root_folder, 'test_common.txt'), 'w') as f:
        for anno in annos:
            if not 'ibug' in anno:
                f.write(anno)
    with open(os.path.join(root_folder, 'test_challenge.txt'), 'w') as f:
        for anno in annos:
            if 'ibug' in anno:
                f.write(anno)

    with open(os.path.join(root_folder, 'train.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]
    annos = [[float(x) for x in anno] for anno in annos]
    annos = np.array(annos)
    meanface = np.mean(annos, axis=0)
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]

    with open(os.path.join(root_folder, 'indices.txt'), 'w') as f:
        f.write(' '.join(meanface))


convert('../Dataset/IR', 256)
