import os
import cv2
import copy
import math
import random

import numpy as np
import torch
from scipy.integrate import simps


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def get_criterion(criterion_name):
    criteria = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }
    return criteria.get(criterion_name)


def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme


def compute_fr_and_auc(nme, thresh=0.1, step=0.0001):
    num_data = len(nme)
    xs = np.arange(0, thresh + step, step)
    ys = np.array([np.count_nonzero(nme <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thresh
    return fr, auc


def compute_indices(indices_file, num_nb):
    with open(indices_file) as f:
        indices = f.readlines()[0]

    indices = indices.strip().split()
    indices = [float(x) for x in indices]
    indices = np.array(indices).reshape(-1, 2)
    mean_indices = []
    for i in range(indices.shape[0]):
        pt = indices[i, :]
        dists = np.sum(np.power(pt - indices, 2), axis=1)
        indices_sort = np.argsort(dists)
        mean_indices.append(indices_sort[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    mean_indices_reversed = {}
    for i in range(indices.shape[0]):
        mean_indices_reversed[i] = [[], []]
    for i in range(indices.shape[0]):
        for j in range(num_nb):
            mean_indices_reversed[mean_indices[i][j]][0].append(i)
            mean_indices_reversed[mean_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(indices.shape[0]):
        tmp_len = len(mean_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(indices.shape[0]):
        tmp_len = len(mean_indices_reversed[i][0])
        mean_indices_reversed[i][0] += mean_indices_reversed[i][0] * 10
        mean_indices_reversed[i][1] += mean_indices_reversed[i][1] * 10
        mean_indices_reversed[i][0] = mean_indices_reversed[i][0][:max_len]
        mean_indices_reversed[i][1] = mean_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(indices.shape[0]):
        reverse_index1 += mean_indices_reversed[i][0]
        reverse_index2 += mean_indices_reversed[i][1]
    return mean_indices, reverse_index1, reverse_index2, max_len


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class ComputeLoss:
    def __init__(self, params):
        super().__init__()
        self.cls = params['cls_weight']
        self.reg = params['reg_weight']
        self.num_neighbor = params['num_nb']
        self.criterion_reg = torch.nn.L1Loss()
        self.criterion_cls = torch.nn.MSELoss()

    def __call__(self, outputs, targets):
        device = outputs[0].device
        b, c, h, w = outputs[0].size()

        score = outputs[0]
        offset_x = outputs[1].view(b * c, -1)
        offset_y = outputs[2].view(b * c, -1)
        neighbor_x = outputs[3].view(b * self.num_neighbor * c, -1)
        neighbor_y = outputs[4].view(b * self.num_neighbor * c, -1)

        target_score = targets[0].to(device).view(b * c, -1)
        target_offset_x = targets[1].to(device).view(b * c, -1)
        target_offset_y = targets[2].to(device).view(b * c, -1)
        target_neighbor_x = targets[3].to(device).view(b * self.num_neighbor * c, -1)
        target_neighbor_y = targets[4].to(device).view(b * self.num_neighbor * c, -1)

        target_max_index = torch.argmax(target_score, 1).view(-1, 1)
        target_max_index_neighbor = target_max_index.repeat(1, self.num_neighbor).view(-1, 1)

        offset_x_select = torch.gather(offset_x, 1, target_max_index)
        offset_y_select = torch.gather(offset_y, 1, target_max_index)
        neighbor_x_select = torch.gather(neighbor_x, 1, target_max_index_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, target_max_index_neighbor)

        target_offset_x_select = torch.gather(target_offset_x, 1, target_max_index)
        target_offset_y_select = torch.gather(target_offset_y, 1, target_max_index)
        target_neighbor_x_select = torch.gather(target_neighbor_x, 1, target_max_index_neighbor)
        target_neighbor_y_select = torch.gather(target_neighbor_y, 1, target_max_index_neighbor)

        loss_cls = self.criterion_cls(score, target_score.view(b, c, h, w))
        loss_offset_x = self.criterion_reg(offset_x_select, target_offset_x_select)
        loss_offset_y = self.criterion_reg(offset_y_select, target_offset_y_select)
        loss_neighbor_x = self.criterion_reg(neighbor_x_select, target_neighbor_x_select)
        loss_neighbor_y = self.criterion_reg(neighbor_y_select, target_neighbor_y_select)

        loss_cls = self.cls * loss_cls
        loss_reg = self.reg * (loss_offset_x + loss_offset_y + loss_neighbor_x + loss_neighbor_y)
        return loss_cls + loss_reg


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x,
                                     1.0 / 128,
                                     input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = outputs[idx][0]
                boxes = outputs[idx + fmc][0]
                boxes = boxes * stride
            else:
                scores = outputs[idx]
                boxes = outputs[idx + fmc]
                boxes = boxes * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = np.where(scores >= score_thresh)[0]
            bboxes = self.distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list

    def detect(self, image, input_size=None, score_threshold=0.5, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if image_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list = self.forward(det_img, score_threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            index = np.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
        return det

    def nms(self, outputs):
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep

    @staticmethod
    def distance2box(points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


def export(args, weight_path):
    import onnx  # noqa
    import torch  # noqa
    import onnxsim  # noqa

    model = torch.load(weight_path)['model'].float().cpu()
    model.eval()
    image = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(model,
                      image,
                      weight_path.replace('pt', 'onnx'),
                      verbose=False,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['inputs'],
                      output_names=['outputs', 'score'],
                      dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(weight_path.replace('pt', 'onnx'))  # load onnx model

    # Simplify
    try:
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'Simplified ONNX model could not be validated'
    except Exception as e:
        print(e)

    onnx.save(model_onnx, weight_path.replace('pt', 'onnx'))


class LandmarkDetector:
    def __init__(self, onnx_path=None, session=None):
        self.session = session
        from onnxruntime import InferenceSession

        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CPUExecutionProvider'])
        self.output_names = []
        for output in self.session.get_outputs():
            self.output_names.append(output.name)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        outputs = self.session.run(self.output_names, {self.input_name: x})
        # output = outputs[0]
        # score = outputs[1]
        return outputs
