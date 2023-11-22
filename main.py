import os
import csv
import cv2
import tqdm
import numpy as np

import torch
from torchvision import models
from torch.utils.data import DataLoader

from utils import util
from models.pipnet import PIPNet
from utils.dataset import Dataset
from utils.face_detector import FaceDetector

depth = '101'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args, params):
    backbone = getattr(models, f"resnet{depth}")(params['pretrained'])
    model = PIPNet(params, backbone, depth).to(device)
    weight_decay = 0 if params['pretrained'] else 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['decay_steps'], gamma=0.1)
    criterion = util.ComputeLoss(params)

    dataset = Dataset(args, params, params['data_dir'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                        drop_last=True)
    weight_dir = os.path.join(params['save_dir'], 'weights')
    os.makedirs(weight_dir, exist_ok=True)

    with open(f'{weight_dir}/step_{depth}.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'NME'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            p_bar = tqdm.tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{args.epochs}")
            for i, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = {k: v.cuda() for k, v in labels.items()}
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_bar.set_postfix(NME=loss.item(), refresh=False)
                p_bar.update()
            p_bar.close()

            filename = os.path.join(params['save_dir'], 'weights', f'last_{depth}.pth')
            torch.save(model.state_dict(), filename)

            scheduler.step()
            writer.writerow({'NME': str(f'{loss.item():.3f}'),
                             'epoch': str(epoch + 1).zfill(3)})
        return model


def test(args, params):
    backbone = getattr(models, f"resnet{depth}")(params['pretrained'])
    model = PIPNet(params, backbone, depth).to(device)
    meanface_indices, reverse_index1, reverse_index2, max_len = util.get_meanface(
        os.path.join(params['data_dir'], 'meanface.txt'), params['num_nb'])
    weight_file = os.path.join(params['save_dir'], 'weights', f'last_{depth}.pth')
    model.load_state_dict(torch.load(weight_file))
    dataset = Dataset(args, params, params['data_dir'], 'test', False)
    loader = DataLoader(dataset)

    nmes_merge = []
    for i, (images, labels) in enumerate(loader):
        image_name, lms_gt = dataset.images[i]
        images = images.to(device)
        eye_norm = lms_gt.reshape(-1, 2)
        norm = np.linalg.norm(eye_norm[params['norm_indices'][0]] - eye_norm[params['norm_indices'][1]])

        lms_pred_merge = util.forward_pip(params, model, images, args.input_size, reverse_index1, reverse_index2,
                                          max_len)

        nme_merge = util.compute_nme(lms_pred_merge, lms_gt, norm)
        nmes_merge.append(nme_merge)

    print('nme: {}'.format(np.mean(nmes_merge)))
    fr, auc = util.compute_fr_and_auc(nmes_merge)
    print('fr : {}'.format(fr))
    print('auc: {}'.format(auc))


@torch.no_grad()
def demo(args, params):
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    meanface_indices, reverse_index1, reverse_index2, max_len = util.get_meanface(
        os.path.join(params['data_dir'], 'meanface.txt'), params['num_nb'])
    backbone = getattr(models, f"resnet{depth}")(params['pretrained'])
    model = PIPNet(params, backbone, depth).to(device)

    state_dict = torch.load('./output/weights/best.pth', 'cuda')
    model.load_state_dict(state_dict)
    detector = FaceDetector('./output/weights/detection.onnx')

    scale = 1.2
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("Error opening video stream or file")

    w = int(stream.get(3))
    h = int(stream.get(4))
    out = cv2.VideoWriter(params['save_dir'], cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))
    while stream.isOpened():
        success, frame = stream.read()
        if success:
            boxes = detector.detect(frame, (640, 640))
            boxes = boxes.astype('int32')
            for box in boxes:
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                box_w = x_max - x_min
                box_h = y_max - y_min

                x_min -= int(box_w * (scale - 1) / 2)
                y_min += int(box_h * (scale - 1) / 2)
                x_max += int(box_w * (scale - 1) / 2)
                y_max += int(box_h * (scale - 1) / 2)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w - 1)
                y_max = min(y_max, h - 1)
                box_w = x_max - x_min + 1
                box_h = y_max - y_min + 1
                image = frame[y_min:y_max, x_min:x_max, :]
                image = cv2.resize(image, (args.input_size, args.input_size))
                image = image.astype('float32')
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                cv2.subtract(image, mean, image)
                cv2.multiply(image, 1 / std, image)
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).unsqueeze(0)

                image = image.cuda()
                output = util.forward_pip(params, model, image, args.input_size, reverse_index1, reverse_index2,
                                          max_len)

                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 255), 1)

            cv2.imshow('IMAGE', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stream.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    with open(os.path.join('utils', 'configs.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args, params)


if __name__ == "__main__":
    main()
