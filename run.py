import argparse
import copy
import csv
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import tqdm
import yaml
from torch.utils import data

from model import net
from utils import loader
from utils import util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args, params):
    # Model
    resnet = models.resnet18(True)
    indices_file = os.path.join(params['data_dir'], 'images/indices.txt')
    indices, rev_index1, rev_index2, max_len = util.compute_indices(indices_file, params['num_nb'])
    model = net.PIPNet(args, params, resnet, rev_index1, rev_index2, max_len).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['decay_steps'], gamma=0.1)

    # EMA
    ema = util.EMA(model)

    # Loader
    train_loader = data.DataLoader(loader.Dataset(args, params),
                                   args.batch_size, True,
                                   num_workers=8, pin_memory=True, drop_last=True)
    weight_dir = os.path.join(params['output_dir'], 'weights')
    os.makedirs(weight_dir, exist_ok=True)

    # START TRAINING
    num_batch = len(train_loader)
    criterion = util.ComputeLoss(params)
    with open(f'{weight_dir}/step.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'NME'])
        writer.writeheader()
        print('Training...', '-' * 100)

        for epoch in range(args.epochs):
            model.train()

            print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            m_loss = util.AverageMeter()
            p_bar = tqdm.tqdm(iterable=enumerate(train_loader), total=num_batch)

            for i, (samples, targets) in p_bar:
                samples = samples.to(device)
                loss = criterion(model(samples), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ema:
                    ema.update(model)

                m_loss.update(loss.item(), samples.size(0))
                s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', m_loss.avg)
                p_bar.set_description(s)

            writer.writerow({'NME': str(f'{loss.item():.3f}'),
                             'epoch': str(epoch + 1).zfill(3)})

            ckpt = {'model': copy.deepcopy(ema.ema).half()}
            torch.save(ckpt, os.path.join(weight_dir, 'last.pt'))

            scheduler.step()
            del ckpt


@torch.no_grad()
def test(args, params, model=None):
    index = params['index']
    test_loader = data.DataLoader(loader.Dataset(args, params, 'test', False))
    weight_dir = os.path.join(params['output_dir'], 'weights')
    if model is None:
        model = torch.load(f'{weight_dir}/last.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    nme_merge = []
    for sample, target in tqdm.tqdm(test_loader, '%20s' % 'NME'):
        sample = sample.cuda()
        sample = sample.half()

        output = model(sample)
        output = output[0].cpu().numpy()  # prediction
        score = output[1].cpu().numpy()  # confidence score
        target = target.view(-1)
        target = target.cpu().numpy()

        a = target.reshape(-1, 2)
        b = target.reshape(-1, 2)
        norm = np.linalg.norm(a[index[0]] - b[index[1]])
        nme_merge.append(util.compute_nme(output, target, norm))

    nme = np.mean(nme_merge)
    print('%20.3g' % nme)
    fr, auc = util.compute_fr_and_auc(nme_merge)
    model.float()
    return nme, fr, auc


@torch.no_grad()
def demo(args, params):
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    weight_dir = os.path.join(params['output_dir'], 'weights')
    model = torch.load(f'{weight_dir}/last.pt', 'cuda')
    model = model['model'].float()
    detector = util.FaceDetector(f'{weight_dir}/detection.onnx')

    model.half()
    model.eval()

    scale = 1.2
    stream = cv2.VideoCapture(-1)

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    w = int(stream.get(3))
    h = int(stream.get(4))

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
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

                # remove a part of top area for alignment, see paper for details
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
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, 1 / std, image)  # inplace
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).unsqueeze(0)

                image = image.cuda()
                image = image.half()

                output = model(image)[0].cpu().numpy()

                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 1)

            cv2.imshow('IMAGE', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    stream.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)

    detector = util.FaceDetector('./outputs/weights/detection.onnx')

    landmark = util.LandmarkDetector('./outputs/weights/last.pt')

    scale = 1.2
    stream = cv2.VideoCapture(-1)

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    w = 650
    h = 600

    writer = cv2.VideoWriter('demo.avi',
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             int(stream.get(cv2.CAP_PROP_FPS)), (w, h))

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
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

                # remove a part of top area for alignment, see paper for details
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.stack((image, image, image), -1)
                image = image.astype("float32")
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, 1 / std, image)  # inplace
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = np.expand_dims(image, 0)

                output, score = landmark(image)  # confidence score
                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 1)

            cv2.imshow('IMAGE', frame)

            writer.write(frame.astype('uint8'))

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    stream.release()
    writer.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', default=True, action='store_true')
    parser.add_argument('--export_onnx', action='store_true')

    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'opt.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args, params)
    if args.export_onnx:
        util.export(args, 'outputs/weights/last.pt')


if __name__ == "__main__":
    main()
