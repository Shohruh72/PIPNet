import argparse
import os
import cv2
import csv
import yaml
import copy
import tqdm
import torch
import numpy as np

from nets import nn
from utils import util
from timm import utils
from utils.dataset import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args, params):
    # Seed
    util.setup_seed()
    # Model
    model = nn.PIPNet(params, args.resnet_type, *util.compute_indices(f'{args.data_dir}/indices.txt', params))
    model.cuda()

    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params['init_lr'], weight_decay=1e-4)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

    # Dataset
    dataset = Dataset(params, f'{args.data_dir}/train.txt', True)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(dataset, args.batch_size, not args.distributed,
                        sampler, num_workers=8, pin_memory=True, drop_last=True)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank])

    # Start Training ...
    best = float('inf')
    num_batch = len(loader)
    criterion = util.ComputeLoss(params)
    amp_scale = torch.cuda.amp.GradScaler()
    with open('outputs/weights/logs.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'NME'])
            writer.writeheader()

        for epoch in range(args.epochs):
            model.train()
            p_bar = enumerate(loader)
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_batch)

            for i, (images, labels) in p_bar:
                images = images.to(device)
                labels = [label.to(device) for label in labels]
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                amp_scale.step(optimizer)
                amp_scale.update(None)
                optimizer.zero_grad()

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                avg_loss.update(loss.item(), images.size(0))
                if args.local_rank == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', loss.item())
                    p_bar.set_description(s)

            # Scheduler

            if args.local_rank == 0:
                # Start Testing ...
                last = test(args, params, model)
                scheduler.step(last)
                writer.writerow({'NME': str(f'{last:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                f.flush()

                # Update best NME
                if best > last:
                    best = last

                # Model Save
                ckpt = {'model': copy.deepcopy(model).half()}

                # Save last and best result
                torch.save(ckpt, './outputs/weights/last.pt')
                if best == last:
                    torch.save(ckpt, './outputs/weights/best.pt')
                del ckpt
                print(f"Best NME = {best:.3f}")

    if args.local_rank == 0:
        util.strip_optimizer('./outputs/weights/best.pt')
        util.strip_optimizer('./outputs/weights/last.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    index = [36, 45]
    loader = DataLoader(Dataset(params, f'{args.data_dir}/test.txt', False))

    if model is None:
        model = torch.load(f'./outputs/weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    nme_merge = []
    for sample, target in tqdm.tqdm(loader, '%20s' % 'NME'):
        sample = sample.to(device)
        sample = sample.half()

        output = model(sample).detach().cpu().numpy()

        target = target.view(-1)
        target = target.cpu().numpy()

        a = target.reshape(-1, 2)
        b = target.reshape(-1, 2)
        norm = np.linalg.norm(a[index[0]] - b[index[1]])
        nme_merge.append(util.compute_nme(output, target, norm))

    nme = np.mean(nme_merge) * 100
    print(f"Last NME = {nme:.3f}")
    model.float()
    return nme


@torch.no_grad()
def demo(params):
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)

    model = torch.load('./outputs/weights/best.pt', 'cuda')
    model = model['model'].float()

    detector = util.FaceDetector('./outputs/weights/detection.onnx')

    model.half()
    model.eval()

    scale = 1.2
    stream = cv2.VideoCapture(0)

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
                image = cv2.resize(image, (params['input_size'], params['input_size']))
                image = image.astype('float32')
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, 1 / std, image)  # inplace
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).unsqueeze(0)

                image = image.cuda()
                image = image.half()

                output = model(image)

                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 2)

            cv2.imshow('IMAGE', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stream.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../Datasets/LMK/300W/images')
    parser.add_argument('--resnet_type', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(os.path.join('utils', 'cfg.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(params)


if __name__ == "__main__":
    main()
