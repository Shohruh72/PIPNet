import argparse
import os
import cv2
import csv
import yaml
import copy
import tqdm
import time
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

    model = torch.load('./weights/best.pt', 'cuda', weights_only=False)
    model = model['model'].float()

    detector = util.FaceDetector('./weights/detection.onnx')

    model.half()
    model.eval()

    scale = 1.2
    stream = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    w = int(stream.get(3))
    h = int(stream.get(4))

    writer = cv2.VideoWriter('demo.avi',
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             int(stream.get(cv2.CAP_PROP_FPS)), (w, h))

    # Performance tracking variables
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    processing_times = []

    # Read until video is completed
    while stream.isOpened():
        frame_start_time = time.time()

        # Capture frame-by-frame
        success, frame = stream.read()
        if success:
            # Create dark overlay for professional look
            overlay = frame.copy()

            boxes = detector.detect(frame, (640, 640))
            boxes = boxes.astype('int32')

            for idx, box in enumerate(boxes):
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                box_w = x_max - x_min
                box_h = y_max - y_min

                # Enhanced bounding box with gradient effect
                confidence_color = (0, 255, 100) if box_w > 80 else (0, 200, 255)

                # Draw enhanced bounding box
                cv2.rectangle(frame, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), confidence_color, 3)
                cv2.rectangle(frame, (x_min - 1, y_min - 1), (x_max + 1, y_max + 1), (255, 255, 255), 1)

                # Face ID label with background
                face_label = f"Face #{idx + 1}"
                label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x_min, y_min - 25), (x_min + label_size[0] + 10, y_min), confidence_color, -1)
                cv2.putText(frame, face_label, (x_min + 5, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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

                output, score = model(image)

                # Enhanced landmark visualization with individual confidence
                landmark_colors = [
                    (255, 0, 255),  # Magenta for jaw line
                    (0, 255, 255),  # Cyan for eyebrows
                    (255, 255, 0),  # Yellow for eyes
                    (0, 255, 0),  # Green for nose
                    (255, 100, 100),  # Light red for mouth
                ]

                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)

                    # Color based on landmark region
                    color_idx = min(i // (params['num_lms'] // 5), 4)
                    base_color = landmark_colors[color_idx]

                    # Adjust color intensity based on individual landmark confidence
                    landmark_conf = float(score[i])
                    intensity = max(0.4, min(1.0, landmark_conf))  # Keep between 40% and 100%
                    color = tuple(int(c * intensity) for c in base_color)

                    # Enhanced landmark points with confidence-based sizing
                    point_size = int(landmark_conf * 2)  # Size based on confidence
                    cv2.circle(frame, (x + x_min, y + y_min), point_size + 1, (255, 255, 255), -1)  # White core
                    cv2.circle(frame, (x + x_min, y + y_min), point_size, color, -1)  # Colored center
                    cv2.circle(frame, (x + x_min, y + y_min), point_size + 2, color, 1)  # Colored outline

                # Add confidence score for this face (average of all landmark scores)
                avg_score = float(torch.mean(score))
                min_score = float(torch.min(score))
                conf_text = f"Avg: {avg_score:.3f} | Min: {min_score:.3f}"
                cv2.putText(frame, conf_text, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 25, 25), 1)

            # Professional UI overlay with 50% transparent gray rectangle
            ui_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, ui_height), (128, 128, 128), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.rectangle(frame, (0, 0), (400, ui_height), (100, 100, 100), 2)

            # Calculate FPS
            frame_count += 1
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            else:
                fps = fps_counter / max(current_time - fps_start_time, 0.001)

            # Processing time
            processing_time = (time.time() - frame_start_time) * 1000
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_processing_time = sum(processing_times) / len(processing_times)

            # Status indicators
            status_y = 25
            cv2.putText(frame, "FACIAL LANDMARK DETECTION SYSTEM", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            status_y += 20
            cv2.putText(frame, f"Frame: {frame_count:05d} | FPS: {fps:.1f}", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1)

            status_y += 15
            cv2.putText(frame, f"Faces: {len(boxes):02d} | Proc: {avg_processing_time:.1f}ms", (10, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            status_y += 15
            cv2.putText(frame, f"Resolution: {w}x{h}", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                        1)

            # System status indicator
            status_color = (0, 255, 0) if len(boxes) > 0 else (0, 100, 255)
            status_text = "ACTIVE" if len(boxes) > 0 else "STANDBY"
            cv2.circle(frame, (370, 25), 8, status_color, -1)
            cv2.putText(frame, status_text, (320, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (209, 20, 26), 1)

            # Progress bar for processing load
            bar_width = 200
            bar_height = 8
            bar_x, bar_y = 180, 85
            load_percentage = min(avg_processing_time / 50.0, 1.0)  # Normalize to 50ms max

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * load_percentage), bar_y + bar_height),
                          (0, 255 - int(255 * load_percentage), int(255 * load_percentage)), -1)
            cv2.putText(frame, "Processing Load", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            # Instructions at bottom
            instruction_text = "Press 'Q' to quit | 'S' to screenshot"
            cv2.putText(frame, instruction_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Subtle scanning line effect for tech feel
            scan_y = int((time.time() * 50) % h)
            cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 255), 1)

            cv2.imshow('IMAGE', frame)
            writer.write(frame.astype('uint8'))

            # Enhanced keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Screenshot functionality
                screenshot_name = f"landmark_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot saved: {screenshot_name}")
        else:
            break

    # Cleanup with final stats
    total_time = time.time() - fps_start_time + 1  # Add 1 to avoid initial time
    print(f"\n=== SESSION COMPLETE ===")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    print(f"Average Processing Time: {sum(processing_times) / len(processing_times):.2f}ms")
    print(f"Session Duration: {total_time:.1f}s")

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
