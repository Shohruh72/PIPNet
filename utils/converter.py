import os
import cv2
import numpy as np

''' 
    * Dataset Converting (Preprocessing)
    * Download the datasets from official sources: 300W
    * To preprocess the 300W dataset use below convert.py function.
    * -- data_dir  --> Dataset structure
       |-- afw
       |-- helen
       |-- ibug
       |-- lfpw
'''


def process(root_folder, folder_name, image_name, label_name, target_size):
    image_path = os.path.join(root_folder, folder_name, image_name)
    label_path = os.path.join(root_folder, folder_name, label_name)

    with open(label_path, 'r') as ff:
        scale = 1.2
        anno = ff.readlines()[3:-1]

        anno = [x.strip().split() for x in anno]
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]
        anno_y = [x[1] for x in anno]
        bbox_xmin, bbox_ymin = min(anno_x), min(anno_y)
        bbox_xmax, bbox_ymax = max(anno_x), max(anno_y)
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
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


def convert(data_dir, target=256):
    os.makedirs(os.path.join(data_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images/test'), exist_ok=True)

    folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
    annos_train = {}
    for folder_train in folders_train:
        all_files = sorted(os.listdir(os.path.join(data_dir, folder_train)))
        image_files = [x for x in all_files if '.pts' not in x]
        label_files = [x for x in all_files if '.pts' in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            print(image_name)
            image_crop, anno = process(os.path.join(data_dir), folder_train, image_name,
                                       label_name, target)
            image_crop_name = folder_train.replace('/', '_') + '_' + image_name
            cv2.imwrite(os.path.join(data_dir, 'images/train', image_crop_name), image_crop)
            annos_train[image_crop_name] = anno
    with open(os.path.join(data_dir, 'images', 'train.txt'), 'w') as f:
        for image_crop_name, anno in annos_train.items():
            f.write(image_crop_name + ' ')
            for x, y in anno:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
    annos_test = {}
    for folder_test in folders_test:
        all_files = sorted(os.listdir(os.path.join(data_dir, folder_test)))
        image_files = [x for x in all_files if '.pts' not in x]
        label_files = [x for x in all_files if '.pts' in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            print(image_name)
            image_crop, anno = process(data_dir, folder_test, image_name,
                                       label_name, target)
            image_crop_name = folder_test.replace('/', '_') + '_' + image_name
            cv2.imwrite(os.path.join(data_dir, 'images/test', image_crop_name), image_crop)
            annos_test[image_crop_name] = anno
    with open(os.path.join(data_dir, 'images', 'test.txt'), 'w') as f:
        for image_crop_name, anno in annos_test.items():
            f.write(image_crop_name + ' ')
            for x, y in anno:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    annos = None
    with open(os.path.join(data_dir, 'images', 'test.txt'), 'r') as f:
        annos = f.readlines()
    with open(os.path.join(data_dir, 'images', 'test_common.txt'), 'w') as f:
        for anno in annos:
            if not 'ibug' in anno:
                f.write(anno)
    with open(os.path.join(data_dir, 'images', 'test_challenge.txt'), 'w') as f:
        for anno in annos:
            if 'ibug' in anno:
                f.write(anno)

    with open(os.path.join(data_dir, 'images', 'train.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]
    lengths = [len(anno) for anno in annos]
    if len(set(lengths)) != 1:
        print("Inconsistent sublist lengths found:", set(lengths))
    new_annos = []
    for anno in annos:
        try:
            new_annos.append([float(x) for x in anno])
        except ValueError as e:
            print("Error converting to float:", e, "in annotation:", anno)
    # Convert to numpy array
    annos = np.array(new_annos, dtype=object)
    annos = [[float(x) for x in anno] for anno in annos]
    annos = np.array(annos)
    meanface = np.mean(annos, axis=0)
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]

    with open(os.path.join(data_dir, 'images', 'indices.txt'), 'w') as f:
        f.write(' '.join(meanface))


if __name__ == "__main__":
    convert(data_dir='./Dataset', target=256)
