# PIPNet Facial Landmark Detection

### This repository contains the implementation of PIPNet, a robust approach for facial landmark detection using a deep learning model based on ResNet architectures.
[Click here to watch the video](https://www.youtube.com/watch?v=cxi1WQr-HKE)


### Key Achievements
#### Exceptional Model Performance on the 300W Dataset

> PIPNet model has achieved a significant milestone on the 300W dataset, one of the most challenging benchmarks in facial landmark detection. Successfully attained a minimum Normalized Mean Error (NME) of 2.6%, demonstrating the model's high accuracy and robustness in complex facial recognition tasks.

## Features
* #### Utilizes ResNet as the backbone for the PIPNet model.
* #### Supports training, testing, and real-time demo modes.
* #### Includes a 300W dataset loader and loss computation.
* #### Implements a face detector for real-time landmark detection in videos.
* #### Designed for easy customization and scalability to accommodate research and development needs.
          
### Requirements
```bash
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```           
## Usage
**Datasets: [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)**
* Download the datasets from official sources.
* Run below command:
```bash
$ python converter.py --data_dir './Dataset'
```

### Training
_**To train the model, run:**_
* Configure your dataset path in main.py for training

```bash
$ python main.py --train --input-size 256 --batch-size 16 --epochs 60
```
### Testing
_**For testing the model, use:**_
* Configure your dataset path in main.py for testing

```bash
$ python main.py --test
```

### Real-Time Demo
**_To run the real-time facial landmark detection:_**
```bash
$ python main.py --demo
```
### Results
| Backbone  | Epochs | Test NME |                                                                 Pretrained weights |
|:---------:|:------:|---------:|-----------------------------------------------------------------------------------:|
| ResNet18  |   60   |     3.39 |  [model](https://github.com/Shohruh72/PIPNet/releases/download/untagged-501559a3c4e3225a331b/last.pt) |
 

##### Reference
* https://github.com/jhb86253817/PIPNet
* https://github.com/jahongir7174/PIPNet
