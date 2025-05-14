# 🧠 Face Landmark Auto-Labeling Toolkit

**_A high-performance desktop application for automated facial landmark detection, annotation, and dataset generation—powered by deep learning and built for an intuitive user experience._**

![Vizualization](https://github.com/Shohruh72/Landmark_Auto_Label/releases/download/v1.0.0/short.gif)
![Vizualization](https://github.com/Shohruh72/Landmark_Auto_Label/releases/download/v1.0.0/full.gif)

* [Click here to watch the video](https://youtu.be/T9NZXJYQGjo?si=QEl97S-sDrfmWsZl)

* [Click here to watch the video](https://youtu.be/NugGGDl6rm0?si=DyUCIjKBTgsKXGZy)
---

~~## 🚀 Overview~~

The **Face Landmark Auto-Labeling Toolkit** t simplifies facial landmark annotation with ONNX-based detection and a PyTorch (PIPNet) regressor. It features a modern GUI for efficient navigation, editing, and export—ideal for dataset creation, research, and production workflows.

---

## 🖼️ Features

- **Automatic Landmark Detection**  
  Accurate facial landmark prediction using deep learning.

- **Interactive GUI (Tkinter + TTKBootstrap)**  
  Clean, themeable interface with image browsing and editing.
- **Smart Edit Mode**  
  Easily adjust points with drag-and-drop functionality.
- **Batch Processing**  
  Label entire folders in a single operation.
- **Export Formats**  
 Outputs .json and .pts files with original images.
- **Real-Time Stats Panel**  
  Real-time tracking of image, face, and landmark counts.


## 🛠️ Setup & Installation

1. **Clone this repo:**

   ```bash
   git clone https://github.com/yourusername/face-landmark-autolabeler.git
   cd face-landmark-autolabeler
   ```
2. **Create Environment**
    ```bash
    cd seatbelt
    conda env create -f environment.yaml
    conda activate seatbelt
    ```
### Download Face Landmark Weight
| Backbone  | Epochs | Test NME |                                                                 Pretrained weights |
|:---------:|:------:|---------:|-----------------------------------------------------------------------------------:|
| ResNet18  |   120  |     3.37 |  [model](https://github.com/Shohruh72/PIPNet/releases/download/v1.0.0/best.pt) |
| ResNet50  |   120  |     3.23 |  [model](https://github.com/Shohruh72/PIPNet/releases/download/v1.0.0/best_50.pt) |
| ResNet101 |   120  |     3.17 |  [model](https://github.com/Shohruh72/PIPNet/releases/download/v1.0.0/best_101.pt) |
 
3. **Run The Application**
    ```bash
    python auto_label_gui.py
   ```
### Reference
https://github.com/Shohruh72
https://github.com/Shohruh72/PIPNet
