# Face Mask Detection (CNN) — PyTorch + MobileNetV2

A practical face mask detection project built with **PyTorch** using **transfer learning (MobileNetV2)**. The repository includes:

- A training notebook that downloads and trains on a Kaggle face mask dataset
- A real-time webcam inference notebook using **OpenCV Haar Cascade** for face detection
- A saved model checkpoint (`.pth`) for quick experimentation

---

## Project Overview

This project classifies whether a detected face is:

- **With Mask**
- **Without Mask**

The training pipeline fine-tunes a lightweight MobileNetV2 classifier head, making it suitable for fast inference and easy deployment experiments.

---

## Repository Contents

- `face_mask_detection_cnn.ipynb`  
  Training workflow (dataset download → preprocessing → train/val/test split → model training → checkpoint saving)

- `haar_cascade.ipynb`  
  Real-time webcam demo using OpenCV’s Haar Cascade face detector + the trained PyTorch classifier

- `mask_detector_model.pth`  
  Saved model weights (PyTorch state_dict)

---

## Model & Approach

### Architecture
- **Backbone:** MobileNetV2 (pretrained)
- **Classifier head:**  
  - Dropout(0.2)  
  - Linear(1280 → 128) + ReLU  
  - Linear(128 → 2)

### Training Notes (from the notebook)
- **Framework:** PyTorch + Torchvision
- **Input size:** 224 × 224
- **Normalization:** ImageNet mean/std
- **Augmentations (train):** horizontal/vertical flips + color jitter
- **Optimizer:** Adam (classifier parameters only)
- **Loss:** CrossEntropyLoss
- **Early stopping:** basic patience-based stopping using validation loss
- **Checkpoint:** saves best model as `mask_detector_model.pth`

---

## Dataset

The training notebook downloads the dataset from Kaggle:

- `omkargurav/face-mask-dataset`

You will need Kaggle API credentials configured to download it successfully in Colab or locally.

---

## Getting Started

### Option A — Run in Google Colab
Open the notebooks directly in Colab:

- `face_mask_detection_cnn.ipynb` (training)
- `haar_cascade.ipynb` (webcam inference)

> If running the webcam notebook in Colab, note that webcam access is easiest locally. The notebook is primarily suited for local execution.

---

### Option B — Run Locally

#### 1) Create an environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

#### 2) Install dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pillow scikit-learn numpy
```

#### 3) Run inference (webcam demo)
Open and run:
- `haar_cascade.ipynb`

The script will:
1. Load MobileNetV2 and replace the classifier
2. Load `mask_detector_model.pth`
3. Start webcam capture
4. Detect faces using Haar Cascade
5. Classify each face as **Mask / No Mask**
6. Draw a bounding box + label in real time

Press **q** to quit the window.

---

## Output Labels

The dataset class mapping used in training is:

- `with_mask: 0`
- `without_mask: 1`

In the live demo notebook:
- `0 → Mask`
- `1 → No Mask`

---

## Notes & Limitations

- Haar Cascade is lightweight and fast, but not the most robust face detector (especially under extreme angles/lighting).
- Performance depends heavily on the dataset quality and how well it matches real-world webcam conditions.
- The provided `.pth` file is a model checkpoint; ensure your inference code matches the exact architecture used during training.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

**Samriddha**  
GitHub: `@Samriddhacoderho`
