# Giraffe & Horse Tracking & Classification  
End-to-end pipeline that trains an EfficientNet-B0 classifier, extracts a static background, tracks every moving blob, classifies every track as *giraffe*, *horse* or *noise*, and exports an annotated video.

A second training stage (fine‑tuning) specifically improves the noise class by re‑training on real false‑positive crops collected after the first pass.

---

## 1 · Key Features
| Stage | What Happens | Relevant Script |
|-------|--------------|-----------------|
| **Training** | Supervised training from scratch on a three-class image folder. | `train.py` |
| **Background extraction** | Builds a clean static background (median over frames) from the raw video. | `back.py` |
| **Tracking + classification** | Background subtraction• Centroid tracker• EfficientNet inference every 1‑2 s. Brightness/contrast boost & label overlay. Stores false‑positives to misclassified/noise/ | `tracking.py`  |
| **Fine-tuning for noise** | Adds the collected misclassified/noise/ crops to the noise folder and runs a short additional training to sharpen the decision boundary. | `train_finetine.py`  |
| **Final inference** | Re‑run tracking.py with the fine‑tuned weights → output.mp4. | `tracking.py` |

---

## 2 · Quick Start

** 1. Set up Python ≥3.9 and install core deps ** 
python -m venv venv && source venv/bin/activate
pip install torch torchvision opencv-python numpy

** 2.  Organise data ** 
dataset/
├── giraffe/   *.jpg
├── horse/     *.jpg
└── noise/     *.jpg

** 3.  Train baseline model** 
python train.py \
    --data_dir dataset \
    --output_model_path model_efficientnet_updated.pth \
    --num_epochs 15 --batch_size 32 --lr 1e-3
    
**  Extract static background ** 
python back.py

** Run tracking/classification & harvest misclassifications ** 
python tracking.py
    
** (Optional) fine-tune  mainly on the new noise images ** 
python train_finetine.py \
    --data_dir finetune_dataset \
    --pretrained_model_path model_efficientnet_updated.pth

** Produce final annotated video ** 
python tracking.py \
    --model_path model_ft_noise.pth \
    --output_video output.mp4


