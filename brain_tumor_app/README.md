# 🧠 NeuroDiagnostics: Hybrid AI Brain Tumor Detection

**NeuroDiagnostics** is a professional, hospital-grade web application designed for the early detection of brain tumors using state-of-the-art Deep Learning. By combining the efficiency of **EfficientNet-B0** with the contextual awareness of **Vision Transformers (ViT)**, the system provides high-accuracy diagnostic assistance to clinical professionals.

---

## 🚀 Key Features

- **Hybrid AI Architecture**: Leverages `EfficientNet-B0` for spatial feature extraction and a `Vision Transformer` encoder for global context analysis.
- **FastAPI Backend**: A high-performance, asynchronous Python backend for rapid image processing and inference.
- **Clinical PDF Reporting**: Automatically generates downloadable medical reports with patient details, scan visuals, and clinical recommendations.
- **Modern UI/UX**: Professional hospital-style interface for MRI scan uploads and real-time result visualization.
- **Performance Optimized**: Supports Mixed Precision Training (AMP) for faster execution on modern hardware.

---

## 🏗️ Architecture Deep Dive

The core of NeuroDiagnostics is its **Hybrid Model Architecture**, which overcomes the limitations of traditional CNN-only models.

### 1. Feature Extraction (CNN)
We use **EfficientNet-B0** (pre-trained on ImageNet) to extract high-resolution spatial features from the input MRI scan (224x224). The final classifier is removed, leaving a 1280-dimensional feature vector.

### 2. Contextual Modeling (Transformer)
The 1280-dimensional vector is treated as a single token sequence and passed through a **Vision Transformer (ViT)** encoder (8 heads, 2 layers). This allows the model to capture long-range dependencies and subtle global patterns that local CNN filters might miss.

### 3. Classification Head
A fully connected sequence with ReLU activation and Dropout (0.3) processes the transformer output to produce a single logit representing the probability of a tumor.

---

## 📂 Project Structure

```text
brain_tumor_app/
├── app.py              # FastAPI Web Server & API
├── model.py            # Hybrid Model Definition (ViT + EfficientNet)
├── train.py            # Training Pipeline with Mixed Precision
├── evaluate.py         # Model Evaluation Script
├── generate_metrics.py # Confusion Matrix & ROC Curve Generator
├── pdf_gen.py          # Clinical Report Generator (ReportLab)
├── static/             # Frontend Assets (CSS, JS)
├── templates/          # Jinja2 HTML Templates
├── uploads/            # Temporary storage for uploaded scans
└── reports/            # Generated Clinical PDF Reports
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Linux (Optimized for Arch Linux)
- NVIDIA GPU (Optional, for CUDA acceleration)

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd brain_tumor_app
   ```

2. **Run the setup script**:
   This script creates a virtual environment, installs dependencies, and starts the server.
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:5050`.

---

## 📊 Training & Evaluation

To retrain the model with your own dataset:

```bash
python train.py --train_dir /path/to/train --test_dir /path/to/test --epochs 50 --batch_size 16
```

### Metrics Generation
After training, generate the ROC curve and Confusion Matrix:
```bash
python generate_metrics.py
```

---

## 📜 Disclaimer

*This application is a proof-of-concept AI diagnostic tool. It is NOT a substitute for professional medical diagnosis. All AI-generated results should be verified by a licensed radiologist or neurologist.*

---

**Developed for Advanced Medical AI Research.**

***#Find out the accuracy:::***
***python3 /home/su1/Downloads/AA/brain_tumor_app/train_accuracy.py***
