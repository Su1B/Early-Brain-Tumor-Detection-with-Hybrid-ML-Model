🧠 Early Brain Tumor Detection Using a Hybrid ML Model

An AI-powered brain tumor detection system that combines **EfficientNet-B0** and a **Vision Transformer (ViT)** to classify MRI brain scans as **healthy or tumor-positive.

The project integrates a deep learning pipeline with a **FastAPI web server** and an automated **PDF medical report generator**, providing an end-to-end demonstration of AI-assisted medical image screening.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical decision-making.

---

📌 Project Overview

Brain tumor detection from MRI scans is an important application of medical image analysis. Traditional image classification approaches often rely on CNN architectures that are highly effective at extracting local visual features but may have limitations when modeling broader relationships within an image.

This project implements a **hybrid CNN + Transformer architecture**:

MRI Scan → Image Preprocessing → EfficientNet-B0 → Transformer → Classification → PDF Report

* **EfficientNet-B0** extracts important visual features such as edges, textures, and structural patterns.
* **Transformer layers** process these extracted features to capture broader relationships within the MRI representation.
* A final classification layer predicts whether the scan is **healthy** or **tumor-positive**.
* The prediction can be presented through a **FastAPI web application**.
* A PDF report is automatically generated containing the scan, prediction, and relevant information.

---

✨ Key Features

* 🧠 Hybrid **EfficientNet-B0 + Transformer** architecture
* 🖼️ MRI image preprocessing and resizing to **224 × 224**
* 🔬 Binary classification:

  * Healthy
  * Tumor
* 🚀 Automatic **GPU detection** with CPU fallback
* 🔄 Data augmentation during training
* 💾 Automatic saving of the best-performing model
* 🌐 FastAPI-based web interface
* 📄 Automated PDF report generation
* 📊 Confusion Matrix generation
* 📈 ROC Curve generation
* 🧪 Automated model evaluation
* 🪟 Windows startup script
* 🐧 Linux startup script

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │   MRI Scan      │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Image Preprocessing │
                  │     224 × 224       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   EfficientNet-B0   │
                  │   Feature Extractor │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Transformer Layer   │
                  │ Global Relationships│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Classification Head │
                  └──────────┬──────────┘
                             │
                       ┌─────┴─────┐
                       ▼           ▼
                  ┌─────────┐ ┌─────────┐
                  │ Healthy │ │  Tumor  │
                  └─────────┘ └─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   PDF Report        │
                  │   Generation        │
                  └─────────────────────┘
```

---

## 📂 Project Structure

```text
HEADLOCK/
│
├── model.py                 # Hybrid AI model
├── train.py                 # Model training pipeline
├── evaluate.py              # Model evaluation
├── generate_metrics.py      # Metrics and visualization
├── train_accuracy.py        # Training accuracy analysis
├── app.py                   # FastAPI web server
├── pdf_gen.py               # PDF report generation
│
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── ...
│
├── README.md
├── Help.txt
├── requirements.txt
│
├── start.bat                # Windows startup script
├── start.sh                 # Linux startup script
│
└── .gitignore
```

---

## 🔬 Model Architecture

### EfficientNet-B0

The first stage uses **EfficientNet-B0** as the primary CNN feature extractor.

CNNs are particularly effective at identifying local visual patterns such as:

* Edges
* Textures
* Shapes
* Anatomical structures
* Local abnormalities

EfficientNet provides a strong balance between model performance and computational efficiency.

### Transformer Layer

The extracted feature representation is then passed to a Transformer-based component.

Transformers can model relationships between different feature regions and provide a mechanism for understanding broader structural patterns in the MRI representation.

### Classification

The combined representation is passed through the classification head to produce the final prediction:

```text
MRI Image
    ↓
EfficientNet-B0
    ↓
Feature Representation
    ↓
Transformer
    ↓
Classification Head
    ↓
Healthy / Tumor
```

---

## 🧪 Training Pipeline

The training pipeline is implemented in `train.py`.

The training process includes:

1. Loading the MRI dataset
2. Image preprocessing
3. Resizing images to `224 × 224`
4. Binary label mapping
5. Data augmentation
6. Model training
7. Validation
8. Monitoring performance
9. Saving the best-performing model

The system also automatically detects whether a compatible GPU is available.

```text
GPU Available → GPU Training
       │
       └── No GPU → CPU Training
```

---

## 📊 Model Evaluation

The project includes dedicated evaluation scripts for testing model performance on unseen data.

### `evaluate.py`

Used to evaluate the trained model and generate performance statistics.

### `generate_metrics.py`

Automatically generates evaluation visualizations such as:

* Confusion Matrix
* ROC Curve
* Classification metrics

### `train_accuracy.py`

Provides additional analysis of training performance across training runs.

These tools make it easier to evaluate model behavior rather than relying only on training accuracy.

---

## 🌐 Web Application

The project includes a **FastAPI** backend implemented in `app.py`.

The application workflow is:

```text
User uploads MRI scan
        ↓
FastAPI receives image
        ↓
Image preprocessing
        ↓
AI model prediction
        ↓
Prediction result
        ↓
PDF report generation
        ↓
Report returned to user
```

This provides a complete integration between the machine learning model and a usable application layer.

---

## 📄 Automated PDF Reports

`pdf_gen.py` is responsible for generating the prediction report.

The generated report can include:

* Uploaded MRI scan
* Prediction result
* Classification information
* Result tables
* Custom information based on the prediction

This demonstrates how an AI image classification system can be integrated into an automated reporting workflow.

---

## ⚙️ Installation

### Requirements

* Python 3.x
* PyTorch
* FastAPI
* Uvicorn
* OpenCV / PIL
* NumPy
* Matplotlib
* Scikit-learn
* ReportLab
* Other dependencies listed in `requirements.txt`

### Windows

Run:

```bash
start.bat
```

The startup script is designed to install the required dependencies and start the application.

### Linux

Make the script executable:

```bash
chmod +x start.sh
```

Then run:

```bash
./start.sh
```

---

## ▶️ Running Manually

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The application can then be accessed through the local server address displayed by FastAPI/Uvicorn.

---

## 🧪 Testing

The project includes multiple testing and evaluation utilities.

| Script                | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `evaluate.py`         | Evaluate model performance                             |
| `generate_metrics.py` | Generate evaluation metrics and plots                  |
| `train_accuracy.py`   | Analyze training performance                           |
| `app.py`              | Test end-to-end prediction through the web application |

### Evaluation Metrics

The project can generate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve

These metrics provide a more complete assessment of classification performance.

---

## 📋 Design & Implementation Status

### Design Document

The system design is fully implemented.

The architecture follows a modular pipeline:

```text
Preprocessing
      ↓
CNN Feature Extraction
      ↓
Transformer Processing
      ↓
Classification
      ↓
Web Integration
      ↓
Report Generation
```

### Module Integration

| Module                | Status     | Description                    |
| --------------------- | ---------- | ------------------------------ |
| `model.py`            | ✅ Complete | Hybrid CNN + Transformer model |
| `train.py`            | ✅ Complete | Training and data pipeline     |
| `evaluate.py`         | ✅ Complete | Model evaluation               |
| `generate_metrics.py` | ✅ Complete | Metrics and visualization      |
| `train_accuracy.py`   | ✅ Complete | Training analysis              |
| `app.py`              | ✅ Complete | FastAPI backend                |
| `pdf_gen.py`          | ✅ Complete | Automated PDF reports          |
| `start.bat`           | ✅ Complete | Windows setup                  |
| `start.sh`            | ✅ Complete | Linux setup                    |

---

## 💻 Technical Knowledge Demonstrated

This project combines concepts from several areas of computer science and artificial intelligence:

### Machine Learning

* Deep Learning
* Image Classification
* Transfer Learning
* CNN architectures
* Transformers
* Model Evaluation

### Computer Vision

* MRI image preprocessing
* Image resizing
* Data augmentation
* Feature extraction
* Medical image classification

### Software Engineering

* Modular project structure
* Backend API development
* Automated setup scripts
* Model/application integration
* Automated report generation

### Data Analysis

* Confusion Matrix
* ROC Curve
* Accuracy
* Precision
* Recall
* F1-score

---

## 🔮 Future Improvements

Potential future development includes:

* Training on larger and more diverse datasets
* Multi-class tumor classification
* Hyperparameter optimization
* Comparison with CNN-only architectures
* Comparison with Transformer-only architectures
* Explainable AI using techniques such as Grad-CAM
* Improved confidence calibration
* Model versioning
* Cloud deployment
* Secure user authentication
* Database integration
* More detailed clinical reporting

---

## 📝 Potential Research / Paper Topics

The project provides several possible directions for academic research.

### 1. Hybrid CNN-Transformer vs. CNN-Only Models

Compare the proposed EfficientNet + Transformer architecture against traditional CNN architectures to determine whether the hybrid approach provides improvements in classification performance.

### 2. Automated Medical Report Generation

Investigate the integration of AI-based image screening with automated report generation and evaluate its usefulness as a supporting tool for medical image analysis workflows.

### 3. Explainable Brain Tumor Detection

Extend the system with explainability techniques to visualize which regions of an MRI contributed most strongly to the model's prediction.

---

## 👥 Team & Project Organization

The project follows a modular structure that separates:

* Model development
* Training
* Evaluation
* Web application
* Report generation
* Testing

This separation makes the system easier to maintain, debug, test, and extend.

---

## ⚠️ Medical Disclaimer

This project is a **research and educational prototype**.

The predictions generated by this system should **not be considered a medical diagnosis** and should not be used to make clinical decisions.

A qualified healthcare professional and appropriate clinical investigation must be used for actual diagnosis.

---

## 📜 License

Add your preferred license here if you intend to distribute the project publicly.

---

## ⭐ Project Summary

**Early Brain Tumor Detection Using a Hybrid ML Model** demonstrates how modern deep learning techniques can be combined with conventional software engineering to create an end-to-end medical image classification prototype.

The core approach combines:

**EfficientNet-B0 + Transformer + FastAPI + Automated PDF Reporting**

with dedicated training, evaluation, and testing utilities.

The project serves as a foundation for further research into **hybrid deep learning architectures, medical image classification, explainable AI, and automated healthcare workflows.**
