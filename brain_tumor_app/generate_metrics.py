import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from model import InferenceModel

# Auto-redirect to the virtual environment if not already running inside it
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python"))
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"🔄 Automatically switching to virtual environment at {venv_python}...")
    os.execv(venv_python, [venv_python] + sys.argv)

from tqdm import tqdm

def main():
    # Use absolute paths to avoid issues with different execution contexts
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_dir = "/home/su1/Downloads/AA/Testing"
    weights_path = os.path.join(base_dir, "best_model.pth")
    output_dir = os.path.join(base_dir, "reports")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations (standard for EfficientNet/ViT)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Target mapping function: 0 for 'notumor' (index 2), 1 for others (glioma: 0, meningioma: 1, pituitary: 3)
    # Alphabetical order: glioma (0), meningioma (1), notumor (2), pituitary (3)
    target_mapping = lambda t: torch.tensor([0.0]) if t == 2 else torch.tensor([1.0])

    print(f"Loading dataset from {test_dir}...")
    test_dataset = ImageFolder(test_dir, transform=test_transform, target_transform=target_mapping)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    print(f"Loading weights from {weights_path}...")
    inference_wrapper = InferenceModel(weight_path=weights_path)
    if not inference_wrapper.has_weights:
        print(f"Error: Weights not found at {weights_path}")
        return
    
    model = inference_wrapper.model
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    print("Running inference on test set... This may take a few moments.")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            # sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (outputs >= 0.0).float().cpu().numpy().flatten()
            
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # ROC Curve
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")

    print("\nEvaluation Complete.")
    print(f"Accuracy: {np.mean(all_labels == all_preds) * 100:.2f}%")
    print(f"AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
