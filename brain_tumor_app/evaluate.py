import sys
import os

# Auto-redirect to the virtual environment if not already running inside it
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python"))
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"🔄 Automatically switching to virtual environment at {venv_python}...")
    os.execv(venv_python, [venv_python] + sys.argv)

import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from model import InferenceModel

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Brain Tumor Detection Model")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--weights", type=str, default="best_model.pth", help="Path to the model weights")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Target mapping function: 0 for 'notumor' (index 2), 1 for others
    target_mapping = lambda t: torch.tensor([0.0]) if t == 2 else torch.tensor([1.0])
    
    test_dataset = ImageFolder(args.test_dir, transform=test_transform, target_transform=target_mapping)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize inference model
    # We use InferenceModel which already handles weight loading and device placement
    inference_wrapper = InferenceModel(weight_path=args.weights)
    
    if not inference_wrapper.has_weights:
        print(f"Error: Weights not found at {args.weights}. Please train the model first.")
        return
        
    model = inference_wrapper.model
    model.eval()
    
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            # Logits >= 0 is same as Sigmoid(Logits) >= 0.5
            preds = (outputs >= 0.0).float()
            correct_test += torch.sum(torch.eq(preds, labels)).item()
            total_test += labels.size(0)
            
    epoch_test_loss = test_loss / total_test
    epoch_test_acc = correct_test / total_test * 100
    
    print(f"Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%")

if __name__ == "__main__":
    main()
