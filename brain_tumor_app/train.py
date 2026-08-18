import sys
import os

# Auto-redirect to the virtual environment if not already running inside it
# This permanently prevents "ModuleNotFoundError" regardless of how the script is run
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python"))
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"🔄 Automatically switching to virtual environment at {venv_python}...")
    os.execv(venv_python, [venv_python] + sys.argv)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from tqdm import tqdm
from model import HybridBrainTumorModel

def get_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Brain Tumor Detection Model")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training dataset directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing/validation dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing best_model.pth")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    # The dataset contains folders like 'glioma', 'meningioma', 'notumor', 'pituitary'
    # We want binary classification: 0 for 'notumor', 1 for others ('glioma', 'meningioma', 'pituitary')
    # ImageFolder assigns indices alphabetically: glioma: 0, meningioma: 1, notumor: 2, pituitary: 3
    # Target mapping function
    target_mapping = lambda t: torch.tensor([0.0]) if t == 2 else torch.tensor([1.0])
    
    train_dataset = ImageFolder(args.train_dir, transform=train_transform, target_transform=target_mapping)
    test_dataset = ImageFolder(args.test_dir, transform=test_transform, target_transform=target_mapping)
    
    print(f"Training classes: {train_dataset.classes}")
    print(f"Total training images: {len(train_dataset)}")
    print(f"Total testing images: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    
    # Initialize Model
    model = HybridBrainTumorModel(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)
    
    # Loss and Optimizer
    # BCEWithLogitsLoss is more stable and compatible with AMP autocast
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Initialize scaler for Mixed Precision Training (speeds up processing / saves VRAM)
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Resume Logic
    if args.resume and os.path.exists(args.save_path):
        print(f"📂 Resuming training from {args.save_path}...")
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        # Optional: You could also load best_val_acc here if saved, but we'll restart metrics for the new session
    
    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use AMP Autocast if GPU is available to drastically speed up training
            if device.type == 'cuda' and scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Accuracy (Logits >= 0 corresponds to Probability >= 0.5)
            preds = (outputs >= 0.0).float()
            correct_train += torch.sum(torch.eq(preds, labels)).item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                # Accuracy (Logits >= 0 corresponds to Probability >= 0.5)
                preds = (outputs >= 0.0).float()
                correct_val += torch.sum(torch.eq(preds, labels)).item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val * 100
        
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Save best model
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            print(f"--> Validation metric improved. Saving model weights to {args.save_path}...")
            torch.save(model.state_dict(), args.save_path)
            
    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
