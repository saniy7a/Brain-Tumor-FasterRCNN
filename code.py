# Brain Tumor Detection using Faster R-CNN (Training without weighs)

# ==========================================
#  Import Libraries
# ==========================================
import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from PIL import Image  # Import for converting images from NumPy arrays
import cv2  # Import for OpenCV image operations
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ==========================================
# Dataset Preparation
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms if transforms else Compose([Resize((224, 224)), ToTensor()])
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
        
        # Check if label file exists, skip if it doesn't
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found. Skipping this image.")
            return self.__getitem__((idx + 1) % len(self))
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(Image.fromarray(img))
        
        # Load Label
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                if width > 0 and height > 0: 
                    x_min = max(0, x_center - width / 2)
                    y_min = max(0, y_center - height / 2)
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if len(boxes) == 0:
            boxes = torch.zeros((1, 4), dtype=torch.float32) + 1e-6
            labels = torch.zeros((1,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return img, target

# Dataset Paths (Modify as per your Kaggle dataset structure)
train_images = '/content/drive/MyDrive/brain_tumor_dataset/train/images'
train_labels = '/content/drive/MyDrive/brain_tumor_dataset/train/labels'
valid_images = '/content/drive/MyDrive/brain_tumor_dataset/valid/images'
valid_labels = '/content/drive/MyDrive/brain_tumor_dataset/valid/labels'

# Datasets and Dataloaders
train_dataset = BrainTumorDataset(train_images, train_labels)
valid_dataset = BrainTumorDataset(valid_images, valid_labels)

# Collate function for DataLoader
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ==========================================
# Model Definition
# ==========================================
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a custom ResNet-50 backbone without pre-trained weights
backbone = resnet50(weights=None)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # Remove the classification head
backbone.out_channels = 2048  # Set the number of output channels from the backbone

# Define an anchor generator to match feature maps
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Define anchor sizes and aspect ratios for each feature map
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),  # Different anchor sizes for each feature map
    aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Same aspect ratios across all feature maps
)

# Create the Faster R-CNN model using the custom backbone
model = FasterRCNN(
    backbone,
    num_classes=2,  # Tumor or No Tumor
    rpn_anchor_generator=anchor_generator  # Pass the anchor generator
)

num_classes = 2
# Modify the box predictor for 2 classes (background + tumor)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Optimizer and Learning Rate Scheduler
optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # AdamW optimizer
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

# ==========================================
# Training Loop
# ==========================================
def train_model(model, train_loader, valid_loader, device, num_epochs=20):
    model.train()
    train_loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.float().to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        train_loss_history.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        lr_scheduler.step()
    
    print("Training complete!")
    return model, train_loss_history

# Clear GPU memory before training
import gc

del model
gc.collect()
torch.cuda.empty_cache()

# Train the model
model, train_loss_history = train_model(model, train_loader, valid_loader, device, num_epochs=10)

# ==========================================
# Evaluation and Metrics
# ==========================================
def evaluate_model(model, valid_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, targets in valid_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            
            for pred, target in zip(preds, targets):
                if len(pred['labels']) > 0:
                    y_true.append(int(target['labels'][0].item()))
                    y_pred.append(int(pred['labels'][0].item()))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    return cm

# Evaluate the model
conf_matrix = evaluate_model(model, valid_loader, device)

def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.savefig('/kaggle/working/training_loss.png')
    plt.show()

plot_loss(train_loss_history)

# Save Metrics to CSV
metrics_df = pd.DataFrame({'Epoch': list(range(1, len(train_loss_history) + 1)),
                           'Training Loss': train_loss_history})
metrics_df.to_csv('/kaggle/working/results.csv', index=False)

