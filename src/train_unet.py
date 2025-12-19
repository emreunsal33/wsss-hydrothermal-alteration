#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: emreunsal

Created on Fri Nov 21 2025
Subject: Ultimate U-Net Training (Adaptive Masks + Optimized Weights)
Features: Hybrid Loss, Cosine Annealing, Multi-Metric Plotting, ROC, Visualization
Author: Dr. Emre Ünsal
Affiliation: Sivas Cumhuriyet University, Software Engineering
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# Metrics
from sklearn.metrics import jaccard_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
IMAGE_DIR = "data/images"
MASK_DIR = "outputs/pseudomasks"
SAVE_PATH = "outputs/checkpoints"


if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

# Mineral Classes
MINERAL_DIRS = ['epidotlasma', 'karbonatlasma', 'kloritlesme', 'serizit', 'silislesme']
MINERAL_CLASSES_EN = [
    "Carbonatization",
    "Chloritization",
    "Epidotization",
    "Sericitization",
    "Silicification"]
CLASS_NAMES_EN = ['Background'] + MINERAL_CLASSES_EN

CLASS_MAP = {cls: i+1 for i, cls in enumerate(MINERAL_DIRS)}
NUM_CLASSES = len(MINERAL_DIRS) + 1 

BATCH_SIZE = 16 
EPOCHS = 25 
LR = 1e-4
IMG_SIZE = (224, 224)

# Normalization (Specific to your Thin Sections)
NORM_MEAN = [0.3838, 0.3899, 0.384]
NORM_STD  = [0.219, 0.206, 0.2028]

# --- GÜNCELLEME: YENİ HESAPLANAN V3 AĞIRLIKLARI ---
# [Background, Epidot, Carbonate, Chlorite, Sericite, Silica]
# Background 0.75, Diğerleri 1.0-1.2 civarı (Dengeli Dağılım)
CLASS_WEIGHTS = [0.7554, 1.0325, 1.0275, 1.2356, 1.0239, 1.0543]

CLASS_COLORS = ['black', 'red', 'lime', 'blue', 'yellow', 'magenta']
CUSTOM_CMAP = ListedColormap(CLASS_COLORS)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✅ Device: {device}")
print(f"✅ Mask Strategy: V3 Aggressive")
print(f"✅ Applied Weights: {CLASS_WEIGHTS}")

# ---------------------------------------------------------
# 2. METRIC HELPERS
# ---------------------------------------------------------
def calculate_batch_metrics(pred_tensor, target_tensor, n_classes):
    preds = pred_tensor.cpu().numpy().flatten()
    targets = target_tensor.cpu().numpy().flatten()
    # Zero division handling added
    iou = jaccard_score(targets, preds, average='macro', labels=list(range(n_classes)), zero_division=0)
    dice = f1_score(targets, preds, average='macro', labels=list(range(n_classes)), zero_division=0)
    return iou, dice

# ---------------------------------------------------------
# 3. MODEL (SimpleUNet)
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5); x = torch.cat([x4, x], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x3, x], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x2, x], dim=1); x = self.conv3(x)
        x = self.up4(x); x = torch.cat([x1, x], dim=1); x = self.conv4(x)
        return self.outc(x)

# ---------------------------------------------------------
# 4. HYBRID LOSS
# ---------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1] * self.n_classes
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = 2.0 * ((inputs[:, i] * target[:, i]).sum()) / ((inputs[:, i] + target[:, i]).sum() + 1e-8)
            loss += (1 - dice) * weight[i]
        return loss / self.n_classes

# ---------------------------------------------------------
# 5. DATASET
# ---------------------------------------------------------
class HydroSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, classes, augment=False):
        self.pairs = [] 
        self.augment = augment
        for cls in classes:
            cls_img_dir = os.path.join(image_dir, cls)
            cls_mask_dir = os.path.join(mask_dir, cls)
            cls_id = CLASS_MAP[cls]
            if not os.path.isdir(cls_img_dir): continue
            fnames = os.listdir(cls_img_dir)
            for f in fnames:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    mask_name = os.path.splitext(f)[0] + ".png"
                    mask_path = os.path.join(cls_mask_dir, mask_name)
                    if os.path.exists(mask_path):
                        self.pairs.append((os.path.join(cls_img_dir, f), mask_path, cls_id))
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, cls_id = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)
        
        if self.augment and random.random() > 0.5:
            img = self.color_jitter(img)

        img = np.array(img)
        mask = np.array(mask)
        mask_label = np.zeros_like(mask, dtype=np.int64)
        # V3 Maskeleri için Eşik Değeri
        mask_label[mask > 50] = cls_id 

        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)(img_tensor)
        mask_tensor = torch.from_numpy(mask_label).long()
        return img_tensor, mask_tensor

full_dataset = HydroSegDataset(IMAGE_DIR, MASK_DIR, MINERAL_DIRS, augment=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------
# 6. TRAINING LOOP
# ---------------------------------------------------------
model = SimpleUNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_dice = DiceLoss(n_classes=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print("\n🚀 Starting Training with V3 Aggressive Masks...")
history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'iou': [], 'val_iou': [], 'dice': [], 'val_dice': []}
best_val_iou = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    running_iou = 0.0
    running_dice = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss_ce = criterion_ce(outputs, masks)
        loss_dice = criterion_dice(outputs, masks, weight=CLASS_WEIGHTS)
        loss = 0.6 * loss_ce + 0.4 * loss_dice
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        batch_iou, batch_dice = calculate_batch_metrics(preds, masks, NUM_CLASSES)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += torch.numel(preds)
        
        running_loss += loss.item() * images.size(0)
        running_iou += batch_iou * images.size(0)
        running_dice += batch_dice * images.size(0)
        pbar.set_postfix({'Loss': loss.item(), 'IoU': batch_iou})
    
    scheduler.step()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_pixels / total_pixels
    epoch_iou = running_iou / len(train_dataset)
    epoch_dice = running_dice / len(train_dataset)
    
    model.eval()
    val_running_loss = 0.0
    val_correct_pixels = 0
    val_total_pixels = 0
    val_running_iou = 0.0
    val_running_dice = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks, weight=CLASS_WEIGHTS)
            loss = 0.6 * loss_ce + 0.4 * loss_dice
            
            val_running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct_pixels += (preds == masks).sum().item()
            val_total_pixels += torch.numel(preds)
            batch_iou, batch_dice = calculate_batch_metrics(preds, masks, NUM_CLASSES)
            val_running_iou += batch_iou * images.size(0)
            val_running_dice += batch_dice * images.size(0)
            
    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_correct_pixels / val_total_pixels
    val_iou = val_running_iou / len(val_dataset)
    val_dice = val_running_dice / len(val_dataset)
    
    print(f"Train - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")
    print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
    
    history['loss'].append(epoch_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(epoch_acc)
    history['val_acc'].append(val_acc)
    history['iou'].append(epoch_iou)
    history['val_iou'].append(val_iou)
    history['dice'].append(epoch_dice)
    history['val_dice'].append(val_dice)
    
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_unet_v3_ultimate.pth"))

# ---------------------------------------------------------
# 7. PLOTTING METRICS (2x2)
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='s')
plt.title('Hybrid Loss'); plt.legend(); plt.grid(True, alpha=0.5)

plt.subplot(2, 2, 2)
plt.plot(history['acc'], label='Train Acc', marker='o', color='purple')
plt.plot(history['val_acc'], label='Val Acc', marker='s', color='violet')
plt.title('Pixel Accuracy'); plt.legend(); plt.grid(True, alpha=0.5)

plt.subplot(2, 2, 3)
plt.plot(history['iou'], label='Train mIoU', marker='o', color='green')
plt.plot(history['val_iou'], label='Val mIoU', marker='s', color='lime')
plt.title('Mean IoU'); plt.legend(); plt.grid(True, alpha=0.5)

plt.subplot(2, 2, 4)
plt.plot(history['dice'], label='Train Dice', marker='o', color='blue')
plt.plot(history['val_dice'], label='Val Dice', marker='s', color='cyan')
plt.title('Dice Score (F1)'); plt.legend(); plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "v3_training_metrics.png"), dpi=300)
plt.show()

# ---------------------------------------------------------
# 8. ROC / AUC CALCULATION
# ---------------------------------------------------------
print("\nCalculating ROC/AUC...")
subset_size = max(1, int(len(val_dataset) * 0.1)) 
subset_indices = random.sample(range(len(val_dataset)), subset_size)
all_targets, all_probs = [], []

model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "best_unet_v3_ultimate.pth")))
model.eval()

with torch.no_grad():
    for idx in tqdm(subset_indices, desc="ROC Analysis"):
        img, mask = val_dataset[idx]
        input_img = img.unsqueeze(0).to(device)
        output = model(input_img)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
        flat_mask = mask.numpy().flatten()
        flat_probs = probs.reshape(NUM_CLASSES, -1).transpose()
        sample_rate = 20 
        all_targets.append(flat_mask[::sample_rate])
        all_probs.append(flat_probs[::sample_rate])

all_targets = np.concatenate(all_targets)
all_probs = np.concatenate(all_probs)
all_targets_bin = label_binarize(all_targets, classes=list(range(NUM_CLASSES)))

plt.figure(figsize=(10, 8))
fpr, tpr, roc_auc = {}, {}, {}
fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_bin.ravel(), all_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-Avg ROC (AUC={roc_auc["micro"]:.4f})', color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['black', 'aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(NUM_CLASSES), colors):
    fpr[i], tpr[i], _ = roc_curve(all_targets_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{CLASS_NAMES_EN[i]} (AUC={roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Pixel-wise Multi-Class ROC Curve (V3 Masks)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_PATH, "v3_roc_curve.png"))
plt.show()

# ---------------------------------------------------------
# 9. VISUAL COMPARISON
# ---------------------------------------------------------
print("\nGenerating Comparison Results...")
samples_to_show = {}
found_classes = set()
val_indices = list(range(len(val_dataset)))
random.shuffle(val_indices)

for idx in val_indices:
    _, mask_tensor = val_dataset[idx]
    unique_vals = torch.unique(mask_tensor)
    for val in unique_vals:
        v = val.item()
        if v != 0 and v not in found_classes:
            samples_to_show[v] = idx
            found_classes.add(v)
    if len(found_classes) == len(MINERAL_DIRS): break

sorted_classes = sorted(list(samples_to_show.keys()))
final_indices = [samples_to_show[k] for k in sorted_classes]

fig, axes = plt.subplots(len(final_indices), 3, figsize=(12, 3 * len(final_indices)))

for row, idx in enumerate(final_indices):
    img, mask = val_dataset[idx]
    cls_id = sorted_classes[row]
    cls_name_en = MINERAL_CLASSES_EN[cls_id - 1]
    
    with torch.no_grad():
        input_img = img.unsqueeze(0).to(device)
        output = model(input_img)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array(NORM_STD)) + np.array(NORM_MEAN)
    img_np = np.clip(img_np, 0, 1)
    
    axes[row, 0].imshow(img_np)
    axes[row, 0].set_ylabel(cls_name_en.upper(), fontsize=12, fontweight='bold', rotation=90, labelpad=10)
    if row==0: axes[row, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    
    axes[row, 1].imshow(mask, cmap=CUSTOM_CMAP, vmin=0, vmax=len(CLASS_COLORS)-1, interpolation='nearest')
    if row==0: axes[row, 1].set_title("V3 Aggressive Mask", fontsize=12, fontweight='bold')
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=len(CLASS_COLORS)-1, interpolation='nearest')
    if row==0: axes[row, 2].set_title("U-Net Prediction", fontsize=12, fontweight='bold')
    axes[row, 2].axis('off')

patches = [mpatches.Patch(color=CLASS_COLORS[i], label=f"{CLASS_NAMES_EN[i]}") for i in range(len(CLASS_COLORS))]
fig.legend(handles=patches, loc='lower center', ncol=len(CLASS_COLORS), fontsize=10, frameon=True)
plt.subplots_adjust(bottom=0.05 + (0.02 * len(final_indices)))
plt.savefig(os.path.join(SAVE_PATH, "v3_results_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Process Completed. Results saved to: {SAVE_PATH}")
