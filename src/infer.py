#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 2025
Subject: Batch Inference (English Output) - Larger Fonts & Closer Legend
Author: Dr. Emre Ünsal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

# ---------------------------------------------------------
# 1. SETTINGS & PATHS
# ---------------------------------------------------------
MODEL_PATH = "/Users/emreunsal/Hidrothermal-Alteration/Results_PyTorch_V3_Ultimate_Fixed/best_unet_v3_ultimate.pth"
INPUT_DIR = "/Users/emreunsal/Hidrothermal-Alteration/Results_PyTorch_V3_Ultimate_Fixed/Test_New"
OUTPUT_DIR = "/Users/emreunsal/Hidrothermal-Alteration/Results_PyTorch_V3_Ultimate_Fixed/Test_New_results_english"

IMG_SIZE = (224, 224)
OVERLAY_ALPHA = 0.35  # Slightly more visible overlay

# Normalization
NORM_MEAN = [0.3838, 0.3899, 0.384]
NORM_STD  = [0.219, 0.206, 0.2028]

# Classes
MINERAL_CLASSES = ['Epidotization', 'Carbonatization', 'Chloritization', 'Sericitization', 'Silicification']
CLASS_NAMES = ['Background'] + MINERAL_CLASSES
CLASS_COLORS = ['black', 'red', 'lime', 'blue', 'yellow', 'magenta']
CMAP = ListedColormap(CLASS_COLORS)

# Fonts
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 15

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✅ Device: {device}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------------------------------------------------
# 2. MODEL ARCHITECTURE
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
        super().__init__()
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

try:
    model = SimpleUNet(n_channels=3, n_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except FileNotFoundError:
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    exit()

# ---------------------------------------------------------
# 3. PROCESSING
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def process_batch():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
    if not files:
        print("⚠️ No images found.")
        return

    print(f"🚀 Processing {len(files)} images...")
    for filename in tqdm(files):
        analyze_and_save(os.path.join(INPUT_DIR, filename), filename)
    print(f"\n✅ Done! Results: {OUTPUT_DIR}")

def analyze_and_save(path, filename):
    try:
        original_img = Image.open(path).convert("RGB")
    except: return

    # Inference
    display_img = original_img.resize(IMG_SIZE, Image.BILINEAR)
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        confidence = torch.max(probs, dim=1)[0].squeeze().cpu().numpy()

    # Statistics
    total = pred_mask.size
    unique, counts = np.unique(pred_mask, return_counts=True)
    stats = dict(zip(unique, counts))
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(22, 6)) # Wider figure
    
    # 1. Original
    axes[0].imshow(np.array(display_img))
    axes[0].set_title("Original Image", fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    axes[0].axis("off")
    
    # 2. Overlay
    axes[1].imshow(np.array(display_img))
    axes[1].imshow(pred_mask, cmap=CMAP, vmin=0, vmax=len(CLASS_COLORS)-1, alpha=OVERLAY_ALPHA, interpolation='nearest')
    axes[1].set_title("Prediction Overlay", fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    axes[1].axis("off")
    
    # 3. Mask
    axes[2].imshow(pred_mask, cmap=CMAP, vmin=0, vmax=len(CLASS_COLORS)-1, interpolation='nearest')
    axes[2].set_title("Segmentation Map", fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    axes[2].axis("off")
    
    # 4. Confidence
    im = axes[3].imshow(confidence, cmap='magma', vmin=0, vmax=1)
    axes[3].set_title("Confidence Map", fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    axes[3].axis("off")
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    # LEGEND (Closer & Bigger)
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        pct = (stats.get(i, 0) / total) * 100
        if pct > 0.0:
            patches.append(mpatches.Patch(color=CLASS_COLORS[i], label=f"{name} ({pct:.1f}%)"))
            
    # bbox_to_anchor ile lejantı yukarı (grafiklerin hemen altına) çekiyoruz
    fig.legend(handles=patches, loc='lower center', 
               ncol=min(len(patches)+1, 6), 
               fontsize=LEGEND_FONT_SIZE,
               bbox_to_anchor=(0.5, 0.08),  # 0.08 = Sayfanın altından %8 yukarıda
               frameon=False)               # Çerçevesiz daha temiz görünür
    
    # Alt boşluğu ayarlayarak lejantın grafiğe yakın olmasını sağla
    plt.subplots_adjust(bottom=0.22, wspace=0.15) 
    
    save_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + "_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    process_batch()