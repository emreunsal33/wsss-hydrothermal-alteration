#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 2025
Subject: Generate Adaptive PesudoMasks with DenseCRF (FIXED: Binary Feed Strategy)
Author: Dr. Emre Ünsal
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

# ---------------------------------------------------------
# 1. AYARLAR (AGRESİF MOD)
# ---------------------------------------------------------
MODEL_PATH = "outputs/checkpoints/best_classifier.pth"
DATASET_PATH = "data/images"
OUTPUT_MASK_DIR = "outputs/pseudomasks"

MINERAL_CLASSES = [
    "Carbonatization",
    "Chloritization",
    "Epidotization",
    "Sericitization",
    "Silicification"
]

IMG_SIZE = (224, 224)

# --- Adaptive Treshold ---
LOW_THRESHOLD = 0.15 

# SDevice
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✅ Cihaz: {device}")

# Klasör Kontrolü
if not os.path.exists(OUTPUT_MASK_DIR):
    os.makedirs(OUTPUT_MASK_DIR)
    for cls in MINERAL_CLASSES:
        os.makedirs(os.path.join(OUTPUT_MASK_DIR, cls), exist_ok=True)

# ---------------------------------------------------------
# 2. MODEL VE GRAD-CAM
# ---------------------------------------------------------
def build_model(num_classes):
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

print("⏳ Model yükleniyor...")
model = build_model(len(MINERAL_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None: class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        output[:, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Min-Max Normalizasyon
        heatmap_min = torch.min(heatmap)
        heatmap_max = torch.max(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            
        return heatmap.cpu().numpy()

target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# ---------------------------------------------------------
# 3. DENSE CRF (FIXED: BINARY FEED STRATEGY)
# ---------------------------------------------------------
def apply_dense_crf(img, heatmap):
    h, w = img.shape[:2]
    
    # --- DEĞİŞİKLİK BURADA ---
    # Olasılık vermiyoruz, doğrudan "Binary Maske" veriyoruz.
    # Böylece CRF "emin değilim" diyip silemiyor.
    
    initial_mask = np.zeros((h, w), dtype=np.int32)
    initial_mask[heatmap > LOW_THRESHOLD] = 1 # Foreground
    
    # Eğer maske tamamen boşsa CRF hatası almamak için erken dön
    if np.sum(initial_mask) == 0:
        return initial_mask
    
    # gt_prob=0.90 -> "Verdiğim bu maskeye %90 güven"
    U = unary_from_labels(initial_mask, 2, gt_prob=0.90, zero_unsure=False)
    
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    
    # Pairwise Potentials
    # 1. Smoothness (Mekansal tutarlılık)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # 2. Appearance (Renk uyumu - Sınırları kayaca yapıştır)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=img, compat=10,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(5)
    map_res = np.argmax(Q, axis=0).reshape((h, w))
    return map_res

# ---------------------------------------------------------
# 4. İŞLEM DÖNGÜSÜ
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"\n🚀 Maske Üretimi Başlıyor (V3 Aggressive FIXED)...")
print(f"Eşik Değeri: {LOW_THRESHOLD}")

stats = {"success": 0, "fallback": 0, "empty": 0}

for class_name in MINERAL_CLASSES:
    class_dir = os.path.join(DATASET_PATH, class_name)
    save_dir = os.path.join(OUTPUT_MASK_DIR, class_name)
    
    if not os.path.isdir(class_dir): continue
    
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📂 İşleniyor: {class_name} ({len(files)} resim)")
    
    for f in tqdm(files):
        try:
            img_path = os.path.join(class_dir, f)
            
            # Yükle
            input_image = Image.open(img_path).convert('RGB')
            input_tensor = transform(input_image).unsqueeze(0).to(device)
            # CRF için orijinal numpy hali
            img_np = np.array(input_image.resize(IMG_SIZE))
            
            # 1. Grad-CAM Üret
            heatmap = grad_cam(input_tensor) # Zaten normalize dönüyor
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            
            # 2. DenseCRF Uygula (Yeni Fonksiyon)
            mask = apply_dense_crf(img_np, heatmap)
            
            # 3. İstatistik
            pixel_ratio = np.sum(mask) / (IMG_SIZE[0] * IMG_SIZE[1])
            
            if pixel_ratio < 0.001: # Çok çok boşsa
                # Yine de Fallback yapalım, veri kaybı olmasın
                mask = np.where(heatmap > LOW_THRESHOLD, 1, 0).astype(np.uint8)
                if np.sum(mask) > 0:
                    stats["fallback"] += 1
                else:
                    stats["empty"] += 1
            else:
                stats["success"] += 1

            # 4. Kaydet
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            save_name = os.path.splitext(f)[0] + ".png"
            mask_img.save(os.path.join(save_dir, save_name))
            
        except Exception as e:
            print(f"Hata ({f}): {e}")

print("\n" + "="*30)
print("🏁 İŞLEM RAPORU")
print("="*30)
print(f"DenseCRF Başarılı : {stats['success']}")
print(f"Fallback (Yedek)  : {stats['fallback']}")
print(f"Boş Maske         : {stats['empty']}")
print(f"Kayıt Yeri        : {OUTPUT_MASK_DIR}")
