#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 08:47:24 2025

@author: emreunsal

Class Weight Calculator
Author: Dr. Emre Ünsal

Mantık: 'Inverse Class Frequency' yöntemi kullanılır.
Az bulunan minerallerin ağırlığı yüksek, çok bulunanların (örn. Arka plan) düşük olur.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# ---------------------------------------------------------
# 1. AYARLAR
# ---------------------------------------------------------
MASK_DIR = "outputs/pseudomasks"
MINERAL_CLASSES = [
    "Carbonatization",
    "Chloritization",
    "Epidotization",
    "Sericitization",
    "Silicification"
]

IMG_SIZE = (224, 224) # Eğitimdeki boyutla AYNI olmalı

# Sınıf Haritası (0: Arka Plan, 1..5: Mineraller)
CLASS_MAP = {cls: i+1 for i, cls in enumerate(MINERAL_CLASSES)}
NUM_CLASSES = len(MINERAL_CLASSES) + 1

def calculate_weights():
    print(f"⚖️  Sınıf ağırlıkları hesaplanıyor: {MASK_DIR}")
    
    # Piksel sayaçlarını başlat (0: Background, 1: Epidot, vb.)
    class_pixel_counts = {i: 0 for i in range(NUM_CLASSES)}
    total_pixels_in_dataset = 0
    
    # Her sınıf klasörünü gez
    for cls_name in MINERAL_CLASSES:
        cls_id = CLASS_MAP[cls_name]
        cls_dir = os.path.join(MASK_DIR, cls_name)
        
        if not os.path.isdir(cls_dir):
            print(f"⚠️ Uyarı: Klasör bulunamadı - {cls_dir}")
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
        
        for f in tqdm(files, desc=f"Taranıyor: {cls_name}"):
            mask_path = os.path.join(cls_dir, f)
            
            # Maskeyi aç ve yeniden boyutlandır (Eğitimdeki gibi)
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(IMG_SIZE, Image.NEAREST) # Maskeler için Nearest şart!
            mask_np = np.array(mask)
            
            # --- KRİTİK NOKTA ---
            # Sizin veri seti yapınızda her klasörde binary maskeler var.
            # Beyaz kısımlar (>50) o minerale ait.
            
            mineral_pixels = np.sum(mask_np > 50)
            class_pixel_counts[cls_id] += mineral_pixels
            
            # Toplam piksel (Resimdeki her piksel)
            total_pixels_in_dataset += (IMG_SIZE[0] * IMG_SIZE[1])

    # Arka Plan (Background) Hesaplaması
    # Toplam Pikseller - Tüm Minerallerin Toplamı = Arka Plan
    # Not: Eğer mineraller çakışmıyorsa bu yöntem doğrudur.
    total_mineral_pixels = sum([class_pixel_counts[i] for i in range(1, NUM_CLASSES)])
    class_pixel_counts[0] = total_pixels_in_dataset - total_mineral_pixels
    
    print("\n" + "="*40)
    print("📊 PİKSEL SAYILARI (Pixel Counts)")
    print("="*40)
    for i in range(NUM_CLASSES):
        name = "Background" if i == 0 else MINERAL_CLASSES[i-1]
        count = class_pixel_counts[i]
        ratio = (count / total_pixels_in_dataset) * 100
        print(f"Class {i} ({name}): {count:,} px (~%{ratio:.2f})")

    # --- AĞIRLIK HESAPLAMA (Sklearn Tarzı Balanced Weight) ---
    # Formül: N_samples / (N_classes * N_samples_class)
    # Bu formül, her sınıfın etkisini eşitler.
    
    print("\n" + "="*40)
    print("⚖️  HESAPLANAN AĞIRLIKLAR (Weights)")
    print("="*40)
    
    weights = []
    valid_total_pixels = sum(class_pixel_counts.values()) # Kontrol için
    
    for i in range(NUM_CLASSES):
        count = class_pixel_counts[i]
        if count > 0:
            w = valid_total_pixels / (NUM_CLASSES * count)
        else:
            w = 0.0 # Hiç örneği olmayan sınıf (Hata önleyici)
        weights.append(w)
        
        name = "Background" if i == 0 else MINERAL_CLASSES[i-1]
        print(f"Class {i} ({name}): {w:.4f}")

    # Tensör formatında çıktı
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print("\n📋 Kodunuza yapıştıracağınız satır:")
    print(f"class_weights = {weights_tensor.tolist()}")
    print("criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))")

if __name__ == "__main__":
    calculate_weights()
