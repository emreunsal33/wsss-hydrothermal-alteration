#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 08:38:57 2025

@author: emreunsal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Veri Seti İstatistiklerini Hesaplama Modülü (Mean & Std)
Dr. Emre Ünsal - Hidrotermal Alterasyon Projesi
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# 1. AYARLAR (Sizin Path'leriniz)
# ---------------------------------------------------------
IMAGE_DIR = "/Users/emreunsal/Hidrothermal-Alteration/ince_kesit_big"
MINERAL_CLASSES = ['epidotlasma', 'karbonatlasma', 'kloritlesme', 'serizit', 'silislesme']

# Modelinize girecek boyutla aynı olmalı (Doğru istatistik için)
IMG_SIZE = (224, 224) 

def calculate_dataset_stats(image_dir, classes):
    print(f"📊 İstatistikler hesaplanıyor: {image_dir}")
    
    # Değişkenleri başlat
    pop_mean = np.zeros(3) # [R_mean, G_mean, B_mean]
    pop_std = np.zeros(3)
    pop_var = np.zeros(3)
    
    total_images = 0
    
    # Dosya listesini topla
    image_paths = []
    for cls in classes:
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, f))
    
    if len(image_paths) == 0:
        print("❌ Hiç görüntü bulunamadı! Path'leri kontrol edin.")
        return

    print(f"✅ Toplam {len(image_paths)} görüntü işlenecek...")

    # --- 1. Aşama: Ortalama (Mean) Hesaplama ---
    # Tüm pikselleri tek tek toplamak yerine, görüntü bazlı ortalamalar üzerinden gidiyoruz
    # (Büyük veri setleri için daha hızlı ve güvenli bir yaklaşımdır)
    
    pixel_sum = np.zeros(3) # R, G, B toplamı
    pixel_sq_sum = np.zeros(3) # Kareler toplamı (Std için gerekli)
    n_pixels = 0 # Toplam piksel sayısı (H * W * N_images)

    for path in tqdm(image_paths, desc="Hesaplanıyor"):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            img_np = np.array(img) / 255.0 # [0, 1] aralığına çek
            
            # Kanal bazında işlem (H, W, 3) -> (H*W, 3)
            img_np = img_np.reshape(-1, 3)
            
            # Kümülatif toplamlar
            pixel_sum += img_np.sum(axis=0)
            pixel_sq_sum += (img_np ** 2).sum(axis=0)
            
            n_pixels += img_np.shape[0]
            
        except Exception as e:
            print(f"Hata oluşan dosya: {path} | {e}")

    # --- 2. Aşama: Sonuçları Çıkar ---
    # Mean = Toplam / Adet
    final_mean = pixel_sum / n_pixels
    
    # Std = sqrt( (Toplam_Kare / Adet) - Mean^2 )
    final_var = (pixel_sq_sum / n_pixels) - (final_mean ** 2)
    final_std = np.sqrt(final_var)

    print("\n" + "="*40)
    print("🧪 HESAPLANAN DEĞERLER")
    print("="*40)
    print(f"Mean (R, G, B): {final_mean}")
    print(f"Std  (R, G, B): {final_std}")
    print("="*40)
    
    # Kopyalanabilir Format
    print("\n📋 Kodunuza yapıştıracağınız satır:")
    print(f"transforms.Normalize({list(np.round(final_mean, 4))}, {list(np.round(final_std, 4))})")

if __name__ == "__main__":
    calculate_dataset_stats(IMAGE_DIR, MINERAL_CLASSES)