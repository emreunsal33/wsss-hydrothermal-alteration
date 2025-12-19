import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from itertools import cycle

# Sklearn Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# PyTorch Kütüphaneleri
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ---------------------------------------------------------
# 1. AYARLAR VE DONANIM SEÇİMİ (M3 MAX - MPS)
# ---------------------------------------------------------
DATASET_PATH = "/Users/emreunsal/Hidrothermal-Alteration/ince_kesit_big"
SAVE_PATH = "/Users/emreunsal/Hidrothermal-Alteration/Results_PyTorch"
MINERAL_CLASSES = ['epidotlasma', 'karbonatlasma', 'kloritlesme', 'serizit', 'silislesme']

# Hiperparametreler
BATCH_SIZE = 32         
EPOCHS = 10             # Kullanıcı isteği: 10 Epoch
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)   

# Early Stopping Ayarları
EARLY_STOPPING_PATIENCE = 3 # 3 Epoch boyunca loss düşmezse dur

# Klasör kontrolü
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Klasör oluşturuldu: {SAVE_PATH}")

# Cihaz Seçimi
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"✅ Çalışma Cihazı: {device} (Apple Silicon Hızlandırma)")

# ---------------------------------------------------------
# 2. VERİ HAZIRLIĞI VE BÖLÜMLEME
# ---------------------------------------------------------
print("Veri taranıyor ve bölünüyor...")

all_image_paths = []
all_labels = []

for idx, class_name in enumerate(MINERAL_CLASSES):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_dir):
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            all_image_paths.append(os.path.join(class_dir, f))
            all_labels.append(idx)

if len(all_image_paths) == 0:
    raise ValueError("❌ Hata: Resim bulunamadı. DATASET_PATH yolunu kontrol edin.")

# Stratified Split: Train (%70), Val (%10), Test (%20)
X_train, X_temp, y_train, y_temp = train_test_split(
    all_image_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.66, stratify=y_temp, random_state=42 
)

print(f"Eğitim Seti: {len(X_train)} adet")
print(f"Doğrulama Seti: {len(X_val)} adet")
print(f"Test Seti: {len(X_test)} adet")

# ---------------------------------------------------------
# 3. DATASET VE DATALOADER
# ---------------------------------------------------------
class HydrothermalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Hata: {img_path} okunamadı. {e}")
            return torch.zeros((3, 224, 224)), label

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms['test'] = data_transforms['val']

datasets = {
    'train': HydrothermalDataset(X_train, y_train, transform=data_transforms['train']),
    'val': HydrothermalDataset(X_val, y_val, transform=data_transforms['val']),
    'test': HydrothermalDataset(X_test, y_test, transform=data_transforms['test'])
}

# Spyder/macOS uyumluluğu için num_workers=0, pin_memory=False
dataloaders = {
    x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), 
                  num_workers=0, pin_memory=False)
    for x in ['train', 'val', 'test']
}

# ---------------------------------------------------------
# 4. MODEL MİMARİSİ (EFFICIENTNET-B4)
# ---------------------------------------------------------
def build_model(num_classes):
    print("EfficientNet-B4 Modeli yükleniyor...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

model = build_model(len(MINERAL_CLASSES))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Dinamik LR: verbose kaldırıldı, patience düşürüldü
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# ---------------------------------------------------------
# 5. EĞİTİM DÖNGÜSÜ (EARLY STOPPING & MPS FIX)
# ---------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early Stopping Değişkenleri
    min_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # MPS uyumluluğu için .float() kullanımı
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['accuracy'].append(epoch_acc.item())
                history['loss'].append(epoch_loss)
            else:
                history['val_accuracy'].append(epoch_acc.item())
                history['val_loss'].append(epoch_loss)
                
                # Scheduler Step
                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  -> Current LR: {current_lr}")

                # En iyi Modeli Kaydet
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_path = os.path.join(SAVE_PATH, "best_model_efficientnet.pth")
                    torch.save(model.state_dict(), save_path)
                
                # Early Stopping Kontrolü
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    early_stop_counter = 0 
                else:
                    early_stop_counter += 1 
                    print(f"  -> Early Stopping Sayacı: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")
        
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early Stopping Tetiklendi! {epoch+1}. Epoch'ta eğitim durduruluyor.")
            break

    time_elapsed = time.time() - since
    print(f'Eğitim tamamlandı: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'En yüksek Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ---------------------------------------------------------
# 6. BAŞLATMA, TEST VE GÖRSELLEŞTİRME (AUC EKLENDİ)
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1. Eğitimi Başlat
    start_time = time.time()
    model, history = train_model(model, dataloaders, criterion, optimizer, EPOCHS)

    # 2. Test Tahminleri ve Olasılıklar
    print("\nTest seti üzerinde tahminler ve olasılıklar alınıyor...")
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = [] # AUC için olasılıkları tutacağız

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Sınıf Tahmini
            _, preds = torch.max(outputs, 1)
            
            # Olasılık Değerleri (Softmax)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 3. Confusion Matrix & Report
    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=MINERAL_CLASSES))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=MINERAL_CLASSES, yticklabels=MINERAL_CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_PATH, "confusion_matrix.png"), dpi=300)
    plt.show()

    # 4. Loss & Accuracy Grafikleri
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "loss_graph.png"), dpi=300)
    plt.show()

    plt.figure()
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "accuracy_graph.png"), dpi=300)
    plt.show()

    # 5. Multi-Class ROC / AUC Eğrisi
    print("AUC eğrileri çiziliyor...")
    
    # Etiketleri binarize et
    y_test_bin = label_binarize(y_true, classes=list(range(len(MINERAL_CLASSES))))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Her sınıf için ayrı AUC hesapla
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Mikro-ortalama (Micro-average) ROC hesapla
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # AUC Çizimi
    plt.figure(figsize=(10, 8))
    lw = 2
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.4f})'.format(MINERAL_CLASSES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve (EfficientNet-B4)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(SAVE_PATH, "auc_roc_curve.png"), dpi=300)
    plt.show()

    print(f"Tüm analizler tamamlandı. Sonuçlar: {SAVE_PATH}")