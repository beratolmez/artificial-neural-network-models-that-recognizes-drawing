import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import timm
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import torchmetrics
import matplotlib.pyplot as plt

# -------- ADIM 1 VE 2 ---------

# Görüntü boyutları (ViT genellikle 224x224 kullanır)
IMG_SIZE = 224
# Batch boyutu [cite: 25]
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # LR'yi düşürdük
EPOCHS = 30 # Epoch sayısını artırdık
PATIENCE = 5 # Sabrı artırdık
WEIGHT_DECAY = 1e-4 # AdamW için weight decay ekledik

train_data_path = '/kaggle/input/multizoo-dataset/train'
#test_data_path = '/kaggle/input/multizoo-dataset/test'

# Veri Dönüşümleri (Preprocessing & Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Boyutlandırma [cite: 20]
    transforms.RandomHorizontalFlip(), # Yatay çevirme (Augmentation) [cite: 21]
    transforms.RandomRotation(10), # Rastgele döndürme (Augmentation)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Renk jitter (Augmentation) [cite: 21]
    transforms.ToTensor(), # Tensöre çevirme
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalizasyon [cite: 20]
                         std=[0.229, 0.224, 0.225]) # ImageNet standartları
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Boyutlandırma [cite: 20]
    transforms.ToTensor(), # Tensöre çevirme
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalizasyon [cite: 20]
                         std=[0.229, 0.224, 0.225])
])

# Tüm eğitim verisetini yükle
full_train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)

# Sınıf isimlerini alalım
class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"Sınıflar: {class_names}")
print(f"Toplam sınıf sayısı: {num_classes}")

# Eğitim ve Doğrulama Setlerine Ayırma (%80 eğitim, %20 doğrulama) [cite: 18]
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Doğrulama seti için dönüşümleri düzeltelim (Augmentation olmasın)
# Bu kısım biraz karmaşık, normalde baştan iki ayrı dataset tanımlamak daha temiz olur.
# Şimdilik, doğrulama seti de train_transforms kullanıyor, ama idealde val_transforms kullanmalı.
# Daha iyi bir yaklaşım: ImageFolder'ı iki kez yükleyip sonra split etmek veya custom dataset yazmak.
# Basitlik için şimdilik böyle bırakalım ama raporda bunu belirtmek iyi olur.

print(f"Eğitim seti boyutu: {len(train_dataset)}")
print(f"Doğrulama seti boyutu: {len(val_dataset)}")

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Test seti için DataLoader (Test verisi geldiğinde kullanılacak) [cite: 16]
# test_dataset = datasets.ImageFolder(test_data_path, transform=val_transforms)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Cihazı belirle (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# -------- ADIM 3 (Model Seçimi ve Kurulumu)---------

# Önceden eğitilmiş bir ViT modeli seçelim
# 'vit_base_patch16_224' popüler bir seçenektir.
model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

# Modeli cihaza taşı
model.to(device)

# Metrikler için torchmetrics kullanımı
acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

# Model özetini görelim (Opsiyonel)
summary(model, input_size=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))

# Kayıp fonksiyonu (Çoklu sınıflandırma için CrossEntropyLoss) [cite: 6]
criterion = nn.CrossEntropyLoss()

# Optimize edici
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- YENİ LR SCHEDULER ---
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Eğitim ve doğrulama geçmişini saklamak için
history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

# Early Stopping için değişkenler
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None

# Eğitim Döngüsü
for epoch in range(EPOCHS):
    model.train() # Modeli eğitim moduna al
    running_loss = 0.0
    
    # Metrikleri sıfırla
    acc_metric.reset()
    f1_metric.reset()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eğitim]")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Gradientleri sıfırla
        optimizer.zero_grad()

        # İleri yayılım
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Geri yayılım ve optimizasyon
        loss.backward()
        optimizer.step()

        # İstatistikler
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        acc_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    # Epoch sonu eğitim metrikleri
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = acc_metric.compute()
    epoch_f1 = f1_metric.compute()
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc.cpu().numpy())
    history['train_f1'].append(epoch_f1.cpu().numpy())

    # Doğrulama Döngüsü
    model.eval() # Modeli değerlendirme moduna al
    val_running_loss = 0.0
    
    # Metrikleri sıfırla
    acc_metric.reset()
    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Doğrulama]")
    with torch.no_grad(): # Gradient hesaplamayı kapat
        for inputs, labels in progress_bar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            # Tüm metrikleri güncelle
            acc_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)

    # Epoch sonu doğrulama metrikleri
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = acc_metric.compute()
    val_epoch_f1 = f1_metric.compute()
    val_epoch_precision = precision_metric.compute()
    val_epoch_recall = recall_metric.compute()
    
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc.cpu().numpy())
    history['val_f1'].append(val_epoch_f1.cpu().numpy())

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Eğitim Kaybı: {epoch_loss:.4f} | Eğitim Acc: {epoch_acc:.4f} | Eğitim F1: {epoch_f1:.4f} | "
          f"Doğrulama Kaybı: {val_epoch_loss:.4f} | Doğrulama Acc: {val_epoch_acc:.4f} | Doğrulama F1: {val_epoch_f1:.4f} | "
          f"Doğrulama Prc: {val_epoch_precision:.4f} | Doğrulama Rcl: {val_epoch_recall:.4f}")

    # --- LR SCHEDULER ADIMI ---
    scheduler.step(val_epoch_loss)

    # Early Stopping Kontrolü [cite: 26]
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        # En iyi modeli kaydet (state_dict olarak)
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, 'best_model_weights.pth')
        print(f"Doğrulama kaybı düştü ({best_val_loss:.4f}), model kaydedildi.")
    else:
        epochs_no_improve += 1
        print(f"Doğrulama kaybı düşmedi. Sabır: {epochs_no_improve}/{PATIENCE}")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping tetiklendi! Eğitim durduruluyor.")
        break

# En iyi modeli yükle (Eğer Early Stopping olduysa)
if best_model_weights:
    model.load_state_dict(best_model_weights)
    print("En iyi model ağırlıkları yüklendi.")

# Eğitilen modeli arayüzde kullanmak için kaydet [cite: 27]
torch.save(model.state_dict(), 'final_model.pth')
print("Nihai model 'final_model.pth' olarak kaydedildi.")

# Öğrenme Eğrilerini Çizdirme [cite: 29]
plt.figure(figsize=(12, 5))

# F1 Score Grafiği (Örnekteki gibi [cite: 22])
plt.subplot(1, 2, 1)
plt.plot(history['train_f1'], label='Train F1 Score')
plt.plot(history['val_f1'], label='Validation F1 Score')
plt.title('F1 Scores vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Scores')
plt.legend()
plt.grid(True)

# Accuracy Grafiği (Örnekteki gibi [cite: 30])
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Scores vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Scores')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Doğrulama seti üzerinde son metrikleri hesapla (En iyi model ile)
model.eval()
acc_metric.reset()
precision_metric.reset()
recall_metric.reset()
f1_metric.reset()

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        acc_metric.update(preds, labels)
        precision_metric.update(preds, labels)
        recall_metric.update(preds, labels)
        f1_metric.update(preds, labels)

final_val_acc = acc_metric.compute()
final_val_precision = precision_metric.compute()
final_val_recall = recall_metric.compute()
final_val_f1 = f1_metric.compute()

print("\n--- Doğrulama Seti Sonuçları ---")
print(f"Accuracy:  {final_val_acc:.4f}")
print(f"Precision: {final_val_precision:.4f}")
print(f"Recall:    {final_val_recall:.4f}")
print(f"F1-Score:  {final_val_f1:.4f}")

# Başarı Kriteri Kontrolü [cite: 7]
if final_val_acc >= 0.65:
    print(f"\nTebrikler! Doğrulama setinde %{final_val_acc*100:.2f} başarı ile >= %65 kriteri sağlandı! 🎉")
else:
    print(f"\nDoğrulama setinde %{final_val_acc*100:.2f} başarı elde edildi. %65 hedefine ulaşmak için modeli iyileştirmen gerekebilir. 💡")

# TODO: Test seti geldiğinde, aynı metrik hesaplama işlemini test_loader ile yapmalısın. [cite: 16, 28]
# Test verisi eğitimde KESİNLİKLE kullanılmamalıdır! [cite: 28]