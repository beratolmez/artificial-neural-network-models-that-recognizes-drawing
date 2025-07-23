import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView, QProgressDialog,
                             QMessageBox, QSizePolicy, QSpacerItem)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import timm
import torch.nn.functional as F
from natsort import natsorted # natsort kütüphanesi eklendi

# --- AYARLAR: BU KISIMLARI KENDİ PROJENE GÖRE DÜZENLE ---
MODEL_NAME = 'vit_base_patch16_224'
IMG_SIZE = 224
MODEL_PATH = r'C:\Users\SD\VisualStudioCode\multizoo_project\final_model.pth'
# !!! ÇOK ÖNEMLİ: Sınıf isimlerini almak için EĞİTİM VERİSİNİN YOLUNU GİR !!!
TRAIN_DATA_PATH = r'C:\Users\SD\VisualStudioCode\multizoo_project\train'
THUMBNAIL_SIZE = 64 # Tablodaki önizleme boyutu
# --------------------------------------------------------

# Global değişkenler
model = None
class_names = None
device = None
val_transforms = None

class ModelLoaderThread(QThread):
    """Modeli ve sınıf isimlerini arka planda yüklemek için bir thread."""
    finished = pyqtSignal(bool, str) # Başarılı mı, mesaj

    def run(self):
        global model, class_names, device, val_transforms

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Arayüz için kullanılan cihaz: {device}")

            # Sınıf isimlerini yükle
            try:
                temp_dataset = datasets.ImageFolder(TRAIN_DATA_PATH)
                class_names = temp_dataset.classes
                if not class_names:
                    raise ValueError("Eğitim klasöründe sınıf bulunamadı.")
            except FileNotFoundError:
                self.finished.emit(False, f"HATA: Eğitim verisi yolu bulunamadı: {TRAIN_DATA_PATH}\nLütfen TRAIN_DATA_PATH değişkenini doğru ayarlayın.")
                return
            except Exception as e:
                self.finished.emit(False, f"Sınıf isimleri yüklenirken bir hata oluştu: {e}")
                return

            # Modeli yükle
            try:
                loaded_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
                loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                loaded_model.to(device)
                loaded_model.eval() # Modeli değerlendirme moduna al
                model = loaded_model
            except FileNotFoundError:
                self.finished.emit(False, f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
                return
            except Exception as e:
                self.finished.emit(False, f"Model yüklenirken bir hata oluştu: {e}")
                return
            
            val_transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.finished.emit(True, "Model ve sınıflar başarıyla yüklendi.")

        except Exception as e:
            self.finished.emit(False, f"Genel yükleme hatası: {e}")

class PredictionWorker(QThread):
    """Tahmin işlemini arka planda yapmak için worker thread."""
    prediction_done = pyqtSignal(str, float, str, int, int) # Sınıf, Güven, Resim Yolu, Genişlik, Yükseklik (tekil tahmin için)
    folder_prediction_update = pyqtSignal(str, str, float, str, int, int) # Dosya Adı, Tahmin, Güven, Tam Resim Yolu, Genişlik, Yükseklik (klasör tahmin için)
    folder_prediction_finished = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, image_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.folder_path = folder_path

    def run(self):
        if self.image_path:
            self._predict_single_image(self.image_path)
        elif self.folder_path:
            self._predict_folder_images(self.folder_path)

    def _predict_single_image(self, image_path):
        if model is None or val_transforms is None or class_names is None:
            self.error_signal.emit("Model, transformlar veya sınıflar henüz yüklenmedi.")
            return

        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            image_tensor = val_transforms(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            self.prediction_done.emit(predicted_class, confidence_score, image_path, width, height)
        except Exception as e:
            self.error_signal.emit(f"Tahmin yapılırken hata: {e}")

    def _predict_folder_images(self, folder_path):
        if model is None or val_transforms is None or class_names is None:
            self.error_signal.emit("Model, transformlar veya sınıflar henüz yüklenmedi.")
            self.folder_prediction_finished.emit()
            return
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        # os.walk kullanarak tüm alt klasörlerdeki resimleri bul
        all_image_paths = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(supported_formats):
                    all_image_paths.append(os.path.join(dirpath, filename))
        
        for file_path in all_image_paths:
            try:
                image = Image.open(file_path).convert('RGB')
                width, height = image.size
                image_tensor = val_transforms(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
                # file_path'tan sadece dosya adını al
                filename_only = os.path.basename(file_path) 
                self.folder_prediction_update.emit(filename_only, predicted_class, confidence_score, file_path, width, height)

            except Exception as e:
                filename_only = os.path.basename(file_path)
                self.folder_prediction_update.emit(filename_only, "Hata", 0.0, file_path, 0, 0) # Hata durumunda 0,0 boyut gönder
                print(f"'{file_path}' için tahmin yapılırken hata: {e}")
        
        self.folder_prediction_finished.emit()

# Özel QTableWidgetItem sınıfları
class ConfidenceItem(QTableWidgetItem):
    """Güven skoru için sayısal sıralama yapabilen QTableWidgetItem."""
    def __init__(self, text, value):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        # Sayısal değere göre sırala
        return self._value < other._value

class FileNameItem(QTableWidgetItem):
    """Resim adı için büyük/küçük harf duyarsız ve doğal sıralama yapabilen QTableWidgetItem."""
    def __init__(self, text, full_path):
        super().__init__(text)
        self._full_path = full_path # Resim yolunu sakla
    
    # Getter for full_path, useful for retrieving the path later
    def full_path(self):
        return self._full_path

    def __lt__(self, other):
        # natsort kullanarak doğal sıralama yap
        return natsorted([self.text(), other.text()])[0] == self.text()


class MultiZooApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiZoo Hayvan Sınıflandırma Uygulaması")
        self.setGeometry(100, 100, 1200, 800) # Başlangıç boyutu büyütüldü

        self.current_display_image_path = None # Ana ekranda gösterilen resmin yolunu tutar
        self.folder_image_details = [] # Klasördeki her resmin detaylarını saklamak için

        self.init_ui()
        self.load_initial_data()

    def init_ui(self):
        main_layout = QHBoxLayout() # Ana düzeni QHBox olarak değiştirildi

        # Sol Panel: Butonlar ve Tekil Resim Gösterimi
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setAlignment(Qt.AlignTop) # İçeriği yukarı hizala
        left_panel_layout.setContentsMargins(20, 20, 20, 20)

        # Başlık
        title_label = QLabel("Hayvan Sınıflandırma")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        left_panel_layout.addWidget(title_label)

        # Butonlar
        self.btn_select_image = QPushButton("Tek Resim Seç ve Tahmin Et")
        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_select_image.setFixedHeight(50)
        self.btn_select_image.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71; /* Zümrüt yeşili */
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        left_panel_layout.addWidget(self.btn_select_image)

        self.btn_select_folder = QPushButton("Klasör Seç ve Toplu Tahmin Et")
        self.btn_select_folder.clicked.connect(self.select_folder_for_prediction)
        self.btn_select_folder.setFixedHeight(50)
        self.btn_select_folder.setStyleSheet("""
            QPushButton {
                background-color: #3498db; /* Peter River mavisi */
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                padding: 10px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        left_panel_layout.addWidget(self.btn_select_folder)

        # Resim Gösterimi Alanı
        left_panel_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)) # Boşluk
        self.image_label = QLabel("Seçilen Resim Burada Görünecek")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(350, 350) # Sabit boyut
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #95a5a6; /* Gümüş grisi */
                background-color: #ecf0f1; /* Hafif gri */
                border-radius: 10px;
                font-size: 14px;
                color: #7f8c8d;
            }
        """)
        left_panel_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        left_panel_layout.addStretch() # Boş alanı doldur

        # Resim Detayları Paneli
        left_panel_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)) # Boşluk
        details_group_box = QWidget()
        details_layout = QVBoxLayout()
        details_group_box.setLayout(details_layout)
        details_group_box.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        details_title = QLabel("Resim Detayları")
        details_title.setFont(QFont("Arial", 14, QFont.Bold))
        details_title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        details_layout.addWidget(details_title)

        self.detail_image_name = QLabel("Dosya Adı: ")
        self.detail_image_size = QLabel("Boyut: ")
        self.detail_true_class = QLabel("Gerçek Sınıf: ")
        self.detail_predicted_class = QLabel("Tahmin Edilen Sınıf: ")
        self.detail_confidence = QLabel("Güven Skoru: ")

        for label in [self.detail_image_name, self.detail_image_size, self.detail_true_class, self.detail_predicted_class, self.detail_confidence]:
            label.setStyleSheet("font-size: 12px; color: #34495e;")
            details_layout.addWidget(label)
        
        details_layout.addStretch() # İçeriği yukarı yasla
        left_panel_layout.addWidget(details_group_box)
        left_panel_layout.addStretch() # Alt boşluğu doldur
        main_layout.addLayout(left_panel_layout, 2) # Sol panel daha dar (oran 2)

        # Sağ Panel: Klasör Tahmin Sonuçları Tablosu
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setContentsMargins(20, 20, 20, 20)

        self.folder_results_label = QLabel("Klasör Tahmin Sonuçları")
        self.folder_results_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.folder_results_label.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        right_panel_layout.addWidget(self.folder_results_label)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Resim Adı", "Tahmin Edilen Sınıf", "Güven Skoru", "Önizleme"])
        
        # Sütunları dinamik olarak genişlet, son sütun içeriğe göre ayarlanır
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch) # Resim Adı
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents) # Tahmin Edilen Sınıf
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) # Güven Skoru
        self.table_widget.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed) # Önizleme
        self.table_widget.setColumnWidth(3, THUMBNAIL_SIZE + 20) # Önizleme sütunun genişliği
        
        self.table_widget.verticalHeader().setDefaultSectionSize(THUMBNAIL_SIZE + 10) # Satır yüksekliğini ayarla
        self.table_widget.setSortingEnabled(True) # Sıralamayı etkinleştir
        self.table_widget.cellClicked.connect(self.on_table_cell_clicked) # Hücre tıklama olayını bağla
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                selection-background-color: #cceeff; /* Açık mavi seçim */
                gridline-color: #dfe6e9;
                alternate-background-color: #f7f9fa; /* Hafif alternatif satır rengi */
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #bdc3c7; /* Gümüş */
                color: #2c3e50;
                padding: 8px;
                border: 1px solid #7f8c8d;
                font-weight: bold;
                font-size: 13px;
            }
        """)
        right_panel_layout.addWidget(self.table_widget)

        main_layout.addLayout(right_panel_layout, 3) # Sağ panel daha geniş (oran 3)

        self.status_bar = QLabel("") # Başlangıçta boş ve daha az görünür
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #34495e; /* Koyu gri/mavi */
                color: white;
                padding: 8px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
        """)
        # Status bar'ı ana layout'un en altına ekle
        # QHBoxLayout kullandığımız için, status bar'ı doğrudan main_layout'a ekleyemeyiz
        # Bunun yerine, genel pencere layout'una ekleyeceğiz.
        overall_layout = QVBoxLayout()
        overall_layout.addLayout(main_layout)
        overall_layout.addWidget(self.status_bar)
        self.setLayout(overall_layout)


        # Genel pencere stili
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7f8; /* Çok açık gri */
                font-family: Arial, sans-serif;
            }
        """)
        
        self.setLayout(overall_layout)

        # Model yükleme durumunu kapatmak için butonları devre dışı bırak
        self.set_buttons_enabled(False)

    def load_initial_data(self):
        self.status_bar.setText("Model ve sınıf isimleri yükleniyor... Lütfen bekleyin.")
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.finished.connect(self.on_model_loaded)
        self.model_loader_thread.start()

    def on_model_loaded(self, success, message):
        self.status_bar.setText(message)
        if success:
            self.set_buttons_enabled(True)
            # Statü çubuğunu başlangıçta boş bırak
            self.status_bar.setText("Hazır. Resim veya klasör seçin.") 
        else:
            QMessageBox.critical(self, "Yükleme Hatası", message)
            self.status_bar.setText("HATA: Model yüklenemedi! Lütfen yolları kontrol edin.")
            self.set_buttons_enabled(False) # Hata durumunda da butonlar devre dışı kalmalı

    def set_buttons_enabled(self, enabled):
        self.btn_select_image.setEnabled(enabled)
        self.btn_select_folder.setEnabled(enabled)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Bir Hayvan Resmi Seçin", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if not file_path:
            return

        self.status_bar.setText(f"'{os.path.basename(file_path)}' değerlendiriliyor...")
        self.clear_image_details()
        self.table_widget.setRowCount(0) # Klasör tablosunu temizle
        self.current_display_image_path = file_path # Güncel gösterilen resim yolunu kaydet

        self.prediction_worker = PredictionWorker(image_path=file_path)
        self.prediction_worker.prediction_done.connect(self.on_single_prediction_done)
        self.prediction_worker.error_signal.connect(self.on_prediction_error)
        self.prediction_worker.start()

    def display_image_and_details(self, image_path, predicted_class, confidence_score, width, height):
        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.image_label.setText("Resim yüklenemedi!")
                return
            
            # QLabel'ın boyutlarına göre resmi yeniden boyutlandır
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("") # Yazıyı gizle

            # Resim Detaylarını Güncelle
            self.detail_image_name.setText(f"Dosya Adı: {os.path.basename(image_path)}")
            self.detail_image_size.setText(f"Boyut: {width}x{height} px")
            
            # Gerçek sınıfı bulmaya çalış (eğitim klasörü yapısına göre)
            true_class = "Bilinmiyor"
            try:
                # TRAIN_DATA_PATH'tan sonraki ilk klasörün sınıf adı olduğunu varsayıyoruz
                # Örneğin, C:\train\kedi\kedi1.jpg için TRAIN_DATA_PATH = C:\train
                # os.path.dirname(image_path) -> C:\train\kedi
                # os.path.relpath(C:\train\kedi, C:\train) -> kedi
                # kedi.split(os.sep)[0] -> kedi
                # Bu kısım, tahmin edilen klasör yapısının TRAIN_DATA_PATH ile aynı olduğunu varsayar.
                # Eğer tahmin klasörü (örn: 'test') farklı bir kök dizine sahipse,
                # bu mantık doğru 'Gerçek Sınıf'ı bulamayabilir.
                # Genel bir tahmin uygulaması için 'Gerçek Sınıf' her zaman bilinemez.
                if os.path.commonpath([TRAIN_DATA_PATH, image_path]) == TRAIN_DATA_PATH:
                    relative_path = os.path.relpath(os.path.dirname(image_path), TRAIN_DATA_PATH)
                    if relative_path and relative_path != ".": # "." mevcut klasörü temsil eder
                        # Eğer alt klasörler varsa (örn: train/cat/image.jpg), ilk alt klasörü alırız
                        true_class = relative_path.split(os.sep)[0].replace('_', ' ').capitalize() # Alt çizgileri boşlukla değiştir
                else: # Klasör TRAIN_DATA_PATH altında değilse, üst klasör adını almayı dene
                    parent_dir_name = os.path.basename(os.path.dirname(image_path))
                    if parent_dir_name:
                        true_class = parent_dir_name.replace('_', ' ').capitalize()

            except ValueError: # path is not in the subpath
                pass
            self.detail_true_class.setText(f"Gerçek Sınıf: {true_class}")
            
            self.detail_predicted_class.setText(f"Tahmin Edilen Sınıf: {predicted_class.capitalize()}")
            self.detail_confidence.setText(f"Güven Skoru: {confidence_score:.2f}%")

        except Exception as e:
            self.image_label.setText(f"Resim görüntüleme hatası: {e}")
            self.clear_image_details()

    def clear_image_details(self):
        self.detail_image_name.setText("Dosya Adı: ")
        self.detail_image_size.setText("Boyut: ")
        self.detail_true_class.setText("Gerçek Sınıf: ")
        self.detail_predicted_class.setText("Tahmin Edilen Sınıf: ")
        self.detail_confidence.setText("Güven Skoru: ")

    def on_single_prediction_done(self, predicted_class, confidence_score, image_path, width, height):
        self.display_image_and_details(image_path, predicted_class, confidence_score, width, height)
        self.status_bar.setText(f"'{os.path.basename(image_path)}' için tahmin tamamlandı.")

    def on_prediction_error(self, message):
        self.status_bar.setText(f"Hata: {message}")
        self.image_label.setText("Hata oluştu!")
        self.clear_image_details()
        QMessageBox.warning(self, "Tahmin Hatası", message)

    def select_folder_for_prediction(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Tahmin Edilecek Klasörü Seçin")
        if not folder_path:
            return
        
        self.status_bar.setText(f"'{folder_path}' klasöründeki resimler taranıyor...")
        self.image_label.setText("Klasördeki resimler işleniyor...") # Resim alanını bilgilendir
        self.image_label.setPixmap(QPixmap()) # Önceki resmi temizle
        self.clear_image_details()
        self.table_widget.setRowCount(0) # Tabloyu temizle
        self.folder_image_details = [] # Klasördeki her resmin detaylarını sıfırla

        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        # Klasördeki tüm resim dosyalarını özyinelemeli olarak bul
        all_image_paths_in_folder = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(supported_formats):
                    all_image_paths_in_folder.append(os.path.join(dirpath, filename))

        if not all_image_paths_in_folder:
            QMessageBox.information(self, "Klasör Boş", "Seçilen klasörde veya alt klasörlerinde desteklenen resim formatında dosya bulunamadı.")
            self.status_bar.setText("İşlem tamamlandı: Klasör boş.")
            self.image_label.setText("Seçilen Resim Burada Görünecek")
            return

        self.progress_dialog = QProgressDialog("Resimler İşleniyor...", "İptal", 0, len(all_image_paths_in_folder), self)
        self.progress_dialog.setWindowTitle("Klasör Tahmini")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        self.folder_prediction_worker = PredictionWorker(folder_path=folder_path)
        self.folder_prediction_worker.folder_prediction_update.connect(self.on_folder_prediction_update)
        self.folder_prediction_worker.folder_prediction_finished.connect(self.on_folder_prediction_finished)
        self.folder_prediction_worker.error_signal.connect(self.on_prediction_error)
        self.folder_prediction_worker.start()

    def on_folder_prediction_update(self, filename, predicted_class, confidence_score, full_image_path, width, height):
        current_row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(current_row_count)
        
        # Resim Adı (Özel FileNameItem ile büyük/küçük harf duyarsız sıralama)
        file_name_item = FileNameItem(filename, full_image_path)
        self.table_widget.setItem(current_row_count, 0, file_name_item)
        
        # Tahmin Edilen Sınıf
        self.table_widget.setItem(current_row_count, 1, QTableWidgetItem(predicted_class.capitalize()))
        
        # Güven Skoru (Özel ConfidenceItem ile sayısal sıralama)
        confidence_item = ConfidenceItem(f"{confidence_score:.2f}%", confidence_score)
        self.table_widget.setItem(current_row_count, 2, confidence_item)
        
        # Önizleme
        try:
            pixmap = QPixmap(full_image_path)
            if not pixmap.isNull():
                thumbnail_pixmap = pixmap.scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumbnail_label = QLabel()
                thumbnail_label.setPixmap(thumbnail_pixmap)
                thumbnail_label.setAlignment(Qt.AlignCenter)
                self.table_widget.setCellWidget(current_row_count, 3, thumbnail_label)
            else:
                self.table_widget.setItem(current_row_count, 3, QTableWidgetItem("Yüklenemedi"))
        except Exception as e:
            self.table_widget.setItem(current_row_count, 3, QTableWidgetItem(f"Hata: {e}"))
            print(f"Önizleme oluşturulurken hata: {e}")

        # Resim yolunu, boyutlarını ve tahmin bilgilerini sakla
        self.folder_image_details.append({
            'path': full_image_path,
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'width': width,
            'height': height
        })
        
        self.progress_dialog.setValue(self.progress_dialog.value() + 1)
        if self.progress_dialog.wasCanceled():
            self.folder_prediction_worker.terminate() # İşlemi sonlandır
            self.on_folder_prediction_finished() # Bitirme sinyalini manuel olarak gönder
            QMessageBox.information(self, "İptal Edildi", "Klasör değerlendirme işlemi iptal edildi.")

    def on_folder_prediction_finished(self):
        self.progress_dialog.close()
        self.status_bar.setText("Klasör değerlendirme tamamlandı.")
        self.image_label.setText("Seçilen Resim Burada Görünecek") # Resim alanını orijinal haline getir
        self.clear_image_details() # Detayları temizle

    def on_table_cell_clicked(self, row, column):
        # Tıklanan hücrenin yolunu al
        # 0. sütun (Resim Adı) veya 3. sütun (Önizleme) tıklandığında resmi göster
        try:
            # Resim yolu bilgisi 0. sütunun (FileNameItem) içinde saklanıyor
            file_name_item = self.table_widget.item(row, 0)
            if file_name_item and isinstance(file_name_item, FileNameItem):
                full_image_path = file_name_item.full_path()
                
                # Saklanan tüm tahmin verilerini kullan
                if row < len(self.folder_image_details):
                    image_data = self.folder_image_details[row]
                    
                    self.display_image_and_details(
                        image_data['path'],
                        image_data['predicted_class'],
                        image_data['confidence'],
                        image_data['width'],
                        image_data['height']
                    )
                    self.status_bar.setText(f"'{os.path.basename(image_data['path'])}' görüntülendi.")
                    self.current_display_image_path = image_data['path'] # Güncel gösterilen resim yolunu kaydet
                else:
                    QMessageBox.warning(self, "Hata", "Seçilen resim detayları bulunamadı.")
            else:
                QMessageBox.warning(self, "Hata", "Resim bilgisi alınamadı.")

        except Exception as e:
            QMessageBox.critical(self, "Görsel Görüntüleme Hatası", f"Resmi görüntülerken bir hata oluştu: {e}")
            print(f"Hata: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MultiZooApp()
    ex.show()
    sys.exit(app.exec_())