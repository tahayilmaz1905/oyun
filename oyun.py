import cv2
import random
import numpy as np
from tensorflow import keras
import mediapipe as mp
import time
import os

# Keras modelini yükle
try:
    model = keras.models.load_model('keras_model.h5')
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    exit()

# Etiketleri oku
try:
    with open('labels.txt', 'r') as file:
        labels = file.read().splitlines()
        labels = [label.split(' ', 1)[1] if ' ' in label else label for label in labels]
except FileNotFoundError:
    print("labels.txt bulunamadı!")
    exit()

# Resimler
resimler = {
    "Arabalar": ['Arabalar/Audi.jpg', 'Arabalar/Bmw.jpg', 'Arabalar/Citroen.jpg', 'Arabalar/Egea.jpg', 'Arabalar/Ferrari.jpg', 'Arabalar/Kia.jpg', 'Arabalar/Linea.jpg', 'Arabalar/Mercedes.jpg', 'Arabalar/Porshe.jpg', 'Arabalar/Togg.jpg', 'Arabalar/Honda.jpg', 'Arabalar/Prelude.jpg', 'Arabalar/Cle.jpg', 'Arabalar/Convortible.jpg', 'Arabalar/Audia5.jpg'],
    "Hayvanlar": ['Hayvanlar/Aslan.jpg', 'Hayvanlar/At.jpg', 'Hayvanlar/Boga.jpg', 'Hayvanlar/İnek.jpg', 'Hayvanlar/Keçi.jpg', 'Hayvanlar/Kopek.jpg', 'Hayvanlar/Koyun.jpg', 'Hayvanlar/Kus.jpg', 'Hayvanlar/Zebra.jpg', 'Hayvanlar/Zürafa.jpg', 'Hayvanlar/Tilki.jpg', 'Hayvanlar/Fok.jpg', 'Hayvanlar/Horoz.jpg', 'Hayvanlar/Cita.jpg', 'Hayvanlar/Goril.jpg'],
    "Meyveler": ['Meyveler/Ahududu.jpg', 'Meyveler/Armut.jpg', 'Meyveler/Ayva.jpg', 'Meyveler/Çilek.jpg', 'Meyveler/Erik.jpg', 'Meyveler/Karpuz.jpg', 'Meyveler/Kavun.jpg', 'Meyveler/Kayisi.jpg', 'Meyveler/Kiraz.jpg', 'Meyveler/Şeftali.jpg', 'Meyveler/Tarlasera.jpg', 'Meyveler/Nar.jpg', 'Meyveler/Granadila.jpg', 'Meyveler/Ananas.jpg', 'Meyveler/Yerelmasi.jpg']
}

# Dinamik arka planlar
backgrounds = {
    "Arabalar": cv2.imread("backgrounds/road.jpg"),
    "Hayvanlar": cv2.imread("backgrounds/forest.jpg"),
    "Meyveler": cv2.imread("backgrounds/orchard.jpg")
}
# Arka planların yüklendiğini kontrol et
for kategori, bg in backgrounds.items():
    if bg is None:
        print(f"Arka plan yüklenemedi: backgrounds/{kategori.lower()}.jpg")
        backgrounds[kategori] = np.zeros((480, 640, 3), dtype=np.uint8)  # Yedek siyah arka plan

# Oyun durumu
game_state = "category_select"
secilen_bolum = None
puan = 0
resimler_listesi = []
max_resim_sayisi = 3
base_hiz = 3
flash_effect = None
flash_duration = 0
secim_yapildi = False
secim_bekleme = 0
start_time = None
game_duration = 30  # Oyun süresi 30 saniye
game_over = False
kullanilan_resimler = []
hatali_resimler = []
bolge_index = 0  # Bölgeleri döngüsel olarak kullanmak için
dogru_resim_sayisi = 0  # Doğru resim sayısını takip et

# Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılmadı!")
    exit()

# MediaPipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Resim düşür fonksiyonu (Güncellenmiş)
def resim_dusur():
    global kullanilan_resimler, hatali_resimler, secilen_bolum, bolge_index, start_time, dogru_resim_sayisi
    tum_resimler = []
    for kategori, resim_listesi in resimler.items():
        for resim_path in resim_listesi:
            if resim_path not in hatali_resimler and os.path.exists(resim_path):
                tum_resimler.append((resim_path, kategori))
            else:
                if resim_path not in hatali_resimler:
                    print(f"Dosya bulunamadı: {resim_path}")
                    hatali_resimler.append(resim_path)
    
    if not tum_resimler:
        print("Hiçbir geçerli resim dosyası bulunamadı!")
        return None, -50, random.randint(50, 490), random.uniform(2.0, 4.0), None
    
    # Kullanılmayan resimleri seç, yoksa sıfırla
    kullanilmayan = [(path, kat) for path, kat in tum_resimler if path not in kullanilan_resimler]
    if not kullanilmayan:
        print("Kullanılan resimler sıfırlanıyor...")
        kullanilan_resimler.clear()
        kullanilmayan = [(path, kat) for path, kat in tum_resimler]
    
    # Oyun süresine bağlı olarak doğru kategori olasılığı
    elapsed_time = time.time() - start_time if start_time else 0
    if elapsed_time < 10:
        dogru_kategori_olasilik = 0.50  # İlk 10 saniye: %50
    elif elapsed_time < 20:
        dogru_kategori_olasilik = 0.70  # 10-20 saniye: %70
    else:
        dogru_kategori_olasilik = 0.90  # 20-30 saniye: %90
    
    # Doğru kategori seçimi
    if secilen_bolum and random.random() < dogru_kategori_olasilik:
        secilen_kategori = secilen_bolum
    else:
        diger_kategoriler = [kat for kat in resimler.keys() if kat != secilen_bolum]
        secilen_kategori = random.choice(diger_kategoriler) if diger_kategoriler else secilen_bolum
    
    # Doğru kategoriden zaten bir resim varsa ve erken safhadaysa başka kategori dene
    if secilen_kategori == secilen_bolum and elapsed_time < 15:
        mevcut_kategoriler = [res["kategori"] for res in resimler_listesi if res["kategori"]]
        if secilen_bolum in mevcut_kategoriler and dogru_resim_sayisi >= 6:
            diger_kategoriler = [kat for kat in resimler.keys() if kat != secilen_bolum]
            if diger_kategoriler:
                secilen_kategori = random.choice(diger_kategoriler)
    
    kategori_resimleri = [(path, kat) for path, kat in kullanilmayan if kat == secilen_kategori]
    
    if not kategori_resimleri:
        print(f"{secilen_kategori} kategorisinde kullanılmayan resim kalmadı, tüm resimler sıfırlanıyor...")
        kullanilan_resimler.clear()
        kategori_resimleri = [(path, kat) for path, kat in tum_resimler if kat == secilen_kategori]
        if not kategori_resimleri:
            print(f"{secilen_kategori} kategorisinde geçerli resim bulunamadı!")
            return None, -50, random.randint(50, 490), random.uniform(2.0, 4.0), None
    
    max_deneme = 10
    deneme = 0
    while kategori_resimleri and deneme < max_deneme:
        resim_path, bolum = random.choice(kategori_resimleri)
        resim = cv2.imread(resim_path)
        if resim is not None:
            kullanilan_resimler.append(resim_path)
            resim = cv2.resize(resim, (120, 120))
            bolgeler = [(50, 190), (200, 340), (350, 490)]
            # Döngüsel olarak bölge seç ve çakışmayı önle
            for _ in range(len(bolgeler)):
                secilen_bolge = bolgeler[bolge_index % len(bolgeler)]
                x_min, x_max = secilen_bolge
                mevcut_x = [int(res["x"]) for res in resimler_listesi]
                mevcut_y = [int(res["y"]) for res in resimler_listesi]
                x = random.randint(x_min, x_max)
                # Çakışma kontrolü (x ve y ekseninde, minimum 140 piksel boşluk)
                for _ in range(5):
                    if not any(abs(x - mx) < 140 and abs(-50 - my) < 140 for mx, my in zip(mevcut_x, mevcut_y)):
                        bolge_index += 1
                        print(f"Seçilen resim: {resim_path}, Kategori: {bolum}, x: {x}, Bölge: {secilen_bolge}")
                        # Doğru resim seçildiyse sayacı artır
                        if bolum == secilen_bolum:
                            dogru_resim_sayisi += 1
                        return resim, -50, x, random.uniform(2.0, 4.0), bolum
                    x = random.randint(x_min, x_max)
                bolge_index += 1  # Çakışma varsa bir sonraki bölgeyi dene
            # Tüm bölgelerde çakışma varsa rastgele x
            x = random.randint(50, 490)
            print(f"Seçilen resim: {resim_path}, Kategori: {bolum}, x: {x}, Bölge: Rastgele")
            # Doğru resim seçildiyse sayacı artır
            if bolum == secilen_bolum:
                dogru_resim_sayisi += 1
            return resim, -50, x, random.uniform(2.0, 4.0), bolum
        else:
            print(f"Resim yüklenemedi: {resim_path}")
            hatali_resimler.append(resim_path)
            kategori_resimleri = [(p, k) for p, k in kategori_resimleri if p != resim_path]
            deneme += 1
    
    print(f"{secilen_kategori} kategorisinde geçerli resim bulunamadı!")
    return None, -50, random.randint(50, 490), random.uniform(2.0, 4.0), None

# Resmi sınıflandır (Keras)
def siniflandir_resim(resim):
    try:
        if resim is None or np.all(resim == 0):
            print("Sınıflandırma hatası: Resim None veya boş")
            return None
        resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
        resim = cv2.resize(resim, (224, 224))
        resim = np.expand_dims(resim, axis=0)
        resim = resim / 255.0
        pred = model.predict(resim, verbose=0)
        label_idx = np.argmax(pred)
        predicted_label = labels[label_idx]
        print(f"Resim sınıflandırma sonucu: {predicted_label}")
        return predicted_label
    except Exception as e:
        print(f"Resim sınıflandırma hatası: {e}")
        return None

# El hareketini algıla (MediaPipe ile)
def algila_el_hareketi(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return index_x, index_y
    return None, None

# Kategori kutuları çiz
def draw_category_boxes(frame, index_x, index_y):
    global secilen_bolum, game_state, start_time
    categories = ["Hayvanlar", "Meyveler", "Arabalar"]
    for i, cat in enumerate(categories):
        x, y = 50, 100 + i * 80
        # Gölge efekti için arka plan
        cv2.rectangle(frame, (x - 7, y - 7), (x + 200 + 7, y + 60 + 7), (20, 20, 20), -1)
        color = (0, 255, 0) if cat == secilen_bolum else (255, 165, 0)
        cv2.rectangle(frame, (x, y), (x + 200, y + 60), color, -1)
        cv2.putText(frame, cat, (x + 10, y + 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, cat, (x + 12, y + 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)  # Gölge
        if index_x is not None and index_y is not None:
            if x <= index_x <= x + 200 and y <= index_y <= y + 60:
                secilen_bolum = cat
                game_state = "playing"
                start_time = time.time()
    cv2.putText(frame, "Bir kategori seciniz", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Bir kategori seciniz", (22, 52), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)

# İlk resimleri başlat
for _ in range(max_resim_sayisi):
    resim, y, x, hiz, kategori = resim_dusur()
    if resim is not None:
        resimler_listesi.append({"resim": resim, "y": y, "x": x, "hiz": hiz, "kategori": kategori, "feedback": None})

# Fare tıklama için callback
def mouse_callback(event, x, y, flags, param):
    global game_state, secilen_bolum, puan, start_time, game_over, dogru_resim_sayisi
    if event == cv2.EVENT_LBUTTONDOWN and game_over:
        if 300 <= x <= 500 and 300 <= y <= 360:
            game_state = "category_select"
            secilen_bolum = None
            puan = 0
            start_time = None
            game_over = False
            kullanilan_resimler.clear()
            resimler_listesi.clear()
            dogru_resim_sayisi = 0  # Sayacı sıfırla
            for _ in range(max_resim_sayisi):
                resim, y, x, hiz, kategori = resim_dusur()
                if resim is not None:
                    resimler_listesi.append({"resim": resim, "y": y, "x": x, "hiz": hiz, "kategori": kategori, "feedback": None})

cv2.namedWindow("Game Screen")
cv2.setMouseCallback("Game Screen", mouse_callback)

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    frame = cv2.flip(frame, 1)
    # Dinamik arka plan ekle
    if secilen_bolum and backgrounds.get(secilen_bolum) is not None:
        bg = cv2.resize(backgrounds[secilen_bolum], (frame.shape[1], frame.shape[0]))
        frame_resim = cv2.addWeighted(frame, 0.8, bg, 0.2, 0)
    else:
        default_bg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        frame_resim = cv2.addWeighted(frame, 0.8, default_bg, 0.2, 0)

    try:
        index_x, index_y = algila_el_hareketi(frame_resim)
    except Exception as e:
        print(f"El algılama hatası: {e}")
        index_x, index_y = None, None

    if game_state == "category_select":
        draw_category_boxes(frame_resim, index_x, index_y)

    elif game_state == "playing" and not game_over:
        if start_time is not None:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, game_duration - int(elapsed_time))
            if remaining_time <= 0:
                game_over = True

        if secim_yapildi and time.time() - secim_bekleme > 0.5:
            secim_yapildi = False

        i = 0
        while i < len(resimler_listesi):
            res = resimler_listesi[i]
            if res["resim"] is None:
                resimler_listesi.pop(i)
                if res["kategori"] == secilen_bolum:
                    dogru_resim_sayisi -= 1  # Doğru resim silindiyse sayacı azalt
                continue
            res["y"] += res["hiz"]
            y = int(res["y"])
            x = int(res["x"])
            resim_end_y = min(y + res["resim"].shape[0], frame.shape[0])
            resim_end_x = min(x + res["resim"].shape[1], frame.shape[1])
            if y >= 0 and resim_end_y > y and x >= 0 and resim_end_x > x:
                try:
                    frame_resim[y:resim_end_y, x:resim_end_x] = res["resim"][:resim_end_y - y, :resim_end_x - x]
                    # Animasyonlu geri bildirim: Çerçeve çizimi
                    if res.get("feedback") and res["feedback"]["duration"] > 0:
                        color = res["feedback"]["color"]
                        cv2.rectangle(frame_resim, (x-5, y-5), (resim_end_x+5, resim_end_y+5),  color, 3)
                        res["feedback"]["duration"] -= 1
                        if res["feedback"]["duration"] == 0:
                            res["feedback"] = None
                    print(f"Resim çizildi: x={x}, y={y}, kategori={res['kategori']}")
                except Exception as e:
                    print(f"Resim yerleştirme hatası: {e}")
            i += 1

        if not secim_yapildi and index_x is not None and index_y is not None and not game_over:
            for i, res in enumerate(resimler_listesi):
                if res["resim"] is None or np.all(res["resim"] == 0):
                    continue
                y = int(res["y"])
                x = int(res["x"])
                resim_end_y = min(y + res["resim"].shape[0], frame.shape[0])
                resim_end_x = min(x + res["resim"].shape[1], frame.shape[1])
                if (x - 20 <= index_x <= resim_end_x + 20) and (y - 20 <= index_y <= resim_end_y + 20):
                    resim_kategori = siniflandir_resim(res["resim"])
                    print(f"Seçim: Model Tahmini={resim_kategori}, Seçilen Bölüm={secilen_bolum}, Gerçek Kategori={res['kategori']}")
                    if res["kategori"] and res["kategori"] == secilen_bolum:
                        puan += 10
                        flash_effect = (0, 255, 0, 50)  # Yumuşak yeşil flaş
                        flash_duration = 8
                        res["feedback"] = {"color": (0, 255, 0), "duration": 10}  # Yeşil çerçeve, 10 kare
                        print(f"Doğru seçim! Puan: {puan}")
                        dogru_resim_sayisi -= 1  # Doğru resim seçildiyse sayacı azalt
                    elif res["kategori"]:
                        puan -= 5
                        flash_effect = (255, 0, 0, 50)  # Yumuşak kırmızı flaş
                        flash_duration = 8
                        res["feedback"] = {"color": (255, 0, 0), "duration": 10}  # Kırmızı çerçeve, 10 kare
                        print(f"Yanlış seçim! Puan: {puan}")
                    secim_yapildi = True
                    secim_bekleme = time.time()
                    resimler_listesi.pop(i)
                    break

        resimler_listesi[:] = [res for res in resimler_listesi if int(res["y"]) <= frame.shape[0]]
        while len(resimler_listesi) < max_resim_sayisi and not game_over:
            resim, y, x, hiz, kategori = resim_dusur()
            if resim is not None:
                resimler_listesi.append({"resim": resim, "y": y, "x": x, "hiz": hiz, "kategori": kategori, "feedback": None})

        if flash_effect and flash_duration > 0:
            overlay = frame_resim.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), flash_effect[:3], -1)
            frame_resim = cv2.addWeighted(overlay, flash_effect[3]/255, frame_resim, 1 - flash_effect[3]/255, 0)
            flash_duration -= 1
            if flash_duration == 0:
                flash_effect = None

        # Puan, kategori ve süre göster (geliştirilmiş görünüm)
        overlay = frame_resim.copy()
        cv2.rectangle(overlay, (10, 10), (300, 140), (30, 30, 30), -1)  # Yarı saydam koyu arka plan
        alpha = 0.7
        frame_resim = cv2.addWeighted(overlay, alpha, frame_resim, 1 - alpha, 0)
        cv2.rectangle(frame_resim, (10, 10), (300, 140), (200, 200, 200), 2)  # Şık kenarlık
        cv2.putText(frame_resim, f"Kategori: {secilen_bolum}", 
                    (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame_resim, f"Kategori: {secilen_bolum}", 
                    (22, 52), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)  # Gölge efekti
        cv2.putText(frame_resim, f"Puan: {puan}", 
                    (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame_resim, f"Puan: {puan}", 
                    (22, 92), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)  # Gölge efekti
        cv2.putText(frame_resim, f"Kalan Sure: {remaining_time}s", 
                    (20, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame_resim, f"Kalan Sure: {remaining_time}s", 
                    (22, 132), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)  # Gölge efekti

    if game_over:
        # Oyun bitti ekranı (geliştirilmiş)
        overlay = frame_resim.copy()
        cv2.rectangle(overlay, (150, 150), (550, 400), (30, 30, 30), -1)  # Yarı saydam arka plan
        alpha = 0.8
        frame_resim = cv2.addWeighted(overlay, alpha, frame_resim, 1 - alpha, 0)
        cv2.rectangle(frame_resim, (150, 150), (550, 400), (255, 215, 0), 3)  # Altın sarısı kenarlık
        cv2.putText(frame_resim, f"Oyun Bitti! Puan: {puan}", 
                    (180, 250), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame_resim, f"Oyun Bitti! Puan: {puan}", 
                    (183, 253), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)  # Gölge efekti
        # Yeniden Oyna butonu
        cv2.rectangle(frame_resim, (300, 300), (500, 360), (0, 200, 0), -1)  # Canlı yeşil
        cv2.rectangle(frame_resim, (300, 300), (500, 360), (255, 255, 255), 3)  # Beyaz kenarlık
        cv2.rectangle(frame_resim, (295, 295), (505, 365), (255, 215, 0), 2)  # Altın sarısı dış çerçeve
        cv2.putText(frame_resim, "Yeniden Oyna", (320, 340), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_resim, "Yeniden Oyna", (322, 342), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
        
        # El ile Yeniden Oyna butonuna tıklama
        if index_x is not None and index_y is not None and not secim_yapildi:
            if 300 <= index_x <= 500 and 300 <= index_y <= 360:
                game_state = "category_select"
                secilen_bolum = None
                puan = 0
                start_time = None
                game_over = False
                kullanilan_resimler.clear()
                resimler_listesi.clear()
                dogru_resim_sayisi = 0  # Sayacı sıfırla
                for _ in range(max_resim_sayisi):
                    resim, y, x, hiz, kategori = resim_dusur()
                    if resim is not None:
                        resimler_listesi.append({"resim": resim, "y": y, "x": x, "hiz": hiz, "kategori": kategori, "feedback": None})
                secim_yapildi = True
                secim_bekleme = time.time()

    try:
        cv2.namedWindow("Game Screen", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Screen", 800, 600)
        cv2.imshow("Game Screen", frame_resim)
    except Exception as e:
        print(f"Pencere görüntüleme hatası: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
hands.close()