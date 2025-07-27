import os
import cv2
import numpy as np
import pickle
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cosine

# === Konfigurasi ===
DATASET_DIR = 'dataset/training-originals'
WAIFU_DIR = 'dataset/wibu'
TARGET_SIZE = (128, 128)
MODEL_PATH = 'model_cnn_crop.h5'
LABEL_MAP_PATH = 'label_map.pkl'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Label Mapping Otomatis ===
label_map = {}
label_to_index = {}
int_to_label = {}
current_index = 0

for fname in sorted(os.listdir(DATASET_DIR)):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        match = re.search(r'waifu_(\d+)', fname)
        if match:
            code = match.group(1).zfill(4)
        else:
            code = fname[:4]

        if code not in label_map:
            if code == '0010':
                label = 'saya'
            elif int(code) >= 11:
                label = f'waifu-{int(code)}'
            else:
                label = f'person-{int(code)}'

            label_map[code] = label
            label_to_index[label] = current_index
            int_to_label[current_index] = label
            current_index += 1

NUM_CLASSES = len(label_to_index)

# === Load Data Wajah ===
def load_face_data():
    images = [] 
    labels = []
    for fname in os.listdir(DATASET_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_code = fname[6:10] if fname.startswith("waifu_") else fname[:4]
            if label_code not in label_map:
                continue
            label_name = label_map[label_code]
            img_path = os.path.join(DATASET_DIR, fname)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_crop = img[y:y+h, x:x+w]
                face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).resize(TARGET_SIZE)
                img_array = np.array(face_img) / 255.0
                images.append(img_array)
                labels.append(label_to_index[label_name])
                break
    return np.array(images), to_categorical(labels, num_classes=NUM_CLASSES)

# === Training CNN ===
print(" Memuat dan memproses data...")
X, y = load_face_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Data: {len(X_train)} train | {len(X_val)} val | {NUM_CLASSES} kelas")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_val, y_val))

# === Ekstrak embedding waifu ===
waifu_embeddings = []
waifu_labels = []
for fname in os.listdir(WAIFU_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(WAIFU_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Gagal membaca gambar: {img_path}")
            continue
        img_resized = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(TARGET_SIZE)
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        emb = model.predict(img_array, verbose=0)[0]
        waifu_embeddings.append(emb)
        label_code = fname[6:10] if fname.startswith("waifu_") else fname[:4]
        label_name = label_map.get(label_code, fname)
        waifu_labels.append(label_name)

model.save(MODEL_PATH)
with open(LABEL_MAP_PATH, 'wb') as f:
    pickle.dump(int_to_label, f)

print(" Model dan label_map disimpan!")

# === Fungsi pencocokan waifu terbaik ===
def get_best_waifu_match(face_img_array):
    user_vector = model.predict(face_img_array, verbose=0)[0]
    best_score = float('inf')
    best_waifu = None
    for i, emb in enumerate(waifu_embeddings):
        score = cosine(user_vector, emb)
        if score < best_score:
            best_score = score
            best_waifu = waifu_labels[i]
    return best_waifu, 1 - best_score

print(" Membuka kamera untuk pengujian model...")
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, 'rb') as f:
    int_to_label = pickle.load(f)

cap = cv2.VideoCapture(0)
prev_waifu = None
waifu_counter = 0
change_cooldown = 3  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize(TARGET_SIZE)
        img_array = np.expand_dims(np.array(face_img) / 255.0, axis=0)
        pred = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred)
        confidence = float(np.max(pred)) * 100
        label = int_to_label.get(pred_class, "Unknown")  

      
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
        if label == 'saya':
            best_waifu, sim_score = get_best_waifu_match(img_array)

            if best_waifu != prev_waifu:
                waifu_counter += 1
            else:
                waifu_counter = 0

            if waifu_counter >= change_cooldown or prev_waifu is None:
                prev_waifu = best_waifu
                waifu_counter = 0  # reset

            if prev_waifu:
                cv2.putText(frame, f"Cocok dengan {prev_waifu} ({sim_score:.2%})", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 20, 255), 2)

    cv2.imshow("Test Kamera - Tekan 'q' untuk keluar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
