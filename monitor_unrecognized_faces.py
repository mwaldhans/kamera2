#!/usr/bin/env python3
"""
Skript pro monitorování kamery a ukládání nerozpoznaných obličejů

Skript kontinuálně monitoruje kameru, detekuje obličeje a ukládá
nerozpoznané obličeje do složky 'osoby/nerozpoznana/'.

Použití:
python monitor_unrecognized_faces.py

Parametry:
- ip_address: IP adresa kamery (výchozí: http://10.1.195.238:8080/video)
- username: Uživatelské jméno (výchozí: admin)
- password: Heslo (výchozí: admin)
- confidence_threshold: Práh jistoty pro rozpoznání (výchozí: 80)
- save_interval: Minimální interval mezi uložením stejného obličeje (sekundy, výchozí: 30)
"""

import cv2
import numpy as np
import os
import sys
import pickle
import time
from datetime import datetime
import argparse

class FaceMonitor:
    def __init__(self, ip_address="http://10.1.195.238:8080/video",
                 username="admin", password="admin",
                 confidence_threshold=80, save_interval=30):

        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.confidence_threshold = confidence_threshold
        self.save_interval = save_interval

        # Vytvoření složky pro nerozpoznané obličeje
        self.unrecognized_folder = os.path.join("osoby", "nerozpoznana")
        os.makedirs(self.unrecognized_folder, exist_ok=True)

        # Inicializace kamery
        self.kamera = KameraOD325HD(ip_address, username, password)

        # Načtení trénovaných dat
        self.load_trained_data()

        # Sledování posledních uložení (pro prevenci duplicit)
        self.last_save_times = {}  # {face_hash: timestamp}

    def load_trained_data(self):
        """Načtení trénovaných dat o obličejích"""
        try:
            if os.path.exists('face_data.pkl'):
                with open('face_data.pkl', 'rb') as f:
                    data = pickle.load(f)

                self.kamera.known_faces = data.get('known_faces', {})
                self.kamera.is_trained = data.get('is_trained', False)

                # Aktualizace trénovacích dat
                self.kamera.update_training_data()

                # Načtení natrénovaného modelu
                if self.kamera.is_trained and self.kamera.face_recognition_available and os.path.exists('face_recognizer.yml'):
                    self.kamera.face_recognizer.read('face_recognizer.yml')

                print(f"✓ Načteno {len(self.kamera.known_faces)} známých obličejů")
                return True
            else:
                print("⚠ Soubor face_data.pkl nenalezen - žádná trénovaná data")
                return False

        except Exception as e:
            print(f"✗ Chyba při načítání dat: {e}")
            return False

    def get_face_hash(self, face_image):
        """Vytvoření jednoduchého hash pro obličej (pro detekci duplicit)"""
        # Zmenšení obrázku pro rychlejší hash
        small_face = cv2.resize(face_image, (50, 50))
        # Výpočet průměrného hash
        return hash(small_face.tobytes())

    def should_save_face(self, face_hash):
        """Rozhodnutí, zda uložit obličej (na základě času od posledního uložení)"""
        current_time = time.time()

        if face_hash in self.last_save_times:
            time_since_last_save = current_time - self.last_save_times[face_hash]
            if time_since_last_save < self.save_interval:
                return False

        self.last_save_times[face_hash] = current_time
        return True

    def save_unrecognized_face(self, face_image, confidence=None):
        """Uložení nerozpoznaného obličeje"""
        # Vytvoření hash pro kontrolu duplicit
        face_hash = self.get_face_hash(face_image)

        if not self.should_save_face(face_hash):
            return False

        # Vytvoření názvu souboru s timestampem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        confidence_str = f"_conf{confidence:.0f}" if confidence is not None else ""
        filename = f"unrecognized_{timestamp}{confidence_str}.jpg"

        filepath = os.path.join(self.unrecognized_folder, filename)

        # Uložení obrázku
        cv2.imwrite(filepath, face_image)

        print(f"✓ Uložen nerozpoznaný obličej: {filename}")
        return True

    def process_frame(self, frame):
        """Zpracování jednoho snímku"""
        if frame is None:
            return

        # Detekce a rozpoznání obličejů
        if self.kamera.is_trained and self.kamera.face_recognition_available:
            frame, faces_count = self.kamera.recognize_faces(frame)
        else:
            # Jen detekce bez rozpoznání
            frame, faces_count = self.kamera.detect_faces(frame)

        # Pokud máme trénovaný model, zpracujeme nerozpoznané obličeje
        if self.kamera.is_trained and self.kamera.face_recognition_available:
            self.process_unrecognized_faces(frame)

    def process_unrecognized_faces(self, frame):
        """Zpracování nerozpoznaných obličejů ve snímku"""
        # Převedení na grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detekce obličejů
        faces = self.kamera.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Filtrování obličejů
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:
                area = w * h
                if 2500 < area < 90000:
                    filtered_faces.append((x, y, w, h))

        # Zpracování každého obličeje
        for (x, y, w, h) in filtered_faces:
            face_roi = gray[y:y+h, x:x+w]

            try:
                # Pokus o rozpoznání
                label, confidence = self.kamera.face_recognizer.predict(face_roi)

                # Kontrola, zda je obličej nerozpoznaný nebo má nízkou jistotu
                should_save = False

                if confidence > self.confidence_threshold:
                    # Vysoká nejistota - pravděpodobně nerozpoznaný
                    should_save = True
                elif label >= len(self.kamera.face_labels):
                    # Neplatný label
                    should_save = True
                else:
                    # Zkontrolujeme, zda je to známá osoba s nízkou jistotou
                    name = self.kamera.face_labels[label]
                    if confidence > 60:  # Střední jistota - může být chyba
                        should_save = True

                if should_save:
                    # Extrakce a uložení obličeje
                    face_color = frame[y:y+h, x:x+w]  # Barevný obličej
                    self.save_unrecognized_face(face_color, confidence)

            except Exception as e:
                # Pokud rozpoznání selže, uložíme obličej
                face_color = frame[y:y+h, x:x+w]
                self.save_unrecognized_face(face_color)

    def run(self):
        """Hlavní loop monitorování"""
        print("=== Monitorování nerozpoznaných obličejů ===")
        print(f"Kamera: {self.ip_address}")
        print(f"Složka pro uložení: {self.unrecognized_folder}")
        print(f"Práh jistoty: {self.confidence_threshold}")
        print(f"Interval uložení: {self.save_interval}s")
        print()

        # Připojení ke kameře
        if not self.kamera.connect():
            print("✗ Nelze se připojit ke kameře!")
            return

        print("✓ Připojeno ke kameře")
        print("Spouštím monitorování... (Ctrl+C pro zastavení)")
        print()

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                frame = self.kamera.get_frame()

                if frame is not None:
                    self.process_frame(frame)
                    frame_count += 1

                    # Zobrazení statistik každých 100 snímků
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Zpracováno {frame_count} snímků, FPS: {fps:.1f}")

                # Krátká pauza pro snížení CPU zatížení
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nZastavování monitorování...")

        finally:
            self.kamera.cleanup()
            print("✓ Monitorování ukončeno")


# Import třídy KameraOD325HD z hlavního souboru
# (Zjednodušená verze bez GUI závislostí)
class KameraOD325HD:
    def __init__(self, ip_address, username="admin", password="admin"):
        self.ip_address = ip_address if ip_address.startswith('http') else f'http://{ip_address}'
        self.username = username
        self.password = password

        # Typické URL pro MJPEG stream
        base_url = self.ip_address.rstrip('/')
        if '/video' in self.ip_address:
            self.stream_urls = [self.ip_address]
        else:
            ip_only = self.ip_address.replace('http://', '').replace('https://', '')
            self.stream_urls = [
                f'http://{username}:{password}@{ip_only}/video?resolution=1920x1080&fps=15',
                f'http://{username}:{password}@{ip_only}/video?resolution=1920x1080',
                f'http://{username}:{password}@{ip_only}/video',
            ]

        self.cap = None
        self.running = False
        self.connected = False

        # Face recognition components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognition_available = hasattr(cv2, 'face')

        if self.face_recognition_available:
            try:
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            except:
                self.face_recognizer = None
                self.face_recognition_available = False

        self.known_faces = {}
        self.face_labels = []
        self.face_data = []
        self.is_trained = False

    def connect(self):
        """Připojení ke kameře"""
        print("Připojování ke kameře...")
        for stream_url in self.stream_urls:
            try:
                self.cap = cv2.VideoCapture(stream_url)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.connected = True
                        return True
                self.cap.release()
            except:
                continue
        return False

    def get_frame(self):
        """Získání snímku"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_faces(self, frame):
        """Detekce obličejů"""
        if frame is None:
            return frame, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Filtrování
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:
                area = w * h
                if 2500 < area < 90000:
                    filtered_faces.append((x, y, w, h))

        # Kreslení obdélníků
        for (x, y, w, h) in filtered_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame, len(filtered_faces)

    def recognize_faces(self, frame):
        """Rozpoznání obličejů"""
        if not self.face_recognition_available or not self.is_trained:
            return self.detect_faces(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Filtrování
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:
                area = w * h
                if 2500 < area < 90000:
                    filtered_faces.append((x, y, w, h))

        faces = filtered_faces

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            try:
                label, confidence = self.face_recognizer.predict(face_roi)

                if confidence > 80:
                    continue

                name = "Neznamy"
                if label < len(self.face_labels):
                    name = self.face_labels[label]

                color = (0, 255, 0) if confidence < 40 else (0, 165, 255) if confidence < 60 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f'{name} ({confidence:.1f})'
                cv2.putText(frame, text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            except:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Oblicej', (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        return frame, len(faces)

    def update_training_data(self):
        """Aktualizace trénovacích dat"""
        self.face_labels = []
        self.face_data = []
        for name, images in self.known_faces.items():
            for img in images:
                self.face_labels.append(name)
                self.face_data.append(img)

    def cleanup(self):
        """Uklídnutí zdrojů"""
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description='Monitorování nerozpoznaných obličejů')
    parser.add_argument('--ip', default='http://10.1.195.238:8080/video', help='IP adresa kamery')
    parser.add_argument('--user', default='admin', help='Uživatelské jméno')
    parser.add_argument('--pass', default='admin', help='Heslo')
    parser.add_argument('--confidence', type=float, default=80, help='Práh jistoty pro rozpoznání')
    parser.add_argument('--interval', type=int, default=30, help='Minimální interval mezi uložením (sekundy)')

    args = parser.parse_args()

    monitor = FaceMonitor(
        ip_address=args.ip,
        username=args.user,
        password=getattr(args, 'pass'),  # pass je rezervované slovo
        confidence_threshold=args.confidence,
        save_interval=args.interval
    )

    monitor.run()


if __name__ == "__main__":
    main()