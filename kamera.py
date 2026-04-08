import cv2
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import io
from datetime import datetime
import os
import pickle

class KameraOD325HD:
    def __init__(self, ip_address, username="admin", password="admin"):
        """
        Inicializace prohlížeče kamery s Full HD rozlišením a 15 FPS
        
        Args:
            ip_address: IP adresa kamery (např. http://10.1.195.238:8080/video)
            username: Uživatelské jméno (výchozí: admin)
            password: Heslo (výchozí: admin)
        """
        self.ip_address = ip_address if ip_address.startswith('http') else f'http://{ip_address}'
        self.username = username
        self.password = password
        
        # Typické URL pro MJPEG stream - detekce typu kamery
        base_url = self.ip_address.rstrip('/')
        
        # Pokud URL už obsahuje stream endpoint (/video), použijeme ji přímo
        if '/video' in self.ip_address:
            self.stream_urls = [self.ip_address]
        else:
            # Standardní stream kamera s Full HD a 15 FPS
            ip_only = self.ip_address.replace('http://', '').replace('https://', '')
            self.stream_urls = [
                f'http://{username}:{password}@{ip_only}/video?resolution=1920x1080&fps=15',
                f'http://{username}:{password}@{ip_only}/video?resolution=1920x1080',
                f'http://{username}:{password}@{ip_only}/video',
            ]
        
        self.cap = None
        self.running = False
        self.window_name = "Kamera Stream - Full HD 1920x1080 @ 15 FPS"
        self.connected = False
        self.frame_count = 0
        self.last_url = None
        
        # Face recognition components
        print("Inicializace rozpoznávání obličejů...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"Face cascade loaded: {not self.face_cascade.empty()}")
        
        # Check if face recognition is available
        self.face_recognition_available = hasattr(cv2, 'face')
        print(f"cv2.face available: {self.face_recognition_available}")
        
        if self.face_recognition_available:
            try:
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                print("✓ LBPH rozpoznávač vytvořen")
            except Exception as e:
                print(f"✗ Chyba při vytváření rozpoznávače: {e}")
                self.face_recognizer = None
                self.face_recognition_available = False
        else:
            self.face_recognizer = None
            print("⚠ cv2.face modul není k dispozici")
        
        self.known_faces = {}  # {name: [face_images]}
        self.face_labels = []  # List of names for training
        self.face_data = []    # List of face images for training
        self.is_trained = False
        
        # Load saved face data if exists
        self.load_face_data()
        
    def connect(self):
        """Pokusí se připojit ke kameře"""
        print(f"\n=== Pokus o připojení ke kameře ===")
        print(f"IP adresa: {self.ip_address}")
        print(f"Uživatel: {self.username}")
        print(f"Zkouším {len(self.stream_urls)} URL...\n")
        
        for i, stream_url in enumerate(self.stream_urls, 1):
            try:
                print(f"[{i}/{len(self.stream_urls)}] {stream_url}")
                
                # Zkusit přímé otevření URL s OpenCV
                self.cap = cv2.VideoCapture(stream_url)
                
                if self.cap.isOpened():
                    # Zkusit přečíst frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✓ ÚSPĚCH! Frame: {frame.shape}")
                        self.last_url = stream_url
                        self.connected = True
                        return True
                    else:
                        self.cap.release()
                        print(f"✗ Nelze čist frame")
                        continue
                    
            except Exception as e:
                print(f"✗ Chyba: {str(e)[:80]}")
                continue
        
        print("\n✗ Selhaly všechny URL")
        print("Pokus o připojení s autentizací...\n")
        
        # Pokud selžou všechny URL, pokusit se s autentizací
        return self.connect_with_auth()
    
    def connect_with_auth(self):
        """Pokus o připojení s HTTP autentizací"""
        try:
            # Test dostupnosti kamery na IP adrese
            test_url = self.ip_address
            response = requests.get(test_url, timeout=5)
            print(f"✓ Kamera je dostupná na {self.ip_address}")
            
            # Zkusit snapshot URL s autentizací
            snapshot_url = f'{self.ip_address.rstrip("/")}/snapshot.jpg'
            
            auth = HTTPBasicAuth(self.username, self.password)
            response = requests.get(snapshot_url, auth=auth, timeout=10)
            response.raise_for_status()
            
            print(f"✓ Snapshot je dostupný! Kamera je funkční.")
            self.connected = True
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"✗ Nelze dosáhnout IP adresy: {self.ip_address}")
            print(f"  Zkontrolujte IP adresu a síťové připojení")
            return False
        except Exception as e:
            print(f"✗ Chyba autentizace: {str(e)[:100]}")
            return False
    
    def get_frame(self):
        """Získat jeden snímek z kamery"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return frame
        return None
    
    def get_snapshot_http(self):
        """Získat snímek přes HTTP nebo z video streamu"""
        try:
            # Nejprve zkusit HTTP snapshot
            auth = HTTPBasicAuth(self.username, self.password)
            
            # Pro různé typy kamer různé snapshot URL
            if '/video' in self.ip_address:
                # Pro stream servery bez autentizace
                snapshot_url = self.ip_address.replace('/video', '/snapshot.jpg')
            else:
                # Standardní OD325HD kamera
                snapshot_url = f'{self.ip_address}/snapshot.jpg'
            
            response = requests.get(snapshot_url, auth=auth, timeout=5)
            response.raise_for_status()
            
            if len(response.content) > 1000:  # Pokud obsahuje data
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
            
            # Pokud HTTP snapshot nefunguje, použijeme frame z video streamu
            print("HTTP snapshot prázdný, používám frame z video streamu...")
            return self.get_frame_from_stream()
                
        except Exception as e:
            print(f"HTTP snapshot selhal ({e}), používám frame z video streamu...")
            return self.get_frame_from_stream()
    
    def get_frame_from_stream(self):
        """Získat aktuální frame z video streamu"""
        try:
            # Pokusit se otevřít stream
            for url in self.stream_urls:
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        return frame
            return None
        except Exception as e:
            print(f"Chyba při získávání frame ze streamu: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detekce obličejů v obraze pomocí Haar cascades"""
        if frame is None:
            return frame
            
        # Převedení na grayscale pro detekci obličejů
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detekce obličejů s přísnějšími parametry
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # Zvýšeno z 5 na 8 pro méně false positives
            minSize=(50, 50),  # Zvětšeno z 30x30 na 50x50
            maxSize=(300, 300),  # Přidáno omezení maximální velikosti
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filtrování detekovaných obličejů podle poměru stran
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Kontrola poměru stran - obličeje by měly být přibližně čtvercové
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:  # Poměr stran mezi 0.7 a 1.4
                # Kontrola velikosti - příliš malé nebo velké obličeje ignorujeme
                area = w * h
                if 2500 < area < 90000:  # Plocha mezi 50x50 a 300x300 pixely
                    filtered_faces.append((x, y, w, h))
        
        faces = filtered_faces
        
        faces_detected = 0
        # Nakreslení obdélníků kolem detekovaných obličejů
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Přidání textu
            cv2.putText(frame, 'Oblicej', (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            faces_detected += 1
        
        # Přidání počtu detekovaných obličejů do rohu obrazu
        cv2.putText(frame, f'Obliceje: {faces_detected}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame, faces_detected
    
    def recognize_faces(self, frame):
        """Rozpoznání obličejů v obraze"""
        if frame is None:
            return frame
            
        # If face recognition is not available, use basic detection
        if not self.face_recognition_available or not self.is_trained:
            return self.detect_faces(frame)
            
        # Převedení na grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detekce obličejů s přísnějšími parametry
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # Zvýšeno z 5 na 8 pro méně false positives
            minSize=(50, 50),  # Zvětšeno z 30x30 na 50x50
            maxSize=(300, 300),  # Přidáno omezení maximální velikosti
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filtrování detekovaných obličejů podle poměru stran
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Kontrola poměru stran - obličeje by měly být přibližně čtvercové
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:  # Poměr stran mezi 0.7 a 1.4
                # Kontrola velikosti - příliš malé nebo velké obličeje ignorujeme
                area = w * h
                if 2500 < area < 90000:  # Plocha mezi 50x50 a 300x300 pixely
                    filtered_faces.append((x, y, w, h))
        
        faces = filtered_faces
        
        faces_recognized = 0
        for (x, y, w, h) in faces:
            # Extrakce regionu obličeje
            face_roi = gray[y:y+h, x:x+w]
            
            # Pokus o rozpoznání
            try:
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Kontrola confidence - příliš nízká jistota = ignorujeme
                if confidence > 80:  # Zvýšeno z 50 na 80 pro přísnější filtrování
                    # Nezobrazujeme nejisté rozpoznání
                    continue
                
                # Získání jména podle labelu
                name = "Neznamy"
                if label < len(self.face_labels):
                    name = self.face_labels[label]
                
                # Barva podle confidence
                if confidence < 40:
                    color = (0, 255, 0)  # Zelená pro vysokou jistotu
                elif confidence < 60:
                    color = (0, 165, 255)  # Oranžová pro střední jistotu
                else:
                    color = (0, 0, 255)  # Červená pro nízkou jistotu
                
                # Nakreslení obdélníku
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Přidání textu s jménem a confidence
                text = f'{name} ({confidence:.1f})'
                cv2.putText(frame, text, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                faces_recognized += 1
                
            except Exception as e:
                # Pokud rozpoznání selže, použij základní detekci
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Oblicej', (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        # Přidání počtu rozpoznaných obličejů do rohu obrazu
        cv2.putText(frame, f'Obliceje: {faces_recognized}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame, faces_recognized
    
    def add_face_sample(self, name, face_image):
        """Přidání vzorku obličeje pro trénování"""
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(face_image)
        
        # Aktualizace trénovacích dat
        self.update_training_data()
    
    def update_training_data(self):
        """Aktualizace trénovacích dat z known_faces"""
        self.face_labels = []
        self.face_data = []
        
        for name, images in self.known_faces.items():
            for img in images:
                self.face_labels.append(name)
                self.face_data.append(img)
    
    def train_recognizer(self):
        """Trénování rozpoznávače obličejů"""
        if not self.face_recognition_available:
            print("Rozpoznávání obličejů není k dispozici")
            return False
            
        if len(self.face_data) < 2:
            print("Nedostatek dat pro trénování - potřebuji alespoň 2 vzorky")
            return False
            
        try:
            # Převedení labels na číselné hodnoty
            label_dict = {}
            numeric_labels = []
            current_id = 0
            
            for name in self.face_labels:
                if name not in label_dict:
                    label_dict[name] = current_id
                    current_id += 1
                numeric_labels.append(label_dict[name])
            
            # Trénování
            self.face_recognizer.train(self.face_data, np.array(numeric_labels))
            self.is_trained = True
            
            # Uložení dat
            self.save_face_data()
            
            print(f"Rozpoznávač natrénován s {len(self.face_data)} vzorky od {len(self.known_faces)} osob")
            return True
            
        except Exception as e:
            print(f"Chyba při trénování: {e}")
            return False
    
    def save_face_data(self):
        """Uložení dat o obličejích"""
        try:
            data = {
                'known_faces': self.known_faces,
                'is_trained': self.is_trained
            }
            
            with open('face_data.pkl', 'wb') as f:
                pickle.dump(data, f)
                
            # Uložení natrénovaného modelu
            if self.is_trained:
                self.face_recognizer.save('face_recognizer.yml')
                
        except Exception as e:
            print(f"Chyba při ukládání dat: {e}")
    
    def load_face_data(self):
        """Načtení dat o obličejích"""
        try:
            if os.path.exists('face_data.pkl'):
                with open('face_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                
                self.known_faces = data.get('known_faces', {})
                self.is_trained = data.get('is_trained', False)
                
                # Aktualizace trénovacích dat
                self.update_training_data()
                
                # Načtení natrénovaného modelu
                if self.is_trained and self.face_recognition_available and os.path.exists('face_recognizer.yml'):
                    self.face_recognizer.read('face_recognizer.yml')
                    
                print(f"Načteno {len(self.known_faces)} známých obličejů")
                
        except Exception as e:
            print(f"Chyba při načítání dat: {e}")
            self.known_faces = {}
            self.is_trained = False
    
    def export_face_database(self, filepath):
        """Export databáze obličejů do souboru"""
        print(f"DEBUG: export_face_database called with filepath='{filepath}'")
        try:
            data = {
                'known_faces': self.known_faces,
                'is_trained': self.is_trained,
                'export_date': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            # Export také trained modelu pokud existuje
            if self.is_trained and os.path.exists('face_recognizer.yml'):
                import shutil
                model_filepath = filepath.replace('.pkl', '_model.yml')
                shutil.copy2('face_recognizer.yml', model_filepath)
                
            return True
        except Exception as e:
            print(f"Chyba při exportu databáze: {e}")
            return False
    
    def import_face_database(self, filepath):
        """Import databáze obličejů ze souboru"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Kontrola verze
            version = data.get('version', '1.0')
            if version != '1.0':
                print(f"Varování: Verze databáze {version} nemusí být kompatibilní")
            
            # Načtení dat
            self.known_faces = data.get('known_faces', {})
            self.is_trained = data.get('is_trained', False)
            
            # Aktualizace trénovacích dat
            self.update_training_data()
            
            # Import trained modelu pokud existuje
            model_filepath = filepath.replace('.pkl', '_model.yml')
            if os.path.exists(model_filepath):
                import shutil
                shutil.copy2(model_filepath, 'face_recognizer.yml')
                if self.face_recognition_available:
                    self.face_recognizer.read('face_recognizer.yml')
            
            print(f"Importováno {len(self.known_faces)} obličejů")
            return True
            
        except Exception as e:
            print(f"Chyba při importu databáze: {e}")
            return False
    
    def cleanup(self):
        """Uklídnout zdroje"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def set_credentials(self, username, password):
        """Nastavit přihlašovací údaje"""
        self.username = username
        self.password = password


def main():
    """Hlavní funkce - spuštění prohlížeče kamery s GUI"""
    
    root = tk.Tk()
    root.title("Prohlížeč Kamery - Full HD 1920x1080 @ 15 FPS")
    # Maximalizace okna
    root.state('zoomed')
    root.resizable(True, True)
    
    app = KameraGUI(root)
    root.mainloop()


class KameraGUI:
    def __init__(self, root):
        self.root = root
        self.kamera = KameraOD325HD(
            ip_address="http://10.1.195.238:8080/video",
            username="admin",
            password="admin"
        )
        
        self.video_running = False
        self.thread = None
        self.face_recognition_enabled = tk.BooleanVar(value=False)  # Rozpoznávání obličejů je ve výchozím stavu vypnuté
        
        # Upravit styl
        style = ttk.Style()
        style.theme_use('clam')
        
        # Hlavní rámec
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Horní panel s informacemi
        info_frame = ttk.LabelFrame(main_frame, text="Informace", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text=f"IP Adresa: {self.kamera.ip_address}").pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(info_frame, text="Stav: Odpojen", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=20)
        
        # Rámec pro video
        video_frame = ttk.LabelFrame(main_frame, text="Video Stream", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Dolní panel s tlačítky
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.start_button = ttk.Button(button_frame, text="▶ Spustit Stream", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Zastavit Stream", command=self.stop_stream, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📷 Pořídit Snímek", command=self.save_snapshot).pack(side=tk.LEFT, padx=5)
        
        # Tlačítka pro správu obličejů
        self.add_face_button = ttk.Button(button_frame, text="➕ Přidat obličej", command=self.add_face)
        self.add_face_button.pack(side=tk.LEFT, padx=5)
        
        self.train_button = ttk.Button(button_frame, text="🎓 Trénovat", command=self.train_faces)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📤 Export DB", command=self.export_database).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📥 Import DB", command=self.import_database).pack(side=tk.LEFT, padx=5)
        
        # Disable face management buttons if face recognition is not available
        if not self.kamera.face_recognition_available:
            self.add_face_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.DISABLED)
        
        ttk.Button(button_frame, text="⚙️ Nastavení", command=self.show_settings).pack(side=tk.LEFT, padx=5)
        
        # Checkbox pro rozpoznávání obličejů
        self.face_recognition_checkbox = ttk.Checkbutton(button_frame, text="👤 Rozpoznávání obličejů", variable=self.face_recognition_enabled)
        self.face_recognition_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Disable face recognition if not available
        if not self.kamera.face_recognition_available:
            self.face_recognition_checkbox.config(state=tk.DISABLED)
            self.face_recognition_enabled.set(False)
        
        ttk.Button(button_frame, text="❌ Zavřít", command=self.close_app).pack(side=tk.RIGHT, padx=5)
        
        # Nastavení zavření okna
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
    def start_stream(self):
        """Başlat video stream"""
        if not self.video_running:
            self.video_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Spustit stream v samostatném vlákně
            self.thread = threading.Thread(target=self.stream_loop, daemon=True)
            self.thread.start()
    
    def stop_stream(self):
        """Zastavit video stream"""
        self.video_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_label.config(text="Stav: Zastaveno", foreground="orange")
    
    def stream_loop(self):
        """Hlavní loop pro čtení a zobrazení videa"""
        # Připojit ke kameře
        if not self.kamera.connect():
            error_msg = f"Nelze se připojit ke kameře!\nIP: {self.kamera.ip_address}\nUživatel: {self.kamera.username}"
            self.root.after(0, lambda: messagebox.showerror("Chyba připojení", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="Stav: Chyba připojení", foreground="red"))
            self.video_running = False
            self.root.after(0, self.stop_stream)
            return
        
        self.root.after(0, lambda: self.status_label.config(text="Stav: Připojeno", foreground="green"))
        
        fps_counter = 0
        last_time = datetime.now()
        error_count = 0
        
        while self.video_running and self.kamera.cap and self.kamera.cap.isOpened():
            try:
                frame = self.kamera.get_frame()
                
                if frame is None:
                    error_count += 1
                    if error_count > 30:  # Pokud se po chvíli nedaří brát frame
                        self.root.after(0, lambda: self.status_label.config(text="Stav: Chyba při čtení", foreground="red"))
                    continue
                
                error_count = 0
                
                # Rozpoznávání obličejů pokud je povoleno
                if self.face_recognition_enabled.get():
                    frame, faces_count = self.kamera.recognize_faces(frame)
                
                # Změnit velikost snímku pro display
                display_frame = cv2.resize(frame, (880, 540))
                
                # Převést na RGB
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Převést na PIL Image
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Aktualizovat label v main threadu
                self.root.after(0, lambda img=photo: self.update_video_label(img))
                
                fps_counter += 1
                current_time = datetime.now()
                elapsed = (current_time - last_time).total_seconds()
                
                if elapsed >= 1:
                    self.root.after(0, lambda fps=fps_counter: self.fps_label.config(text=f"FPS: {fps}"))
                    fps_counter = 0
                    last_time = current_time
                    
            except Exception as e:
                print(f"Chyba ve stream_loop: {str(e)}")
                continue
        
        self.kamera.cleanup()
        self.root.after(0, self.stop_stream)
    
    def update_video_label(self, photo):
        """Aktualizovat video label"""
        self.video_label.config(image=photo)
        self.video_label.image = photo
    
    def save_snapshot(self):
        """Uložit snímek z kamery"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        
        img = self.kamera.get_snapshot_http()
        
        if img is not None:
            cv2.imwrite(filename, img)
            messagebox.showinfo("Úspěch", f"Snímek uložen jako:\n{filename}")
        else:
            messagebox.showerror("Chyba", "Nelze pořídit snímek!")
    
    def show_settings(self):
        """Zobrazit dialog s nastavením"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Nastavení")
        settings_window.geometry("400x250")
        settings_window.resizable(False, False)
        
        # IP adresa
        ttk.Label(settings_window, text="IP Adresa:").pack(pady=5, padx=10, anchor=tk.W)
        ip_entry = ttk.Entry(settings_window, width=40)
        ip_entry.insert(0, self.kamera.ip_address)
        ip_entry.pack(pady=5, padx=10, anchor=tk.W)
        
        # Uživatelské jméno
        ttk.Label(settings_window, text="Uživatelské jméno:").pack(pady=5, padx=10, anchor=tk.W)
        user_entry = ttk.Entry(settings_window, width=40)
        user_entry.insert(0, self.kamera.username)
        user_entry.pack(pady=5, padx=10, anchor=tk.W)
        
        # Heslo
        ttk.Label(settings_window, text="Heslo:").pack(pady=5, padx=10, anchor=tk.W)
        pass_entry = ttk.Entry(settings_window, width=40, show="*")
        pass_entry.insert(0, self.kamera.password)
        pass_entry.pack(pady=5, padx=10, anchor=tk.W)
        
        def save_settings():
            self.kamera.ip_address = ip_entry.get()
            self.kamera.username = user_entry.get()
            self.kamera.password = pass_entry.get()
            messagebox.showinfo("Úspěch", "Nastavení uloženo!")
            settings_window.destroy()
        
        ttk.Button(settings_window, text="Uložit", command=save_settings).pack(pady=10)
    
    def add_face(self):
        """Přidání nového obličeje pro trénování"""
        if not self.kamera.face_recognition_available:
            messagebox.showerror("Nedostupné", "Rozpoznávání obličejů není k dispozici.\nNainstalujte opencv-contrib-python.")
            return
            
        if not self.video_running:
            messagebox.showwarning("Varování", "Nejdříve spusťte video stream!")
            return
        
        # Získání jména od uživatele
        name = simpledialog.askstring("Přidat obličej", "Zadejte jméno osoby:")
        if not name:
            return
        
        # Pořízení snímku
        frame = self.kamera.get_frame()
        if frame is None:
            messagebox.showerror("Chyba", "Nelze pořídit snímek!")
            return
        
        # Detekce obličejů ve snímku s přísnějšími parametry
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.kamera.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # Zvýšeno z 5 na 8
            minSize=(50, 50),  # Zvětšeno z 30x30 na 50x50
            maxSize=(300, 300),  # Přidáno omezení maximální velikosti
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filtrování detekovaných obličejů
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:  # Poměr stran
                area = w * h
                if 2500 < area < 90000:  # Velikost plochy
                    filtered_faces.append((x, y, w, h))
        
        faces = filtered_faces
        
        if len(faces) == 0:
            messagebox.showwarning("Žádný obličej", "V obraze nebyl detekován žádný obličej!")
            return
        
        if len(faces) > 1:
            messagebox.showwarning("Více obličejů", "V obraze je více obličejů. Pořiďte snímek s jedním obličejem!")
            return
        
        # Extrakce obličeje
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Přidání do databáze
        self.kamera.add_face_sample(name, face_resized)
        
        messagebox.showinfo("Úspěch", f"Obličej '{name}' byl přidán do databáze!")
    
    def train_faces(self):
        """Trénování rozpoznávače obličejů"""
        if not self.kamera.face_recognition_available:
            messagebox.showerror("Nedostupné", "Rozpoznávání obličejů není k dispozici.\nNainstalujte opencv-contrib-python.")
            return
            
        if len(self.kamera.face_data) < 2:
            messagebox.showwarning("Nedostatek dat", "Potřebujete alespoň 2 vzorky obličejů pro trénování!")
            return
        
        # Trénování v samostatném vlákně
        def train_thread():
            success = self.kamera.train_recognizer()
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Úspěch", "Rozpoznávač byl natrénován!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Chyba", "Trénování selhalo!"))
        
        threading.Thread(target=train_thread, daemon=True).start()
        messagebox.showinfo("Trénování", "Trénování probíhá... Může to chvíli trvat.")
    
    def export_database(self):
        """Export databáze obličejů"""
        if not self.kamera.face_recognition_available:
            messagebox.showerror("Nedostupné", "Rozpoznávání obličejů není k dispozici.")
            return
            
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Export databáze obličejů"
        )
        
        if filepath:
            success = self.kamera.export_face_database(filepath)
            if success:
                messagebox.showinfo("Úspěch", f"Databáze byla exportována do:\n{filepath}")
            else:
                messagebox.showerror("Chyba", "Export databáze selhal!")
    
    def import_database(self):
        """Import databáze obličejů"""
        if not self.kamera.face_recognition_available:
            messagebox.showerror("Nedostupné", "Rozpoznávání obličejů není k dispozici.")
            return
            
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Import databáze obličejů"
        )
        
        if filepath:
            success = self.kamera.import_face_database(filepath)
            if success:
                messagebox.showinfo("Úspěch", f"Databáze byla importována z:\n{filepath}")
                # Aktualizace GUI stavu
                self.update_face_buttons()
            else:
                messagebox.showerror("Chyba", "Import databáze selhal!")
    
    def update_face_buttons(self):
        """Aktualizace stavu tlačítek pro správu obličejů"""
        # Povolení/zakázání tlačítek podle dostupnosti dat
        if self.kamera.face_recognition_available and len(self.kamera.known_faces) > 0:
            self.train_button.config(state=tk.NORMAL)
        else:
            self.train_button.config(state=tk.DISABLED)
    
    def close_app(self):
        """Zavřít aplikaci"""
        self.video_running = False
        self.kamera.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    main()
