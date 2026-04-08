#!/usr/bin/env python3
"""
Skript pro vytvoření databáze obličejů z obrázků

Použití:
python create_face_database.py

Skript automaticky hledá složku 'osoby' v aktuálním adresáři.

Očekávaná struktura:
osoby/
├── Novak/
│   ├── obrazek1.jpg
│   ├── obrazek2.jpg
│   └── ...
├── Svoboda/
│   ├── obrazek1.jpg
│   ├── obrazek2.jpg
│   └── ...
└── ...

Každá složka v 'osoby' je pojmenována podle příjmení a obsahuje fotky dané osoby.
"""

import cv2
import os
import sys
import pickle
from datetime import datetime

def detect_faces_in_image(image_path):
    """Detekce obličejů v obrázku"""
    # Načtení kaskády pro detekci obličejů
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Načtení obrázku
    img = cv2.imread(image_path)
    if img is None:
        print(f"Nelze načíst obrázek: {image_path}")
        return []

    # Převedení na grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekce obličejů s přísnějšími parametry
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,  # Zvýšeno z 5 na 8
        minSize=(50, 50),
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

    face_images = []
    for (x, y, w, h) in faces:
        # Extrakce regionu obličeje
        face_roi = gray[y:y+h, x:x+w]
        # Změna velikosti na standardní rozměr
        face_resized = cv2.resize(face_roi, (100, 100))
        face_images.append(face_resized)

    return face_images

def create_database_from_folder(folder_path):
    """Vytvoření databáze z obrázků v adresáři 'osoby'."""
    known_faces = {}

    print(f"Prohledávám složku: {folder_path}")
    for surname_folder in sorted(os.listdir(folder_path)):
        surname_path = os.path.join(folder_path, surname_folder)
        if not os.path.isdir(surname_path):
            continue

        person_name = surname_folder
        if person_name not in known_faces:
            known_faces[person_name] = []

        print(f"Zpracovávám příjmení: {person_name}")
        
        file_count = 0
        for filename in sorted(os.listdir(surname_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            image_path = os.path.join(surname_path, filename)
            faces = detect_faces_in_image(image_path)

            if len(faces) == 1:
                known_faces[person_name].append(faces[0])
                file_count += 1
                print(f"  ✓ Přidán obličej z {filename}")
            elif len(faces) > 1:
                print(f"  ⚠ V {filename} nalezeno více obličejů ({len(faces)}), přeskočeno")
            else:
                print(f"  ✗ V {filename} nebyl nalezen žádný obličej")
        
        print(f"  Celkem přidáno pro {person_name}: {file_count} obličejů")

    return known_faces


def save_database(known_faces, output_file="face_database.pkl"):
    """Uložení databáze do souboru"""
    data = {
        'known_faces': known_faces,
        'is_trained': False,  # Bude potřeba trénování
        'created_date': datetime.now().isoformat(),
        'version': '1.0'
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Databáze uložena do {output_file}")
    print(f"  Osoby: {list(known_faces.keys())}")
    print(f"  Celkem vzorků: {sum(len(faces) for faces in known_faces.values())}")

def main():
    # Automaticky hledáme složku 'osoby' v aktuálním adresáři
    folder_path = os.path.join(os.getcwd(), "osoby")

    if not os.path.exists(folder_path):
        print(f"Chyba: Složka '{folder_path}' neexistuje!")
        print(f"Ujistěte se, že máte složku 'osoby' v: {os.getcwd()}")
        sys.exit(1)

    print(f"Vytváření databáze obličejů z: {folder_path}")
    print("-" * 50)

    # Vytvoření databáze
    known_faces = create_database_from_folder(folder_path)

    if not known_faces:
        print("Chyba: Nebyly nalezeny žádné obličeje!")
        sys.exit(1)

    print("-" * 50)
    print("Shrnutí:")
    for person, faces in known_faces.items():
        print(f"  {person}: {len(faces)} vzorků")

    # Uložení databáze
    output_file = "face_database.pkl"
    save_database(known_faces, output_file)

    print("-" * 50)
    print("✓ Databáze vytvořena úspěšně!")
    print(f"  Pro použití v aplikaci použijte '📥 Import DB' a vyberte {output_file}")

if __name__ == "__main__":
    main()