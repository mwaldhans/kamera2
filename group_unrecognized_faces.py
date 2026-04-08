#!/usr/bin/env python3
"""
Skript pro roztřídění nerozpoznaných obličejů do skupin

Skript načte všechny obrázky z složky 'osoby/nerozpoznana/',
detekuje v nich obličeje a seskupí podobné obličeje do nových složek.

Použití:
python group_unrecognized_faces.py

Výstup:
Vytvoří nové složky ve formátu 'skupina_001/', 'skupina_002/', atd.
každá obsahující podobné obličeje.
"""

import cv2
import os
import sys
import shutil
from datetime import datetime
import numpy as np

def detect_face_in_image(image_path):
    """Detekce obličeje v obrázku (očekává se jeden obličej)"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)
    if img is None:
        print(f"Nelze načíst obrázek: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekce obličejů s přísnějšími parametry
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(50, 50),
        maxSize=(300, 300),
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

    if len(filtered_faces) == 1:
        x, y, w, h = filtered_faces[0]
        face_roi = gray[y:y+h, x:x+w]
        # Změna velikosti na standardní rozměr
        face_resized = cv2.resize(face_roi, (100, 100))
        return face_resized
    elif len(filtered_faces) > 1:
        print(f"[!] V {image_path} nalezeno vice obličeju ({len(filtered_faces)}), preskoceno")
        return None
    else:
        print(f"[X] V {image_path} nebyl nalezen zadny obličej")
        return None

def compute_face_histogram(face_image):
    """Výpočet histogramu pro obličej"""
    # Výpočet histogramu s 256 bins
    hist = cv2.calcHist([face_image], [0], None, [256], [0, 256])
    # Normalizace histogramu
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_faces(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """Porovnání dvou histogramů obličejů"""
    return cv2.compareHist(hist1, hist2, method)

def group_similar_faces(face_data, similarity_threshold=0.7):
    """
    Seskupení podobných obličejů na základě podobnosti histogramů

    Args:
        face_data: Seznam slovníků s 'path', 'face', 'hist'
        similarity_threshold: Práh podobnosti (0-1, vyšší = přísnější)

    Returns:
        Seznam skupin, každá skupina je seznam cest k obrázkům
    """
    groups = []
    used = set()

    for i, face1 in enumerate(face_data):
        if i in used:
            continue

        current_group = [face1['path']]
        used.add(i)

        for j, face2 in enumerate(face_data):
            if j in used or j == i:
                continue

            # Porovnání histogramů
            similarity = compare_faces(face1['hist'], face2['hist'])

            if similarity >= similarity_threshold:
                current_group.append(face2['path'])
                used.add(j)

        if len(current_group) > 1:  # Jen skupiny s více než 1 obličejem
            groups.append(current_group)

    return groups

def create_group_folders(groups, base_folder="osoby", delete_ungrouped=False):
    """Vytvoření složek pro skupiny a přesun obrázků"""
    groups_folder = os.path.join(base_folder, "skupiny")
    os.makedirs(groups_folder, exist_ok=True)

    print(f"\nVytváření {len(groups)} skupin ve složce: {groups_folder}")

    # Sestavení množiny seskupených obrázků
    grouped_images = set()
    for group in groups:
        grouped_images.update(group)

    for i, group in enumerate(groups, 1):
        group_name = f"skupina_{i:03d}"
        group_path = os.path.join(groups_folder, group_name)
        os.makedirs(group_path, exist_ok=True)

        print(f"Skupina {group_name}: {len(group)} obličejů")

        for image_path in group:
            filename = os.path.basename(image_path)
            new_path = os.path.join(group_path, filename)
            shutil.move(image_path, new_path)
            print(f"  [OK] Presunuto: {filename}")

    # Smazání neseskupených obrázků
    if delete_ungrouped:
        unrecognized_folder = os.path.join(base_folder, "nerozpoznana")
        deleted_count = 0

        for filename in os.listdir(unrecognized_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(unrecognized_folder, filename)
                if image_path not in grouped_images:
                    os.remove(image_path)
                    deleted_count += 1
                    print(f"  [DEL] Smazano neseskupene: {filename}")

        print(f"Smazano {deleted_count} neseskupenych obrazku")

def main():
    # Cesta ke složce s nerozpoznanými obličeji
    unrecognized_folder = os.path.join("osoby", "nerozpoznana")

    if not os.path.exists(unrecognized_folder):
        print(f"Chyba: Složka '{unrecognized_folder}' neexistuje!")
        sys.exit(1)

    print("=== Roztřídění nerozpoznaných obličejů ===")
    print(f"Zdrojová složka: {unrecognized_folder}")
    print("-" * 50)

    # Načtení všech obrázků
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []

    for filename in os.listdir(unrecognized_folder):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(unrecognized_folder, filename))

    if not image_files:
        print("Žádné obrázky nenalezeny!")
        sys.exit(1)

    print(f"Nalezeno {len(image_files)} obrázků")

    # Zpracování obrázků a detekce obličejů
    face_data = []

    for image_path in image_files:
        print(f"Zpracovávám: {os.path.basename(image_path)}")
        face = detect_face_in_image(image_path)

        if face is not None:
            hist = compute_face_histogram(face)
            face_data.append({
                'path': image_path,
                'face': face,
                'hist': hist
            })
            print("  [OK] Obličej detekovan a zpracovan")
        else:
            print("  [X] Obličej nebyl detekovan")

    if len(face_data) < 2:
        print("Mene nez 2 obličeje pro seskupeni - mazani vsech zbyvajicich obrazku")
        # Smazání všech zbývajících obrázků
        deleted_count = 0
        for image_path in image_files:
            os.remove(image_path)
            deleted_count += 1
            print(f"[DEL] Smazano: {os.path.basename(image_path)}")
        print(f"Celkem smazano {deleted_count} obrazku")
        sys.exit(0)

    print(f"\nZpracováno {len(face_data)} obličejů z {len(image_files)} obrázků")

    # Seskupení podobných obličejů
    print("\nSeskupování podobných obličejů...")
    groups = group_similar_faces(face_data, similarity_threshold=0.7)

    if not groups:
        print("Žádné podobné obličeje nenalezeny!")
        print("Zkuste snížit práh podobnosti.")
        sys.exit(1)

    print(f"Nalezeno {len(groups)} skupin podobných obličejů")

    # Vytvoření složek a přesun obrázků
    create_group_folders(groups, delete_ungrouped=True)

    # Smazání všech zbývajících obrázků v nerozpoznana složce
    unrecognized_folder = os.path.join("osoby", "nerozpoznana")
    remaining_files = [f for f in os.listdir(unrecognized_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    deleted_count = 0

    for filename in remaining_files:
        filepath = os.path.join(unrecognized_folder, filename)
        os.remove(filepath)
        deleted_count += 1
        print(f"[DEL] Smazano zbyvajici: {filename}")

    print(f"Celkem smazano {deleted_count} zbyvajicich obrazku")

    # Shrnutí
    print("\n" + "=" * 50)
    print("SHRNUTÍ:")
    print(f"Celkem obrázků: {len(image_files)}")
    print(f"Zpracovaných obličejů: {len(face_data)}")
    print(f"Vytvořených skupin: {len(groups)}")
    print(f"Celkem seskupených obrázků: {sum(len(g) for g in groups)}")

    remaining_images = len(image_files) - sum(len(g) for g in groups)
    if remaining_images > 0:
        print(f"Zbývajících osamocených obrázků: {remaining_images}")

    print("\n[OK] Roztrideni dokončeno!")

if __name__ == "__main__":
    main()