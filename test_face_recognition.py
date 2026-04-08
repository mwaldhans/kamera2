import cv2
import numpy as np
import os
import sys

def test_face_recognition():
    print("=== Testování rozpoznávání obličejů ===")
    print(f"OpenCV verze: {cv2.__version__}")

    # Test 1: Kontrola cv2.face modulu
    print("\n1. Test cv2.face modulu:")
    if hasattr(cv2, 'face'):
        print("✓ cv2.face modul je k dispozici")
    else:
        print("✗ cv2.face modul není k dispozici")
        print("Nainstalujte: pip install opencv-contrib-python")
        return False

    # Test 2: Vytvoření LBPH rozpoznávače
    print("\n2. Test LBPH rozpoznávače:")
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✓ LBPH rozpoznávač vytvořen úspěšně")
    except Exception as e:
        print(f"✗ Chyba při vytváření rozpoznávače: {e}")
        return False

    # Test 3: Načtení Haar cascade
    print("\n3. Test Haar cascade klasifikátoru:")
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        print(f"Cesta k cascade: {cascade_path}")

        if os.path.exists(cascade_path):
            print("✓ Cascade soubor existuje")
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                print("✗ Haar cascade se nepodařilo načíst (empty)")
                return False
            else:
                print("✓ Haar cascade načten úspěšně")
        else:
            print("✗ Cascade soubor neexistuje")
            print("Dostupné soubory v adresáři:")
            haarcascades_dir = cv2.data.haarcascades
            if os.path.exists(haarcascades_dir):
                files = os.listdir(haarcascades_dir)
                for file in files[:10]:  # Zobraz prvních 10 souborů
                    print(f"  - {file}")
            return False
    except Exception as e:
        print(f"✗ Chyba při načítání cascade: {e}")
        return False

    # Test 4: Test detekce na prázdném obraze s přísnějšími parametry
    print("\n4. Test detekce obličejů:")
    try:
        # Vytvoření testovacího obrazu
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(50, 50),
            maxSize=(300, 300)
        )
        print(f"✓ Detekce funguje (našlo {len(faces)} obličejů v prázdném obraze)")
    except Exception as e:
        print(f"✗ Chyba při detekci: {e}")
        return False

    print("\n🎉 Všechny testy prošly úspěšně!")
    print("Rozpoznávání obličejů by mělo fungovat.")
    return True

if __name__ == "__main__":
    success = test_face_recognition()
    sys.exit(0 if success else 1)