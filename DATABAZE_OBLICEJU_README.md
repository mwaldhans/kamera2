# Databáze Obličejů - Návod k použití

## Přehled
Tato aplikace podporuje rozpoznávání obličejů pomocí algoritmu LBPH (Local Binary Patterns Histograms). Databáze obličejů se ukládá do souborů `face_data.pkl` a `face_recognizer.yml`.

## Vylepšení rozpoznávání (verze 2.0)

### 🔧 Optimalizované parametry detekce
Pro snížení false positives (rozpoznávání neživých předmětů jako obličejů) byly implementovány následující vylepšení:

- **minNeighbors**: Zvýšeno z 5 na 8 (přísnější filtrování)
- **minSize**: Zvětšeno z 30×30 na 50×50 pixelů
- **maxSize**: Přidáno omezení na 300×300 pixelů
- **Poměr stran**: Kontrola 0.7–1.4 (obličeje musí být přibližně čtvercové)
- **Velikost plochy**: 2500–90000 pixelů (50×50 až 300×300)
- **Confidence threshold**: Zvýšen na 80 pro LBPH rozpoznávač

### 🎨 Indikace jistoty rozpoznání
- 🟢 **Zelená** (confidence < 40): Vysoká jistota
- 🟠 **Oranžová** (confidence 40–60): Střední jistota  
- 🔴 **Červená** (confidence 60–80): Nízká jistota
- ⚫ **Ignorováno** (confidence > 80): Příliš nejisté, nezobrazuje se

## Jak nahrát databázi obličejů

### 1. Automatické načtení při spuštění
Aplikace automaticky načte databázi z `face_data.pkl` při spuštění, pokud soubor existuje.

### 2. Přidávání obličejů přes GUI
1. Spusťte aplikaci: `python kamera.py`
2. Klikněte na "▶ Spustit Stream"
3. Zaškrtněte "👤 Rozpoznávání obličejů"
4. Klikněte na "➕ Přidat obličej"
5. Zadejte jméno osoby
6. Ujistěte se, že v obraze je vidět pouze jeden obličej
7. Opakujte pro každou osobu (minimálně 2 vzorky na osobu)

### 3. Trénování rozpoznávače
Po přidání obličejů klikněte na "🎓 Trénovat" pro natrénování modelu.

### 4. Export/Import databáze
- **Export**: Klikněte na "📤 Export DB" pro uložení databáze do souboru
- **Import**: Klikněte na "📥 Import DB" pro načtení databáze ze souboru

## Vytvoření databáze z obrázků

Použijte skript `create_face_database.py` pro hromadné vytvoření databáze z obrázků:

### Struktura složky s podsložkami:
```
obrazky/
├── Jan_Novak/
│   ├── jan1.jpg
│   ├── jan2.jpg
│   └── jan3.jpg
├── Marie_Svobodova/
│   ├── marie1.jpg
│   └── marie2.jpg
```

### Struktura s pojmenovanými soubory:
```
obrazky/
├── Jan_Novak_001.jpg
├── Jan_Novak_002.jpg
├── Marie_Svobodova_001.jpg
└── Marie_Svobodova_002.jpg
```

### Použití:
```bash
python create_face_database.py /cesta/k/složce_s_obrázky
```

Skript vytvoří soubor `face_database.pkl`, který můžete importovat přes GUI.

## Formát databáze

Databáze je uložena jako Python pickle s následující strukturou:
```python
{
    'known_faces': {
        'Jméno_Osoby': [obrazek1, obrazek2, ...],  # numpy arrays 100x100
    },
    'is_trained': True/False,
    'created_date': '2024-01-01T12:00:00',
    'version': '1.0'
}
```

## Řešení problémů

### Rozpoznávání nefunguje
1. Zkontrolujte, zda je nainstalováno `opencv-contrib-python`
2. Ověřte, že máte alespoň 2 vzorky na osobu
3. Zkontrolujte, zda byl model natrénován

### Špatné rozpoznávání
1. Přidejte více vzorků obličejů (minimálně 5 na osobu)
2. Zajistěte dobré osvětlení při pořizování snímků
3. Vyhněte se úhlům pohledu, které výrazně mění vzhled obličeje

### Chyba při načítání databáze
1. Zkontrolujte, zda soubor `face_data.pkl` není poškozený
2. Zkuste smazat soubor a vytvořit databázi znovu
3. Ověřte kompatibilitu verzí (verze 1.0)

## Technické detaily

- **Algoritmus**: LBPH (Local Binary Patterns Histograms)
- **Velikost obličejů**: 100x100 pixelů (grayscale)
- **Detekce obličejů**: Haar cascades (haarcascade_frontalface_default.xml)
- **Minimální počet vzorků**: 2 na osobu pro trénování
- **Formát uložení**: Pickle pro data, YAML pro natrénovaný model