#!/usr/bin/env python3
"""
GUI aplikace pro ruční organizaci seskupených obličejů

Aplikace zobrazuje obsah složek skupina_XXX a umožňuje:
- Přejmenovat složku na příjmení osoby (přesune do 'osoby/')
- Smazat složku pokud obsahuje nechtěné obličeje

Použití:
python organize_groups.py
"""

import os
import shutil
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading

class GroupOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Organizace seskupených obličejů")
        self.root.geometry("1000x700")

        self.base_path = "osoby/skupiny"
        self.current_group_index = 0
        self.groups = []
        self.current_images = []
        self.image_thumbnails = []
        self.selected_images = set()  # Množina indexů vybraných obrázků

        self.setup_ui()
        self.load_groups()

    def setup_ui(self):
        """Nastavení uživatelského rozhraní"""
        # Hlavní frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Konfigurace grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Titulek
        title_label = ttk.Label(main_frame, text="Organizace seskupených obličejů",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Levý panel - seznam skupin
        left_frame = ttk.LabelFrame(main_frame, text="Skupiny", padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Seznam skupin
        self.groups_listbox = tk.Listbox(left_frame, height=20, width=25)
        self.groups_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.groups_listbox.bind('<<ListboxSelect>>', self.on_group_select)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.groups_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.groups_listbox.config(yscrollcommand=scrollbar.set)

        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Pravý panel - detail skupiny
        right_frame = ttk.LabelFrame(main_frame, text="Detail skupiny", padding="5")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Info o skupině
        self.info_text = tk.Text(right_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Canvas pro obrázky
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar pro obrázky
        self.images_canvas = tk.Canvas(self.canvas_frame, height=300)
        self.images_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL,
                                            command=self.images_canvas.yview)
        self.images_canvas.configure(yscrollcommand=self.images_scrollbar.set)

        self.images_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.images_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)

        # Akční tlačítka
        actions_frame = ttk.Frame(right_frame)
        actions_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Pole pro příjmení
        ttk.Label(actions_frame, text="Příjmení:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.surname_entry = ttk.Entry(actions_frame, width=20)
        self.surname_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # Tlačítka
        self.rename_button = ttk.Button(actions_frame, text="Přejmenovat",
                                       command=self.rename_group)
        self.rename_button.grid(row=0, column=2, padx=(0, 5))

        self.delete_button = ttk.Button(actions_frame, text="Smazat skupinu",
                                       command=self.delete_group)
        self.delete_button.grid(row=0, column=3, padx=(0, 5))

        self.delete_images_button = ttk.Button(actions_frame, text="Smazat vybrané obrázky",
                                              command=self.delete_selected_images)
        self.delete_images_button.grid(row=0, column=4, padx=(0, 5))

        self.skip_button = ttk.Button(actions_frame, text="Přeskočit",
                                     command=self.skip_group)
        self.skip_button.grid(row=0, column=5)

        actions_frame.columnconfigure(1, weight=1)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Připraveno")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

    def load_groups(self):
        """Načte seznam skupin"""
        self.groups = list_group_folders(self.base_path)
        self.groups_listbox.delete(0, tk.END)

        for group in self.groups:
            self.groups_listbox.insert(tk.END, group)

        if self.groups:
            self.groups_listbox.selection_set(0)
            self.on_group_select(None)
            self.update_status()
        else:
            self.status_label.config(text="Žádné skupiny nenalezeny")

    def on_group_select(self, event):
        """Handler pro výběr skupiny"""
        selection = self.groups_listbox.curselection()
        if selection:
            index = selection[0]
            group_name = self.groups[index]
            self.load_group_details(group_name)

    def load_group_details(self, group_name):
        """Načte detaily skupiny"""
        group_path = os.path.join(self.base_path, group_name)

        # Načtení obrázků
        self.current_images = []
        for filename in os.listdir(group_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.current_images.append(os.path.join(group_path, filename))

        # Vyčištění výběru pro novou skupinu
        self.selected_images.clear()

        # Aktualizace info textu
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Skupina: {group_name}\n")
        self.info_text.insert(tk.END, f"Počet obrázků: {len(self.current_images)}\n")
        self.info_text.insert(tk.END, f"Cesta: {group_path}")
        self.info_text.config(state=tk.DISABLED)

        # Načtení thumbnails
        self.load_thumbnails()

        # Aktualizace statusu
        self.update_status()

    def load_thumbnails(self):
        """Načte miniatury obrázků"""
        # Vyčištění canvas
        self.images_canvas.delete("all")
        self.image_thumbnails.clear()

        if not self.current_images:
            return

        # Vytvoření frame uvnitř canvas
        self.images_frame = ttk.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame, anchor="nw")

        # Načtení prvních několika obrázků jako thumbnails
        max_images = min(50, len(self.current_images))  # Max 50 thumbnails

        for i, image_path in enumerate(self.current_images[:max_images]):
            try:
                # Načtení a změna velikosti obrázku
                img = Image.open(image_path)
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)

                # Uložení reference
                self.image_thumbnails.append(photo)

                # Vytvoření frame pro obrázek s okrajem
                img_frame = ttk.Frame(self.images_frame, relief="raised", borderwidth=2)
                img_frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)

                # Nastavení barvy okraje podle výběru
                if i in self.selected_images:
                    img_frame.configure(relief="sunken", borderwidth=3)

                # Vytvoření label s obrázkem
                img_label = ttk.Label(img_frame, image=photo)
                img_label.grid(row=0, column=0)

                # Klikací funkce
                img_label.bind("<Button-1>", lambda e, idx=i: self.toggle_image_selection(idx))
                img_frame.bind("<Button-1>", lambda e, idx=i: self.toggle_image_selection(idx))

                # Tooltip s názvem souboru
                filename = os.path.basename(image_path)
                self.create_tooltip(img_label, f"{filename}\n(Klikněte pro výběr)")

            except Exception as e:
                # Pokud se obrázek nepodaří načíst
                error_frame = ttk.Frame(self.images_frame, relief="raised", borderwidth=2)
                error_frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)
                error_label = ttk.Label(error_frame, text="[Chyba]", foreground="red")
                error_label.grid(row=0, column=0)

        # Aktualizace scroll region
        self.images_frame.update_idletasks()
        self.images_canvas.config(scrollregion=self.images_canvas.bbox("all"))

    def toggle_image_selection(self, index):
        """Přepne výběr obrázku"""
        if index in self.selected_images:
            self.selected_images.remove(index)
        else:
            self.selected_images.add(index)

        # Aktualizace zobrazení
        self.load_thumbnails()

        # Aktualizace počtu vybraných v status baru
        self.update_status()

    def delete_selected_images(self):
        """Smaže vybrané obrázky"""
        if not self.selected_images:
            messagebox.showwarning("Varování", "Nevybrali jste žádné obrázky!")
            return

        count = len(self.selected_images)
        result = messagebox.askyesno("Potvrzení",
                                   f"Opravdu chcete smazat {count} vybraných obrázků?")
        if not result:
            return

        # Získání cesty ke skupině
        selection = self.groups_listbox.curselection()
        if not selection:
            return

        group_name = self.groups[selection[0]]
        group_path = os.path.join(self.base_path, group_name)

        deleted_count = 0
        for index in sorted(self.selected_images, reverse=True):
            if index < len(self.current_images):
                image_path = self.current_images[index]
                try:
                    os.remove(image_path)
                    deleted_count += 1
                except Exception as e:
                    messagebox.showerror("Chyba", f"Chyba při mazání {os.path.basename(image_path)}: {e}")

        # Vyčištění výběru
        self.selected_images.clear()

        # Aktualizace zobrazení
        self.load_group_details(group_path)
        self.update_status()

        messagebox.showinfo("Úspěch", f"Smazáno {deleted_count} obrázků")

    def update_status(self):
        """Aktualizuje status bar"""
        selected_count = len(self.selected_images)
        total_groups = len(self.groups)

        if selected_count > 0:
            self.status_label.config(text=f"Vybráno {selected_count} obrázků | Celkem {total_groups} skupin")
        else:
            self.status_label.config(text=f"Nalezeno {total_groups} skupin")

    def create_tooltip(self, widget, text):
        """Vytvoří tooltip pro widget"""
        def show_tooltip(event):
            self.tooltip = tk.Toplevel()
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = ttk.Label(self.tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
            label.pack()

        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def rename_group(self):
        """Přejmenuje skupinu na příjmení"""
        selection = self.groups_listbox.curselection()
        if not selection:
            messagebox.showwarning("Varování", "Vyberte skupinu!")
            return

        surname = self.surname_entry.get().strip()
        if not surname:
            messagebox.showwarning("Varování", "Zadejte příjmení!")
            return

        if not surname.replace(' ', '').replace('-', '').isalpha():
            messagebox.showerror("Chyba", "Příjmení může obsahovat pouze písmena, mezery a pomlčky!")
            return

        index = selection[0]
        group_name = self.groups[index]
        group_path = os.path.join(self.base_path, group_name)

        # Kontrola existence cílové složky
        osoby_path = "osoby"
        new_folder_path = os.path.join(osoby_path, surname)

        if os.path.exists(new_folder_path):
            result = messagebox.askyesno("Potvrzení",
                                       f"Složka '{surname}' už existuje. Sloučit?")
            if not result:
                return

        try:
            # Přesun a přejmenování
            os.makedirs(os.path.dirname(new_folder_path), exist_ok=True)
            shutil.move(group_path, new_folder_path)

            messagebox.showinfo("Úspěch",
                              f"Skupina '{group_name}' přejmenována na '{surname}'")

            # Aktualizace seznamu
            self.groups.pop(index)
            self.groups_listbox.delete(index)

            # Výběr další skupiny
            if self.groups:
                if index >= len(self.groups):
                    index = len(self.groups) - 1
                self.groups_listbox.selection_set(index)
                self.on_group_select(None)

            self.surname_entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba při přejmenovávání: {e}")

    def delete_group(self):
        """Smaže skupinu"""
        selection = self.groups_listbox.curselection()
        if not selection:
            messagebox.showwarning("Varování", "Vyberte skupinu!")
            return

        index = selection[0]
        group_name = self.groups[index]

        result = messagebox.askyesno("Potvrzení",
                                   f"Opravdu chcete smazat skupinu '{group_name}'?")
        if not result:
            return

        group_path = os.path.join(self.base_path, group_name)

        try:
            shutil.rmtree(group_path)
            messagebox.showinfo("Úspěch", f"Skupina '{group_name}' byla smazána")

            # Aktualizace seznamu
            self.groups.pop(index)
            self.groups_listbox.delete(index)

            # Výběr další skupiny
            if self.groups:
                if index >= len(self.groups):
                    index = len(self.groups) - 1
                self.groups_listbox.selection_set(index)
                self.on_group_select(None)

        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba při mazání: {e}")

    def skip_group(self):
        """Přeskočí na další skupinu"""
        selection = self.groups_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if index < len(self.groups) - 1:
            self.groups_listbox.selection_set(index + 1)
            self.on_group_select(None)

def list_group_folders(base_path="osoby/skupiny"):
    """Vrátí seznam všech skupina_XXX složek seřazených podle čísla"""
    if not os.path.exists(base_path):
        return []

    folders = []
    for item in os.listdir(base_path):
        if item.startswith("skupina_") and os.path.isdir(os.path.join(base_path, item)):
            try:
                # Extrakce čísla ze jména složky
                number = int(item.split("_")[1])
                folders.append((number, item))
            except (ValueError, IndexError):
                continue

    # Seřazení podle čísla
    folders.sort(key=lambda x: x[0])
    return [item[1] for item in folders]

def main():
    root = tk.Tk()
    app = GroupOrganizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()