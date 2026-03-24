import os
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer

# Funcție pentru conversia coordonatelor GPS
def converteste_gps(valoare):
    try:
        grade = float(valoare[0])
        minute = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}, {minute:.2f}, {secunde:.2f}"
    except:
        return str(valoare)

class ScannerWorker(QThread):
    progres = Signal(int, int)      
    imagine_reparata = Signal(int)  
    finalizat = Signal()

    def __init__(self, cale_folder, folder_cache, cale_db):
        super().__init__()
        self.cale_folder = cale_folder
        self.folder_cache = folder_cache # Salvam calea primita din main.py
        self.cale_db = cale_db           # Salvam calea catre baza de date primita din main.py
        self.running = True
        self.model = None 
        
        # --- CONFIGURATIE PROMPTE ---
        self.categorii_config = {
            "Oameni": [
                "a photo of a person", 
                "a portrait of a man or a woman", 
                "a group of people standing together",
                "human faces and people"
            ],
            "Natura": [
                "a natural landscape with no buildings", 
                "mountains, forests, trees or lakes", 
                "a beautiful view of nature",
                "outdoor scenery with plants and sky"
            ],
            "Tehnologie": [
                "computer hardware and electronic components", 
                "printed circuit boards and microchips", 
                "electronic gadgets and smartphones",
                "modern technology and devices",
                "a stick of computer memory ram", 
                "pc hardware parts isolated on white background"
            ],
            "Documente": [
                "a screenshot of digital text", 
                "a scanned paper or a document", 
                "black text on a white background",
                "a page from a book or an article"
            ],
            "Arhitectura": [
                "city buildings and architecture", 
                "houses and residential buildings", 
                "urban street view and skyscrapers",
                "interior or exterior of a building"
            ],
            "Vehicule": [
                "cars, trucks or motorcycles on the road", 
                "airplanes, trains or boats", 
                "transportation vehicles",
                "a photo of a vehicle"
            ],
            "Animale": [
                "a photo of a pet like a dog or a cat", 
                "wildlife animals in their habitat", 
                "birds, insects or mammals",
                "a close up of an animal"
            ],
            "Mancare": [
                "a close-up of food on a plate", 
                "a delicious meal or snacks", 
                "culinary photography and cooking",
                "food ingredients and gourmet dishes"
            ],
            "Evenimente": [
                "people celebrating a party or a wedding", 
                "a decorated Christmas tree with lights", 
                "a social gathering or a holiday event",
                "a festive ceremony or a public event"
            ]
        }

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        # REPARAȚIA: Managerul primeste acum calea salvata la __init__
        db = ManagerBazaDate(self.cale_db) 
        
        if self.model is None:
            self.model = SentenceTransformer('clip-ViT-B-32')

        # Pregatire vectori categorii
        nume_categorii = list(self.categorii_config.keys())
        vectori_reprezentativi = []
        for nume in nume_categorii:
            v_prompte = self.model.encode(self.categorii_config[nume], normalize_embeddings=True)
            v_mediu = np.mean(v_prompte, axis=0)
            v_mediu = v_mediu / np.linalg.norm(v_mediu)
            vectori_reprezentativi.append(v_mediu)
        vectori_categorii = np.array(vectori_reprezentativi)

        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        fisiere_totale = []
        
        # --- FILTRU FOLDERE (Blacklist pentru C:) ---
        foldere_ignore = ["Windows", "Program Files", "Program Data", "AppData", "$Recycle.Bin", ".cache", "node_modules"]
        
        try:
            for radacina, directoare, fisiere_nume in os.walk(self.cale_folder):
                # Optimizare: spunem os.walk sa nu intre in folderele de sistem
                directoare[:] = [d for d in directoare if d not in foldere_ignore]
                
                if any(ign in radacina for ign in foldere_ignore):
                    continue

                for nume in fisiere_nume:
                    if nume.lower().endswith(formate):
                        cale_completa = os.path.join(radacina, nume).replace('\\', '/')
                        fisiere_totale.append(cale_completa)
            fisiere_totale.sort()
        except Exception as e:
            print(f"Eroare scanare: {e}"); return

        total = len(fisiere_totale)
        
        # ASIGURARE FOLDER CACHE
        if not os.path.exists(self.folder_cache): 
            os.makedirs(self.folder_cache)

        for i, cale_full in enumerate(fisiere_totale):
            if not self.running: break
            current = i + 1
            nume_fisier = os.path.basename(cale_full)
            existenta = db.cauta_dupa_cale(cale_full)

            # Sarim peste daca exista deja
            if existenta and existenta[10] and os.path.exists(existenta[10]) and existenta[12] is not None:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    # Filtru rezolutie (ignora iconite)
                    if img.width < 200 or img.height < 200:
                        self.progres.emit(current, total)
                        continue
                        
                    img_fix = ImageOps.exif_transpose(img)
                    if img_fix.mode != "RGB": img_fix = img_fix.convert("RGB")
                    
                    # Analiza AI
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    scoruri = np.dot(vectori_categorii, vector_ai)
                    idx = np.argmax(scoruri)
                    # Am pastrat pragul de 0.23 pentru acuratete
                    categorie_finala = nume_categorii[idx] if scoruri[idx] > 0.21 else "Diverse"

                    # Salvare Thumbnail in AppData
                    nume_cache = f"cache_{current}_{nume_fisier}.png"
                    cale_cache_finala = os.path.join(self.folder_cache, nume_cache).replace('\\', '/')
                    
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((256, 256)) 
                    img_thumb.save(cale_cache_finala, "PNG")

                    date_info = {
                        'cale': cale_full, 'nume': nume_fisier, 'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache_finala, 'vector_ai': vector_ai, 'categorie': categorie_finala
                    }
                    
                    # Extragere EXIF (Data, GPS, etc.)
                    exif = img._getexif()
                    if exif:
                        for id_tag, val in exif.items():
                            n_tag = TAGS.get(id_tag, id_tag)
                            if n_tag == "Make": date_info['marca'] = val
                            if n_tag == "Model": date_info['model'] = val
                            if n_tag == "DateTimeOriginal": date_info['data'] = val
                            if n_tag == "GPSInfo":
                                info_g = {}
                                for t in val:
                                    s_tag = GPSTAGS.get(t, t)
                                    info_g[s_tag] = val[t]
                                if "GPSLatitude" in info_g:
                                    lat = converteste_gps(info_g["GPSLatitude"])
                                    lon = converteste_gps(info_g["GPSLongitude"])
                                    date_info['gps'] = f"Lat: {lat} | Lon: {lon}"

                    db.salveaza_sau_actualizeaza(date_info)
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare procesare {cale_full}: {e}")

            self.progres.emit(current, total)
        self.finalizat.emit()