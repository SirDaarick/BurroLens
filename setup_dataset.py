import os
import shutil
import random
import kaggle
from tqdm import tqdm

# --- CONFIGURACIÓN PARA ARTIFACT ---
DATASET_NAME = "awsaf49/artifact-dataset"
RAW_DIR = "dataset_raw"       
FINAL_DIR = "dataset_final"   
SAMPLES_PER_CLASS = 50000     # Tomaremos 15k reales y 15k fakes (30k total)
SPLIT_RATIO = 0.8             # 80% train, 20% val

# Extensiones válidas para evitar archivos basura
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

def setup_directories():
    if os.path.exists(FINAL_DIR):
        print(f"La carpeta {FINAL_DIR} ya existe. Borrándola para empezar de cero...")
        shutil.rmtree(FINAL_DIR)
    
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(FINAL_DIR, split, label), exist_ok=True)

def download_dataset():
    # Verifica si ya se descargó para no bajar gigas de nuevo
    if os.path.exists(RAW_DIR) and len(os.listdir(RAW_DIR)) > 0:
        print("El dataset raw parece que ya existe. Saltando descarga.")
        return

    print(f"Iniciando descarga de: {DATASET_NAME}")
    print("Ve por un café, esto va a tardar (es un dataset grande)...")
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_NAME, path=RAW_DIR, unzip=True)
        print("Descarga y descompresión completada.")
    except Exception as e:
        print(f"Error en la descarga: {e}")
        print("Asegurate de que kaggle.json está en la carpeta y tienes espacio en disco.")

def get_image_paths(root_dir):
    """
    Recorre recursivamente las carpetas de ArtiFact para encontrar imágenes
    y clasificarlas según si están en una carpeta llamada 'real' o 'fake'.
    """
    real_paths = []
    fake_paths = []
    
    print("Escaneando carpetas (esto puede tomar un momento)...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in VALID_EXTS:
                full_path = os.path.join(root, file)
                
                # Heurística para ArtiFact: buscar 'real' o 'fake' en la ruta
                # Convertimos a minusculas para asegurar match
                path_lower = full_path.lower()
                
                # Prioridad: A veces las carpetas se llaman '0_real' o '1_fake'
                if 'real' in path_lower and 'fake' not in path_lower:
                    real_paths.append(full_path)
                elif 'fake' in path_lower:
                    fake_paths.append(full_path)
                    
    print(f"Total encontrado -> Reales: {len(real_paths)} | Fakes: {len(fake_paths)}")
    return real_paths, fake_paths

def process_images():
    real_imgs, fake_imgs = get_image_paths(RAW_DIR)
    
    if not real_imgs or not fake_imgs:
        print("ERROR: No se encontraron imágenes. Revisa que la descarga terminó bien.")
        return

    # Muestreo y copiado
    for label, all_paths in [('real', real_imgs), ('fake', fake_imgs)]:
        print(f"\nProcesando clase: {label}")
        
        # Mezclar y recortar
        random.shuffle(all_paths)
        selected = all_paths[:SAMPLES_PER_CLASS]
        
        split_idx = int(len(selected) * SPLIT_RATIO)
        train_set = selected[:split_idx]
        val_set = selected[split_idx:]
        
        # Función auxiliar para copiar
        def copy_set(file_list, split_name):
            dest_dir = os.path.join(FINAL_DIR, split_name, label)
            for src in tqdm(file_list, desc=f"Copiando a {split_name}/{label}"):
                filename = os.path.basename(src)
                # Evitar duplicados de nombre (al venir de subcarpetas diferentes pueden repetirse)
                # Agregamos un hash o prefijo simple si es necesario, 
                # pero por simplicidad aquí confiamos en names únicos o sobreescritura benigna.
                # Para evitar colisiones reales:
                unique_name = f"{random.randint(0,99999)}_{filename}"
                dst = os.path.join(dest_dir, unique_name)
                shutil.copy(src, dst)

        copy_set(train_set, 'train')
        copy_set(val_set, 'val')

if __name__ == "__main__":
    if os.path.exists("kaggle.json"):
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    
    setup_directories()
    download_dataset()
    process_images()
    
    print("\n✨ ¡LISTO! Dataset ArtiFact preparado en 'dataset_final'")