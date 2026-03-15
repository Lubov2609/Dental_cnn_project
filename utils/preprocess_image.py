import os
from PIL import Image

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

SIZE = (448, 224)

os.makedirs(PROCESSED_DIR, exist_ok=True)

for file in os.listdir(RAW_DIR):

   path = os.path.join(RAW_DIR, file)

   img = Image.open(path).convert("RGB")

   img = img.resize(SIZE)

   save_path = os.path.join(PROCESSED_DIR, file)

   img.save(save_path)

print("Images processed")
