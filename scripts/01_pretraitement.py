import cv2
import os

# Dossiers
input_folder = "validation"  # images originales
output_folder = "preprocessed"
os.makedirs(output_folder, exist_ok=True)

scale_percent = 50  # réduire la taille pour accélérer le traitement

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        # 1️⃣ Redimensionner
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # 2️⃣ Convertir en gris
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 3️⃣ CLAHE pour uniformiser la lumière localement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)

        # 4️⃣ Flou léger pour réduire le bruit et les petits motifs
        blur = cv2.GaussianBlur(gray_clahe, (5,5), 0)

        # Sauvegarder l'image prétraitée
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, blur)
        print(f"[Prétraitement] {filename} traité")