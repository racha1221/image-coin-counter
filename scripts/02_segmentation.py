import cv2
import numpy as np
import os

input_folder = "preprocessed"
output_folder = "masks"
os.makedirs(output_folder, exist_ok=True)

# Paramètres K-means
k = 3  # fond clair, fond foncé, pièces
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # CLAHE sur canal V
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])

        # K-means sur HSV
        pixels = hsv.reshape((-1,3))
        pixels = np.float32(pixels)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = labels.reshape((hsv.shape[0], hsv.shape[1]))
        centers = np.uint8(centers)

        # Identifier le cluster pièces (teinte intermédiaire, saturation > 20)
        sat_means = [np.mean(hsv[:,:,1][segmented==i]) for i in range(k)]
        hue_means = [np.mean(hsv[:,:,0][segmented==i]) for i in range(k)]
        # Pièces = cluster avec saturation intermédiaire
        piece_cluster = np.argmax(sat_means)

        # Masque binaire
        mask = np.where(segmented == piece_cluster, 255, 0).astype(np.uint8)

        # Morphologie
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Sauvegarder
        out_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(out_path, mask)
        print(f"[Segmentation pièces métalliques] {filename} traité")