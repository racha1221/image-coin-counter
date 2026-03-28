import os
import cv2
from scripts.numerisation import lire_image
from scripts.pretraitements import appliquer_pretraitements
from scripts.segmentation import segmenter_cuivre
from scripts.post_traitements import nettoyer_masque_cuivre

INPUT_DIR = "data/validation"
OUTPUT_DIR = "outputs"

EXTENSIONS_VALIDES = (".jpg", ".jpeg", ".png", ".bmp")


def creer_dossier(path):
    os.makedirs(path, exist_ok=True)


def traiter_image(image_path):
    image = lire_image(image_path)
    image_flou, image_hsv = appliquer_pretraitements(image)

    masque_cuivre_brut = segmenter_cuivre(image_hsv)
    masque_cuivre_nettoye = nettoyer_masque_cuivre(masque_cuivre_brut)

    nom_fichier = os.path.splitext(os.path.basename(image_path))[0]
    dossier_sortie = os.path.join(OUTPUT_DIR, nom_fichier)
    creer_dossier(dossier_sortie)

    h, s, v = cv2.split(image_hsv)
    image_hsv_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_originale.jpg"), image)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_floue.jpg"), image_flou)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_hsv_visualisation.jpg"), image_hsv_bgr)

    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_canal_h.png"), h)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_canal_s.png"), s)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_canal_v.png"), v)

    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_masque_cuivre_brut.png"), masque_cuivre_brut)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_fichier}_masque_cuivre_nettoye.png"), masque_cuivre_nettoye)

    print(f"[OK] {nom_fichier} traité -> résultats dans {dossier_sortie}")


def main():
    creer_dossier(OUTPUT_DIR)

    fichiers = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(EXTENSIONS_VALIDES)
    ]

    if not fichiers:
        print("Aucune image trouvée dans", INPUT_DIR)
        return

    for nom_image in fichiers:
        chemin_image = os.path.join(INPUT_DIR, nom_image)

        try:
            traiter_image(chemin_image)
        except Exception as e:
            print(f"[ERREUR] {nom_image} -> {e}")


if __name__ == "__main__":
    main()