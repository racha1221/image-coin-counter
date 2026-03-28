import cv2

def nettoyer_masque_cuivre(masque_cuivre):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masque_nettoye = cv2.morphologyEx(masque_cuivre, cv2.MORPH_CLOSE, kernel_close)

    return masque_nettoye