import cv2
import numpy as np
from config.params import (
    CUIVRE_H_MIN, CUIVRE_H_MAX,
    CUIVRE_S_MIN, CUIVRE_S_MAX,
    CUIVRE_V_MIN, CUIVRE_V_MAX
)

def segmenter_cuivre(image_hsv):
    borne_basse = np.array(
        [CUIVRE_H_MIN, CUIVRE_S_MIN, CUIVRE_V_MIN],
        dtype=np.uint8
    )
    borne_haute = np.array(
        [CUIVRE_H_MAX, CUIVRE_S_MAX, CUIVRE_V_MAX],
        dtype=np.uint8
    )

    masque_cuivre = cv2.inRange(image_hsv, borne_basse, borne_haute)

    return masque_cuivre