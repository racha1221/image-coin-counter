import cv2


def appliquer_pretraitements(image):
    image_flou = cv2.GaussianBlur(image, (5, 5), 0)
    image_hsv = cv2.cvtColor(image_flou, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(image_hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_corrige = clahe.apply(v)

    image_hsv_corrige = cv2.merge([h, s, v_corrige])

    return image_flou, image_hsv_corrige