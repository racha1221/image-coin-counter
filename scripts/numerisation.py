import cv2

def lire_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    return image