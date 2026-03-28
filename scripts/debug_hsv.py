import cv2
from scripts.numerisation import lire_image
from scripts.pretraitements import appliquer_pretraitements
from config.params import IMAGE_PATH

def afficher_hsv_au_clic(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image_hsv = param["image_hsv"]
        h, s, v = image_hsv[y, x]
        print(f"Pixel ({x}, {y}) -> H={h}, S={s}, V={v}")

def main():
    image = lire_image(IMAGE_PATH)
    _, image_hsv = appliquer_pretraitements(image)

    image_affichage = image.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback(
        "Image",
        afficher_hsv_au_clic,
        {"image_hsv": image_hsv}
    )

    while True:
        cv2.imshow("Image", image_affichage)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # touche Echap
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()