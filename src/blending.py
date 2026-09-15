import cv2
import numpy as np

def blend_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # การ Warp พื้นฐาน (ส่งผลลัพธ์ตั้งต้นให้เว็บรันผ่านก่อน)
    panorama = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    panorama[0:h1, 0:w1] = img1
    return panorama
