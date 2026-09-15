import cv2
import numpy as np

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("จุดจับคู่ไม่เพียงพอในการคำนวณ Homography (ต้องมีอย่างน้อย 4 จุด)")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H, mask
