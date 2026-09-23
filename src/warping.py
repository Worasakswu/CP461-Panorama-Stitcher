"""ฉายภาพลงทรงกระบอก (cylindrical projection) และ warp ภาพลงบน canvas ของพาโนรามา"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WarpedImage:
    index: int  # ลำดับภาพที่ผู้ใช้อัปโหลด
    image: np.ndarray  # BGR uint8 เฉพาะส่วนที่ครอบคลุม (ROI)
    mask: np.ndarray  # uint8 0/255 ขนาดเท่า ROI
    x: int  # ตำแหน่งมุมบนซ้ายของ ROI บน canvas
    y: int

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)


def translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def to_cylindrical_points(points: np.ndarray, focal: float, size: tuple[int, int]) -> np.ndarray:
    """แปลงพิกัดจุดในภาพ (x, y) → พิกัดบนทรงกระบอก ใช้กับ keypoints โดยไม่ต้องหา feature ใหม่"""
    cx, cy = size[0] / 2, size[1] / 2
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    theta = np.arctan2(x, focal)
    height = y / np.sqrt(x**2 + focal**2)
    return np.stack([focal * theta + cx, focal * height + cy], axis=1).astype(np.float32)


def cylindrical_warp(image: np.ndarray, focal: float) -> tuple[np.ndarray, np.ndarray]:
    """ฉายภาพลงทรงกระบอกรัศมี = focal คืน (ภาพ, mask) ขนาดเท่าภาพเดิม"""
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    theta = (u - cx) / focal
    cylinder_height = (v - cy) / focal
    # กลับจากจุดบนทรงกระบอก (θ, h) ไปหาพิกัดบนภาพ: x = f·tanθ, y = f·h / cosθ
    map_x = (focal * np.tan(theta) + cx).astype(np.float32)
    map_y = (focal * cylinder_height / np.cos(theta) + cy).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(
        np.full((height, width), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return warped, mask


def canvas_from_transforms(
    transforms: list[np.ndarray],
    sizes: list[tuple[int, int]],
    max_pixels: int,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """หาขนาด canvas ที่ครอบทุกภาพ คืน (เมทริกซ์เลื่อน/ย่อ, (กว้าง, สูง), สเกลที่ย่อ)"""
    corners = []
    for matrix, (width, height) in zip(transforms, sizes):
        points = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(points, matrix).reshape(-1, 2))
    corners = np.concatenate(corners)
    min_x, min_y = np.floor(corners.min(axis=0))
    max_x, max_y = np.ceil(corners.max(axis=0))
    width, height = max_x - min_x, max_y - min_y

    # ถ้า canvas ใหญ่เกินหน่วยความจำที่ Streamlit Cloud รับได้ ให้ย่อผลลัพธ์ลงแทนการ error
    scale = min(1.0, float(np.sqrt(max_pixels / max(width * height, 1.0))))
    offset = np.diag([scale, scale, 1.0]) @ translation(-min_x, -min_y)
    size = (max(1, int(np.ceil(width * scale))), max(1, int(np.ceil(height * scale))))
    return offset, size, scale


def warp_to_canvas(
    index: int,
    image: np.ndarray,
    mask: np.ndarray,
    matrix: np.ndarray,
    canvas_size: tuple[int, int],
) -> WarpedImage:
    """Warp ภาพลงเฉพาะกรอบ (ROI) ที่ภาพครอบคลุมบน canvas เพื่อประหยัดหน่วยความจำ"""
    height, width = image.shape[:2]
    points = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    x0, y0 = np.maximum(np.floor(projected.min(axis=0)).astype(int), 0)
    x1 = min(int(np.ceil(projected[:, 0].max())) + 1, canvas_size[0])
    y1 = min(int(np.ceil(projected[:, 1].max())) + 1, canvas_size[1])
    roi_size = (max(1, x1 - x0), max(1, y1 - y0))
    local = translation(-x0, -y0) @ matrix

    warped = cv2.warpPerspective(image, local, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.warpPerspective(
        mask, local, roi_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    # กันขอบ 1 px ที่ interpolation ผสมกับพื้นหลัง
    warped_mask = cv2.erode(warped_mask, np.ones((3, 3), np.uint8))
    return WarpedImage(index, warped, warped_mask, int(x0), int(y0))
