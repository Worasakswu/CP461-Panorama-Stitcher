"""อ่าน/เขียนไฟล์ภาพ และย่อภาพให้อยู่ในขนาดที่ประมวลผลได้เร็ว"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image(data: bytes, max_side: int | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """แปลงไฟล์ภาพเป็น BGR uint8 และหมุนภาพตาม EXIF (ภาพจากมือถือ)

    คืนค่า (ภาพ, (กว้าง, สูง) ของไฟล์ต้นฉบับ)
    """
    try:
        with Image.open(io.BytesIO(data)) as pil_image:
            original_size = pil_image.size
            if max_side:
                # ให้ JPEG decoder ย่อภาพตั้งแต่ตอนอ่าน ช่วยให้ภาพ 12–48 MP อ่านเร็วขึ้นมาก
                pil_image.draft("RGB", (max_side, max_side))
            pil_image = ImageOps.exif_transpose(pil_image)
            rgb = np.asarray(pil_image.convert("RGB"))
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError("ไม่สามารถอ่านไฟล์นี้เป็นภาพได้ (รองรับ JPG, PNG, WEBP)") from error

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if max_side:
        image, _ = resize_to_max_side(image, max_side)
    return image, original_size


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """ย่อภาพให้ด้านที่ยาวที่สุดไม่เกิน max_side (ไม่ขยายภาพเล็ก)"""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image, scale


def encode_image(image: np.ndarray, extension: str = ".png", jpeg_quality: int = 95) -> bytes:
    """แปลงภาพ BGR เป็นไบต์ PNG/JPEG สำหรับปุ่มดาวน์โหลด"""
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if extension in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        raise ValueError(f"บันทึกภาพเป็น {extension} ไม่สำเร็จ")
    return buffer.tobytes()
