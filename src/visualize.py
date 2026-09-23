"""ภาพประกอบสำหรับหน้าเว็บ: keypoints, คู่จุด inlier/outlier และตำแหน่งภาพบนพาโนรามา"""

from __future__ import annotations

import cv2
import numpy as np

from src.feature import ImageFeatures

# สี BGR ประจำภาพแต่ละภาพ (ภาพ 1, 2, 3, ...)
PALETTE = [
    (232, 121, 56), (80, 175, 76), (36, 146, 245), (180, 70, 160), (60, 76, 231),
    (196, 180, 50), (130, 100, 245), (50, 200, 200), (140, 140, 40), (90, 90, 190),
]
INLIER_COLOR = (80, 200, 80)
OUTLIER_COLOR = (60, 60, 230)


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def _thickness(image: np.ndarray) -> int:
    return max(1, round(max(image.shape[:2]) / 700))


def draw_keypoints(image: np.ndarray, features: ImageFeatures, max_points: int = 600) -> np.ndarray:
    """วาด keypoints ที่แรงที่สุด พร้อมขนาด (scale) และทิศทาง (orientation)"""
    strongest = sorted(features.keypoints, key=lambda keypoint: -keypoint.response)[:max_points]
    return cv2.drawKeypoints(image, strongest, None, (40, 220, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    inlier_mask: np.ndarray,
    show_outliers: bool = True,
    max_lines: int = 80,
) -> np.ndarray:
    """วางภาพสองภาพเคียงกัน เส้นเขียว = RANSAC inliers, เส้นแดง = outliers ที่ถูกตัดทิ้ง"""
    gap = 16
    first_height, first_width = first_image.shape[:2]
    second_height, second_width = second_image.shape[:2]
    canvas = np.full((max(first_height, second_height), first_width + gap + second_width, 3), 255, np.uint8)
    canvas[:first_height, :first_width] = first_image
    canvas[:second_height, first_width + gap:] = second_image
    # หรี่ภาพพื้นหลังลงเล็กน้อยให้เส้นคู่จุดเห็นชัด
    canvas = cv2.addWeighted(canvas, 0.7, np.full_like(canvas, 255), 0.3, 0)
    offset = np.array([first_width + gap, 0], np.float32)
    thickness = _thickness(canvas)

    def sample(indices: np.ndarray, limit: int) -> np.ndarray:
        if len(indices) <= limit:
            return indices
        return indices[np.linspace(0, len(indices) - 1, limit).astype(int)]

    inliers = np.flatnonzero(inlier_mask)
    outliers = np.flatnonzero(~inlier_mask)
    groups = [(sample(inliers, max_lines), INLIER_COLOR)]
    if show_outliers:
        groups.insert(0, (sample(outliers, max_lines // 2), OUTLIER_COLOR))
    for indices, color in groups:
        for index in indices:
            start = tuple(np.round(first_points[index]).astype(int))
            end = tuple(np.round(second_points[index] + offset).astype(int))
            cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)
            for point in (start, end):
                cv2.circle(canvas, point, thickness + 3, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point, thickness + 3, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_layout(result) -> np.ndarray:
    """แสดงว่าแต่ละส่วนของพาโนรามามาจากภาพไหน เส้นขาว = seam, กรอบเหลือง = บริเวณที่ crop"""
    base = result.panorama_uncropped.astype(np.float32)
    tint = np.zeros_like(base)
    labels = result.labels
    for order, warp in enumerate(result.warps):
        tint[labels == order] = color_for(warp.index)
    covered = labels >= 0
    layout = base.copy()
    layout[covered] = base[covered] * 0.6 + tint[covered] * 0.4
    layout = layout.astype(np.uint8)
    thickness = _thickness(layout)

    # seam = พิกเซลที่ label ต่างจากพิกเซลข้างเคียง (เฉพาะภายในพาโนรามา)
    seam = np.zeros(labels.shape, bool)
    seam[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
    seam[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] >= 0) & (labels[:-1, :] >= 0)
    seam = cv2.dilate(seam.astype(np.uint8), np.ones((thickness + 1, thickness + 1), np.uint8)) > 0
    layout[seam] = (255, 255, 255)

    for warp in result.warps:
        contours, _ = cv2.findContours(warp.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea) + np.array([warp.x, warp.y])
        color = color_for(warp.index)
        cv2.polylines(layout, [contour], True, color, thickness + 1, cv2.LINE_AA)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            radius = 14 * thickness
            cv2.circle(layout, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(layout, center, radius, (255, 255, 255), thickness, cv2.LINE_AA)
            text = str(warp.index + 1)
            scale = 0.6 * thickness
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 * thickness)
            origin = (center[0] - text_width // 2, center[1] + text_height // 2)
            cv2.putText(layout, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2 * thickness, cv2.LINE_AA)

    if result.crop_rect is not None:
        x, y, width, height = result.crop_rect
        cv2.rectangle(layout, (x, y), (x + width - 1, y + height - 1), (0, 220, 255), thickness + 1, cv2.LINE_AA)
    return layout


def seam_close_up(result, size: int = 360) -> tuple[int, int, int, int] | None:
    """เลือกกรอบรอบ seam ระหว่างภาพคู่ที่ซ้อนกันมากที่สุด (พิกัดบนภาพหลัง crop) ใช้ซูมดูรอยต่อ"""
    labels = result.labels
    if result.crop_rect is not None:
        x, y, width, height = result.crop_rect
        labels = labels[y:y + height, x:x + width]
    seam = np.zeros(labels.shape, bool)
    seam[:, 1:] = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
    ys, xs = np.nonzero(seam)
    if len(xs) == 0:
        return None
    # ใช้จุดกึ่งกลางของ seam เส้นที่ยาวที่สุด
    pair_ids = labels[ys, xs].astype(np.int32) * 100 + labels[ys, xs - 1].astype(np.int32)
    values, counts = np.unique(pair_ids, return_counts=True)
    chosen = pair_ids == values[np.argmax(counts)]
    center_x, center_y = int(np.median(xs[chosen])), int(np.median(ys[chosen]))
    height, width = labels.shape
    side = min(size, width, height)
    left = int(np.clip(center_x - side // 2, 0, width - side))
    top = int(np.clip(center_y - side // 2, 0, height - side))
    return left, top, side, side
