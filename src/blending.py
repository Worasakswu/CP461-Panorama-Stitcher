"""ปรับแสง (gain compensation) รวมภาพแบบไร้รอยต่อ และตัดขอบดำ"""

from __future__ import annotations

import cv2
import numpy as np

from src.warping import WarpedImage

BLEND_METHODS = ("multiband", "feather", "none")


def _overlap(first: WarpedImage, second: WarpedImage):
    """คืน slices ของส่วนที่ ROI สองภาพซ้อนกัน (พิกัด ROI ของแต่ละภาพ) หรือ None"""
    x0, y0 = max(first.x, second.x), max(first.y, second.y)
    x1 = min(first.x + first.width, second.x + second.width)
    y1 = min(first.y + first.height, second.y + second.height)
    if x1 <= x0 or y1 <= y0:
        return None
    first_slices = (slice(y0 - first.y, y1 - first.y), slice(x0 - first.x, x1 - first.x))
    second_slices = (slice(y0 - second.y, y1 - second.y), slice(x0 - second.x, x1 - second.x))
    return first_slices, second_slices


def overlap_pixels(first: WarpedImage, second: WarpedImage) -> int:
    overlap = _overlap(first, second)
    if overlap is None:
        return 0
    return int(np.count_nonzero((first.mask[overlap[0]] > 0) & (second.mask[overlap[1]] > 0)))


def distance_to_edge(mask: np.ndarray) -> np.ndarray:
    """ระยะ (px) จากแต่ละพิกเซลถึงขอบภาพ โดยถือว่านอก ROI ไม่มีข้อมูล"""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]


def gain_compensation(
    warps: list[WarpedImage], sigma_noise: float = 10.0, sigma_gain: float = 0.3, stride: int = 2
) -> np.ndarray:
    """หาค่า gain ต่อช่องสีของแต่ละภาพให้ความสว่างในส่วนซ้อนทับใกล้กัน (Brown & Lowe, 2007)

    แก้ปัญหา least squares: Σ N_ij [(g_i·I_ij − g_j·I_ji)² / σN² + (1 − g_i)² / σg²]
    ใช้ σg = 0.3 (หลวมกว่า 0.1 ในบทความ) เพราะกล้องมือถือปรับแสงอัตโนมัติต่างกันได้เกิน 10%
    แล้ว normalize ให้ gain เฉลี่ยเป็น 1 เพื่อคงความสว่างรวมของพาโนรามา คืน array (จำนวนภาพ, 3)
    """
    count = len(warps)
    counts = np.zeros((count, count))
    means = np.zeros((count, count, 3))  # means[i, j] = สีเฉลี่ยของภาพ i ในส่วนที่ซ้อนกับภาพ j
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            overlap = _overlap(warps[i], warps[j])
            if overlap is None:
                continue
            first_slices, second_slices = overlap
            both = (warps[i].mask[first_slices][::stride, ::stride] > 0) & (
                warps[j].mask[second_slices][::stride, ::stride] > 0
            )
            pixels = np.count_nonzero(both)
            if pixels == 0:
                continue
            counts[i, j] = pixels
            means[i, j] = warps[i].image[first_slices][::stride, ::stride][both].mean(axis=0)

    alpha, beta = 1.0 / sigma_noise**2, 1.0 / sigma_gain**2
    gains = np.ones((count, 3))
    for channel in range(3):
        # ภาพที่ไม่ซ้อนกับภาพใดเลยจะได้ gain = 1 จากพจน์ prior ที่เติมไว้ 1 พิกเซล
        system = np.eye(count) * beta
        target = np.full(count, beta)
        for i in range(count):
            for j in range(count):
                if i == j or counts[i, j] == 0:
                    continue
                target[i] += beta * counts[i, j]
                system[i, i] += beta * counts[i, j] + 2 * alpha * means[i, j, channel] ** 2 * counts[i, j]
                system[i, j] -= 2 * alpha * means[i, j, channel] * means[j, i, channel] * counts[i, j]
        gains[:, channel] = np.linalg.solve(system, target)
    areas = np.array([np.count_nonzero(warp.mask[::stride, ::stride]) for warp in warps], np.float64)
    gains /= np.average(gains, axis=0, weights=np.maximum(areas, 1.0))
    return np.clip(gains, 0.5, 2.0)


def apply_gains(warps: list[WarpedImage], gains: np.ndarray | None) -> list[WarpedImage]:
    if gains is None:
        return warps
    adjusted = []
    for warp, gain in zip(warps, gains):
        image = np.clip(warp.image.astype(np.float32) * gain.astype(np.float32), 0, 255).astype(np.uint8)
        adjusted.append(WarpedImage(warp.index, image, warp.mask, warp.x, warp.y))
    return adjusted


def seam_labels(warps: list[WarpedImage], canvas_size: tuple[int, int]) -> np.ndarray:
    """กำหนดว่าแต่ละพิกเซลเป็นของภาพไหน โดยเลือกภาพที่พิกเซลนั้นอยู่ลึกจากขอบภาพมากที่สุด

    ได้รอยต่อ (seam) อยู่กลางส่วนซ้อนทับ คืน label map (-1 = ไม่มีภาพ, k = ลำดับใน warps)
    """
    width, height = canvas_size
    labels = np.full((height, width), -1, np.int16)
    best = np.zeros((height, width), np.float32)
    for order, warp in enumerate(warps):
        distance = distance_to_edge(warp.mask)
        region_best = best[warp.slices]
        better = distance > region_best
        region_best[better] = distance[better]
        labels[warp.slices][better] = order
    return labels


def overlay_blend(warps: list[WarpedImage], canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """วางภาพทับกันตามลำดับ (แบบโค้ดเดิม) ใช้เป็น baseline ให้เห็นรอยต่อ"""
    width, height = canvas_size
    panorama = np.zeros((height, width, 3), np.uint8)
    union = np.zeros((height, width), np.uint8)
    for warp in warps:
        valid = warp.mask > 0
        panorama[warp.slices][valid] = warp.image[valid]
        union[warp.slices][valid] = 255
    return panorama, union


def feather_blend(warps: list[WarpedImage], canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """เฉลี่ยถ่วงน้ำหนักตามระยะห่างจากขอบภาพ (distance-weighted feathering)"""
    width, height = canvas_size
    accumulated = np.zeros((height, width, 3), np.float32)
    weights = np.zeros((height, width), np.float32)
    for warp in warps:
        distance = distance_to_edge(warp.mask)
        accumulated[warp.slices] += warp.image.astype(np.float32) * distance[..., None]
        weights[warp.slices] += distance
    panorama = accumulated / np.maximum(weights, 1e-6)[..., None]
    union = np.where(weights > 0, 255, 0).astype(np.uint8)
    panorama[union == 0] = 0
    return np.clip(panorama + 0.5, 0, 255).astype(np.uint8), union


def _fill_invalid(image: np.ndarray, valid: np.ndarray, levels: int) -> np.ndarray:
    """เติมสีจากขอบภาพออกไปในบริเวณที่ไม่มีข้อมูล (push–pull) เพื่อไม่ให้ pyramid ดึงสีดำเข้ามา"""
    if valid.all():
        return image
    weight = valid.astype(np.float32)
    colors = [image * weight[..., None]]
    weights = [weight]
    for _ in range(levels):
        colors.append(cv2.pyrDown(colors[-1]))
        weights.append(cv2.pyrDown(weights[-1]))
    filled = colors[-1] / np.maximum(weights[-1], 1e-6)[..., None]
    for level in range(levels - 1, -1, -1):
        height, width = colors[level].shape[:2]
        upsampled = cv2.pyrUp(filled, dstsize=(width, height))
        level_weight = weights[level][..., None]
        own = colors[level] / np.maximum(level_weight, 1e-6)
        alpha = np.clip(level_weight, 0.0, 1.0)
        filled = own * alpha + upsampled * (1.0 - alpha)
    return np.where(valid[..., None], image, filled)


def _extend_labels(labels: np.ndarray) -> np.ndarray:
    """ให้พิกเซลที่ไม่มีภาพ (-1) ใช้ label ของพิกเซลที่มีภาพซึ่งใกล้ที่สุด"""
    empty = (labels < 0).astype(np.uint8)
    if not empty.any() or empty.all():
        return labels
    _, nearest = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL ให้เลข 1, 2, ... กับพิกเซลที่มีภาพตามลำดับแถว (row-major) เหมือน np.nonzero
    seed_rows, seed_columns = np.nonzero(empty == 0)
    return labels[seed_rows, seed_columns][nearest - 1]


def pyramid_levels(canvas_size: tuple[int, int], requested: int = 5) -> int:
    shortest = max(1, min(canvas_size))
    return int(np.clip(requested, 1, max(1, int(np.log2(shortest)) - 3)))


def multiband_blend(
    warps: list[WarpedImage],
    canvas_size: tuple[int, int],
    labels: np.ndarray | None = None,
    levels: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Laplacian pyramid blending (Burt & Adelson, 1983)

    รายละเอียดความถี่สูงถูกผสมในช่วงแคบใกล้ seam (ภาพคม ไม่เป็นเงาซ้อน)
    ส่วนความถี่ต่ำ (แสง/สี) ถูกผสมในช่วงกว้าง ทำให้รอยต่อกลืนกัน
    """
    width, height = canvas_size
    if labels is None:
        labels = seam_labels(warps, canvas_size)
    levels = pyramid_levels(canvas_size, levels)
    step = 2**levels
    padded_width = int(np.ceil(width / step)) * step
    padded_height = int(np.ceil(height / step)) * step
    padded_labels = np.full((padded_height, padded_width), -1, np.int16)
    padded_labels[:height, :width] = labels

    # ขยาย label ออกไปนอกพาโนรามา (ให้พิกเซลว่างเป็นของภาพที่ใกล้ที่สุด) เพื่อให้ทุกชั้นของ
    # pyramid มีน้ำหนักครบถึงขอบ ไม่เช่นนั้นชั้นความถี่ต่ำจะดึงค่า 0 เข้ามาทำให้ขอบภาพมืด
    extended_labels = _extend_labels(padded_labels)
    margin = 2 * step

    level_sizes = [(padded_width >> level, padded_height >> level) for level in range(levels + 1)]
    accumulated = [np.zeros((h, w, 3), np.float32) for w, h in level_sizes]
    weight_sums = [np.zeros((h, w), np.float32) for w, h in level_sizes]

    for order, warp in enumerate(warps):
        # ขยาย ROI ออกไป margin และให้ตำแหน่ง/ขนาดหารด้วย 2^levels ลงตัว ทุกชั้นของ pyramid จะตรงกับ canvas
        x0 = max(0, (warp.x - margin) // step * step)
        y0 = max(0, (warp.y - margin) // step * step)
        x1 = min(padded_width, int(np.ceil((warp.x + warp.width + margin) / step)) * step)
        y1 = min(padded_height, int(np.ceil((warp.y + warp.height + margin) / step)) * step)
        image = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        valid = np.zeros((y1 - y0, x1 - x0), bool)
        local = (slice(warp.y - y0, warp.y - y0 + warp.height), slice(warp.x - x0, warp.x - x0 + warp.width))
        image[local] = warp.image
        valid[local] = warp.mask > 0
        image = _fill_invalid(image, valid, levels + 3)
        weight = (extended_labels[y0:y1, x0:x1] == order).astype(np.float32)
        if not weight.any():
            continue

        gaussian = image
        for level in range(levels + 1):
            if level < levels:
                smaller = cv2.pyrDown(gaussian)
                band = gaussian - cv2.pyrUp(smaller, dstsize=(gaussian.shape[1], gaussian.shape[0]))
            else:
                band = gaussian
            region = (slice(y0 >> level, y1 >> level), slice(x0 >> level, x1 >> level))
            accumulated[level][region] += band * weight[..., None]
            weight_sums[level][region] += weight
            if level < levels:
                gaussian = smaller
                weight = cv2.pyrDown(weight)

    result = accumulated[levels] / np.maximum(weight_sums[levels], 1e-6)[..., None]
    for level in range(levels - 1, -1, -1):
        w, h = level_sizes[level]
        band = accumulated[level] / np.maximum(weight_sums[level], 1e-6)[..., None]
        result = cv2.pyrUp(result, dstsize=(w, h)) + band

    union = np.where(labels >= 0, 255, 0).astype(np.uint8)
    panorama = np.clip(result[:height, :width] + 0.5, 0, 255).astype(np.uint8)
    panorama[union == 0] = 0
    return panorama, union


def blend(
    warps: list[WarpedImage],
    canvas_size: tuple[int, int],
    method: str = "multiband",
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "multiband":
        return multiband_blend(warps, canvas_size, labels)
    if method == "feather":
        return feather_blend(warps, canvas_size)
    if method == "none":
        return overlay_blend(warps, canvas_size)
    raise ValueError(f"ไม่รู้จักวิธี blending: {method}")


def _max_rectangle(valid: np.ndarray) -> tuple[int, int, int, int]:
    """สี่เหลี่ยมพื้นที่มากที่สุดที่อยู่ใน mask ทั้งหมด (histogram + stack) คืน (x, y, w, h)"""
    rows, columns = valid.shape
    heights = [0] * columns
    best, best_area = (0, 0, 0, 0), 0
    for row in range(rows):
        line = valid[row].tolist()
        heights = [height + 1 if cell else 0 for height, cell in zip(heights, line)]
        stack: list[tuple[int, int]] = []  # (คอลัมน์เริ่ม, ความสูง)
        for column in range(columns + 1):
            current = heights[column] if column < columns else 0
            start = column
            while stack and stack[-1][1] >= current:
                start, height = stack.pop()
                area = height * (column - start)
                if area > best_area:
                    best_area = area
                    best = (start, row - height + 1, column - start, height)
            stack.append((start, current))
    return best


def largest_valid_rectangle(mask: np.ndarray, work_side: int = 480) -> tuple[int, int, int, int]:
    """หากรอบสี่เหลี่ยมใหญ่ที่สุดที่ไม่มีขอบดำ คืน (x, y, w, h) ในพิกัดภาพเต็ม"""
    height, width = mask.shape
    scale = min(1.0, work_side / max(height, width))
    small_width, small_height = max(1, int(width * scale)), max(1, int(height * scale))
    coverage = cv2.resize((mask > 0).astype(np.float32), (small_width, small_height), interpolation=cv2.INTER_AREA)
    x, y, w, h = _max_rectangle(coverage >= 0.999)
    if w == 0 or h == 0:
        return 0, 0, width, height

    sx, sy = width / small_width, height / small_height
    left, top = int(np.ceil(x * sx)), int(np.ceil(y * sy))
    right, bottom = int(np.floor((x + w) * sx)), int(np.floor((y + h) * sy))
    # ตรวจที่ความละเอียดเต็มอีกครั้ง หดขอบด้านที่ยังมีพิกเซลว่างจนกว่าจะเต็มทั้งกรอบ
    valid = mask > 0
    for _ in range(64):
        if right - left < 2 or bottom - top < 2:
            break
        top_bad = not valid[top, left:right].all()
        bottom_bad = not valid[bottom - 1, left:right].all()
        left_bad = not valid[top:bottom, left].all()
        right_bad = not valid[top:bottom, right - 1].all()
        if not (top_bad or bottom_bad or left_bad or right_bad):
            break
        top += top_bad
        bottom -= bottom_bad
        left += left_bad
        right -= right_bad
    return left, top, max(1, right - left), max(1, bottom - top)
