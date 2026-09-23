"""สร้างชุดภาพสังเคราะห์ที่รู้ค่า transform จริง (ground truth) สำหรับทดสอบ ประเมินผล และเดโม"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_WORDS = ["CP461", "SWU", "PANO", "SIFT", "RANSAC", "OPENCV", "BLEND", "VISION", "HOMOGRAPHY", "2026"]


@dataclass
class SyntheticSet:
    images: list[np.ndarray]
    scene: np.ndarray
    view_to_scene: list[np.ndarray]  # 3x3 แปลงพิกเซลของภาพ k → พิกเซลของฉาก (planar เท่านั้น)
    focal: float | None = None  # focal length จริง (rotation set)
    yaw_degrees: list[float] | None = None


def make_scene(width: int = 2400, height: int = 900, seed: int = 0) -> np.ndarray:
    """ฉากที่มีรายละเอียดหลายระดับ (สีไล่ระดับ + ลวดลาย + รูปทรง + ตัวอักษร) ให้ SIFT/ORB หาจุดได้"""
    rng = np.random.default_rng(seed)
    low = rng.uniform(50, 210, size=(4, 10, 3)).astype(np.float32)
    scene = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    middle = rng.normal(0, 22, size=(max(2, height // 10), max(2, width // 10), 3)).astype(np.float32)
    scene = np.clip(scene + cv2.resize(middle, (width, height), interpolation=cv2.INTER_CUBIC), 0, 255)
    scene = scene.astype(np.uint8)

    for _ in range(int(320 * width * height / 1e6)):
        color = tuple(int(value) for value in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        size = int(rng.integers(10, 80))
        filled = -1 if rng.random() < 0.6 else int(rng.integers(2, 5))
        kind = rng.integers(0, 4)
        if kind == 0:
            cv2.rectangle(scene, (x, y), (x + size, y + int(rng.integers(10, 80))), color, filled)
        elif kind == 1:
            cv2.circle(scene, (x, y), size // 2, color, filled, cv2.LINE_AA)
        elif kind == 2:
            points = (rng.integers(-size, size, (int(rng.integers(3, 6)), 2)) + (x, y)).astype(np.int32)
            if filled < 0:
                cv2.fillPoly(scene, [points], color, cv2.LINE_AA)
            else:
                cv2.polylines(scene, [points], True, color, filled, cv2.LINE_AA)
        else:
            text = f"{_WORDS[int(rng.integers(len(_WORDS)))]}-{int(rng.integers(10, 99))}"
            cv2.putText(scene, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.8)),
                        color, int(rng.integers(1, 4)), cv2.LINE_AA)

    grain = rng.normal(0, 4, scene.shape)
    return np.clip(scene.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _photometric(image: np.ndarray, rng, gain_range: float, vignetting: float, noise: float, jpeg_quality: int):
    """จำลองกล้องจริง: แสงต่างกันต่อภาพ, ขอบภาพมืด (vignetting), noise และการบีบอัด JPEG"""
    height, width = image.shape[:2]
    result = image.astype(np.float32)
    gain = rng.uniform(1 - gain_range, 1 + gain_range)
    tint = rng.uniform(0.97, 1.03, size=3) if gain_range else np.ones(3)
    result *= (gain * tint).astype(np.float32)
    if vignetting:
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        radius = ((x - width / 2) ** 2 + (y - height / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
        result *= (1.0 - vignetting * radius)[..., None]
    if noise:
        result += rng.normal(0, noise, result.shape).astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if jpeg_quality:
        encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


def make_planar_set(
    count: int = 3,
    view_size: tuple[int, int] = (640, 480),
    overlap: float = 0.4,
    perspective: float = 0.04,
    gain_range: float = 0.0,
    vignetting: float = 0.0,
    noise: float = 1.5,
    jpeg_quality: int = 92,
    seed: int = 0,
    scene: np.ndarray | None = None,
) -> SyntheticSet:
    """ถ่ายฉากระนาบหลายมุม เรียงจากซ้ายไปขวา ภาพติดกันซ้อนทับกันประมาณ overlap"""
    rng = np.random.default_rng(seed + 1000)
    view_width, view_height = view_size
    if scene is None:
        footprint_width = view_width * 1.1
        scene_width = int(footprint_width * (1 + (1 - overlap) * (count - 1)) + 240)
        scene = make_scene(scene_width, int(view_height * 1.1 * 1.35 + 160), seed)
    scene_height, scene_width = scene.shape[:2]
    margin = 120
    footprint_width = (scene_width - 2 * margin) / (1 + (1 - overlap) * (count - 1))
    footprint_height = footprint_width * view_height / view_width
    step = footprint_width * (1 - overlap)
    top = (scene_height - footprint_height) / 2

    images, view_to_scene = [], []
    view_corners = np.float32([[0, 0], [view_width, 0], [view_width, view_height], [0, view_height]])
    for index in range(count):
        left = margin + index * step
        y_offset = rng.uniform(-0.05, 0.05) * footprint_height
        quad = np.float32([
            [left, top + y_offset],
            [left + footprint_width, top + y_offset],
            [left + footprint_width, top + footprint_height + y_offset],
            [left, top + footprint_height + y_offset],
        ])
        quad += rng.normal(0, perspective * footprint_width, quad.shape).astype(np.float32)
        quad = np.clip(quad, 0, [scene_width - 1, scene_height - 1]).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(view_corners, quad)
        view = cv2.warpPerspective(
            scene, matrix, view_size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT
        )
        images.append(_photometric(view, rng, gain_range, vignetting, noise, jpeg_quality))
        view_to_scene.append(matrix.astype(np.float64))
    return SyntheticSet(images, scene, view_to_scene)


def make_rotation_set(
    count: int = 5,
    yaw_step: float = 30.0,
    horizontal_fov: float = 60.0,
    view_size: tuple[int, int] = (640, 480),
    pitch_jitter: float = 1.5,
    gain_range: float = 0.1,
    vignetting: float = 0.2,
    noise: float = 1.5,
    jpeg_quality: int = 92,
    seed: int = 0,
) -> SyntheticSet:
    """กล้องหมุนรอบตัวเองกลางฉากทรงกระบอก 360° (กรณีมุมกว้างที่ระนาบเดียวรองรับไม่ได้)"""
    rng = np.random.default_rng(seed + 2000)
    view_width, view_height = view_size
    focal = (view_width / 2) / np.tan(np.radians(horizontal_fov) / 2)
    radius = focal
    environment_width = int(2 * np.pi * radius)
    environment_height = int(2.2 * view_height)
    environment = make_scene(environment_width, environment_height, seed)

    x, y = np.meshgrid(np.arange(view_width, dtype=np.float64) - view_width / 2,
                       np.arange(view_height, dtype=np.float64) - view_height / 2)
    images, yaws = [], []
    start = -yaw_step * (count - 1) / 2
    for index in range(count):
        yaw = np.radians(start + index * yaw_step)
        pitch = np.radians(rng.uniform(-pitch_jitter, pitch_jitter))
        # ทิศของรังสีในพิกัดกล้อง → หมุน pitch (แกน x) แล้ว yaw (แกน y)
        ray_y = y * np.cos(pitch) - focal * np.sin(pitch)
        ray_z = y * np.sin(pitch) + focal * np.cos(pitch)
        ray_x = x * np.cos(yaw) + ray_z * np.sin(yaw)
        ray_z = -x * np.sin(yaw) + ray_z * np.cos(yaw)
        angle = np.arctan2(ray_x, ray_z)
        cylinder_height = ray_y / np.hypot(ray_x, ray_z)
        map_x = np.mod(angle * radius + environment_width / 2, environment_width).astype(np.float32)
        map_y = (cylinder_height * radius + environment_height / 2).astype(np.float32)
        view = cv2.remap(environment, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        images.append(_photometric(view, rng, gain_range, vignetting, noise, jpeg_quality))
        yaws.append(float(np.degrees(yaw)))
    return SyntheticSet(images, environment, [], focal=float(focal), yaw_degrees=yaws)


def make_demo_set(seed: int = 7) -> list[np.ndarray]:
    """ชุดภาพตัวอย่างในเว็บ: 4 ภาพ แสงต่างกัน มี vignetting และสลับลำดับไว้"""
    synthetic = make_planar_set(
        count=4, view_size=(720, 540), overlap=0.42, perspective=0.035,
        gain_range=0.18, vignetting=0.28, noise=2.0, seed=seed,
    )
    order = [2, 0, 3, 1]
    return [synthetic.images[index] for index in order]
