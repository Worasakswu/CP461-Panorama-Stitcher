"""ประมาณ Homography ด้วย RANSAC ตรวจความสมเหตุสมผล และประมาณ focal length"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MODELS = ("homography", "similarity")
STRONG_INLIERS = 100


@dataclass
class TransformEstimate:
    matrix: np.ndarray | None  # 3x3 แปลงพิกัด source → destination
    inlier_mask: np.ndarray  # bool (N,)
    num_matches: int
    num_inliers: int
    reprojection_rmse: float  # px เฉลี่ยของ inliers
    failure_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.matrix is not None and self.failure_reason is None

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0


def estimate_transform(
    source: np.ndarray,
    destination: np.ndarray,
    model: str = "homography",
    threshold: float = 4.0,
    min_inliers: int = 15,
    source_size: tuple[int, int] | None = None,
) -> TransformEstimate:
    """RANSAC หา transform ที่แปลง source → destination แล้วตรวจว่าเชื่อถือได้

    - homography: 8 DOF ใช้กับภาพระนาบ/กล้องหมุนรอบจุดเดิม
    - similarity: 4 DOF (หมุน + สเกล + เลื่อน) ใช้หลังฉายภาพลงทรงกระบอก
    """
    count = len(source)
    empty_mask = np.zeros(count, bool)
    minimum_points = 4 if model == "homography" else 3
    if count < minimum_points:
        return TransformEstimate(
            None, empty_mask, count, 0, float("nan"),
            f"คู่จุดน้อยเกินไป ({count} คู่ ต้องมีอย่างน้อย {minimum_points})",
        )

    source = np.float32(source).reshape(-1, 1, 2)
    destination = np.float32(destination).reshape(-1, 1, 2)
    if model == "homography":
        matrix, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, threshold, maxIters=5000, confidence=0.995
        )
    elif model == "similarity":
        affine, mask = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=threshold,
            maxIters=5000, confidence=0.995,
        )
        matrix = None if affine is None else np.vstack([affine, [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {model}")

    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        return TransformEstimate(None, empty_mask, count, 0, float("nan"), "RANSAC หาโมเดลที่สอดคล้องกันไม่ได้")

    matrix = matrix / matrix[2, 2]
    inliers = mask.ravel().astype(bool)
    num_inliers = int(inliers.sum())
    projected = cv2.perspectiveTransform(source[inliers], matrix) if num_inliers else source[:0]
    errors = np.linalg.norm(projected - destination[inliers], axis=2).ravel()
    rmse = float(np.sqrt(np.mean(errors**2))) if num_inliers else float("nan")

    reason = None
    # เกณฑ์ของ Brown & Lowe (2007): inliers ต้องมากพอเมื่อเทียบกับจำนวนคู่ทั้งหมด
    # ยกเว้นเมื่อ inliers มากจนเป็นความบังเอิญไม่ได้ ภาพจริงที่มีวัตถุใกล้กล้อง (parallax)
    # หรือคนเดินผ่านจะมี outliers เยอะแม้ภาพซ้อนกันจริง
    if num_inliers < min_inliers:
        reason = f"inliers น้อยเกินไป ({num_inliers} < {min_inliers})"
    elif num_inliers <= 8 + 0.3 * count and num_inliers < STRONG_INLIERS:
        reason = f"สัดส่วน inliers ต่ำเกินไป ({num_inliers}/{count}) อาจเป็นคู่ภาพที่ไม่ได้ซ้อนทับกัน"
    elif source_size is not None:
        plausible, why = transform_is_plausible(matrix, *source_size)
        if not plausible:
            reason = why

    return TransformEstimate(matrix, inliers, count, num_inliers, rmse, reason)


def project_corners(matrix: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """ฉายมุมภาพ 4 มุม คืน (พิกัด 4x2, ค่า w ของ homogeneous coordinates)"""
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], np.float64).T
    projected = matrix @ corners
    w = projected[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        points = (projected[:2] / w).T
    return points, w


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def transform_is_plausible(
    matrix: np.ndarray,
    width: float,
    height: float,
    max_area_ratio: float = 8.0,
    max_depth_ratio: float = 6.0,
) -> tuple[bool, str | None]:
    """ตรวจว่า transform ไม่บิดภาพจนผิดธรรมชาติ

    ปฏิเสธเมื่อ: ค่าไม่ finite, มุมภาพถูกฉายข้ามเส้นขอบฟ้า (w ≤ 0), ภาพกลับด้าน/ไขว้กัน,
    พื้นที่ขยายหรือหดเกินกำหนด, หรือด้านหนึ่งยืดมากกว่าอีกด้านมากเกินไป
    """
    if matrix is None or not np.all(np.isfinite(matrix)):
        return False, "เมทริกซ์มีค่าไม่ถูกต้อง"
    points, w = project_corners(matrix, width, height)
    if np.any(w <= 1e-9):
        return False, "ภาพถูกฉายเลยเส้นขอบฟ้า (มุมกว้างเกินกว่าระนาบเดียวจะรองรับ)"
    if w.max() / w.min() > max_depth_ratio:
        return False, "ภาพถูกยืดด้านหนึ่งมากเกินไป"

    # ภาพที่ถูกต้องต้องเป็นสี่เหลี่ยมนูนและเรียงมุมทิศเดิม (ไม่สะท้อนกลับด้าน)
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not np.all(cross > 0):
        return False, "ภาพถูกพับหรือกลับด้าน"

    area_ratio = polygon_area(points) / (width * height)
    if not (1.0 / max_area_ratio <= area_ratio <= max_area_ratio):
        return False, f"สเกลของภาพเปลี่ยนมากผิดปกติ ({area_ratio:.2f} เท่า)"
    return True, None


def centered(matrix: np.ndarray, source_size: tuple[int, int], destination_size: tuple[int, int]) -> np.ndarray:
    """เปลี่ยน homography ให้ใช้พิกัดที่มีจุดศูนย์กลางภาพเป็นจุดกำเนิด"""
    to_source = _translation(source_size[0] / 2, source_size[1] / 2)
    from_destination = _translation(-destination_size[0] / 2, -destination_size[1] / 2)
    return from_destination @ matrix @ to_source


def focals_from_homography(matrix: np.ndarray) -> tuple[float | None, float | None]:
    """ประมาณ focal length (px) ของทั้งสองภาพจาก homography ของกล้องที่หมุนรอบจุดเดิม

    ใช้สูตรเดียวกับ cv::detail::focalsFromHomography (Szeliski & Shum, 1997)
    matrix ต้องอยู่ในพิกัดที่มีจุดศูนย์กลางภาพเป็นจุดกำเนิด และแปลงภาพที่สอง → ภาพแรก
    """
    h = (matrix / matrix[2, 2]).ravel()

    def solve(d1: float, d2: float, v1: float, v2: float) -> float | None:
        if v1 < v2:
            v1, v2 = v2, v1
        if v1 > 0 and v2 > 0:
            return float(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
        if v1 > 0:
            return float(np.sqrt(v1))
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = h[6] * h[7]
        d2 = (h[7] - h[6]) * (h[7] + h[6])
        second = solve(d1, d2, -(h[0] * h[1] + h[3] * h[4]) / d1,
                       (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2)
        d1 = h[0] * h[3] + h[1] * h[4]
        d2 = h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
        first = solve(d1, d2, -h[2] * h[5] / d1, (h[5] ** 2 - h[2] ** 2) / d2)
    return first, second


def rotation_residual(matrix: np.ndarray, focal: float) -> float:
    """ถ้ากล้องหมุนรอบจุดเดิม H = K·R·K⁻¹ ดังนั้น K⁻¹·H·K ต้องเป็นเมทริกซ์การหมุน (คูณสเกล)

    คืน ‖M·Mᵀ − I‖ หลังหารสเกลออก ค่ายิ่งใกล้ 0 ยิ่งแปลว่า focal นี้อธิบาย H ได้ดี (สูงสุด 1)
    """
    camera = np.diag([focal, focal, 1.0])
    rotation = np.linalg.inv(camera) @ matrix @ camera
    determinant = np.linalg.det(rotation)
    if not np.isfinite(determinant) or determinant <= 0:
        return 1.0
    rotation = rotation / np.cbrt(determinant)
    return min(float(np.linalg.norm(rotation @ rotation.T - np.eye(3))), 1.0)


def estimate_focal(centered_homographies: list[np.ndarray], image_size: tuple[int, int]) -> tuple[float, bool]:
    """หา focal length เดียวที่ทำให้ทุกคู่ภาพสอดคล้องกับกล้องที่หมุนรอบจุดเดิมที่สุด

    ผู้สมัครมาจากสูตรของ Szeliski & Shum และค่าที่ไล่ละเอียดในช่วง 0.3–5 เท่าของด้านยาวภาพ
    (สูตรอย่างเดียวไม่เสถียรเมื่อกล้องหมุนแนวนอนล้วน ๆ เพราะตัวหารเข้าใกล้ 0)
    คืน (focal, ประมาณได้จริงหรือไม่)
    """
    longest = max(image_size)
    fallback = 0.9 * longest  # ใกล้เลนส์มุมกว้างของมือถือ (มุมรับภาพแนวนอน ~58°)
    if not centered_homographies:
        return fallback, False

    low, high = 0.3 * longest, 5.0 * longest
    candidates = list(np.geomspace(low, high, 300))
    for matrix in centered_homographies:
        for focal in focals_from_homography(matrix):
            if focal and np.isfinite(focal) and low <= focal <= high:
                candidates.append(focal)

    def cost(focal: float) -> float:
        return float(np.mean([rotation_residual(matrix, focal) for matrix in centered_homographies]))

    best = min(candidates, key=cost)
    fine = np.linspace(0.97 * best, 1.03 * best, 61)
    best = float(min(fine, key=cost))
    median_residual = float(np.median([rotation_residual(matrix, best) for matrix in centered_homographies]))
    # ภาพที่ไม่ได้มาจากกล้องหมุน (เช่นเดินถ่ายฉากระนาบ) จะไม่มี focal ไหนทำให้ residual ต่ำ
    if median_residual > 0.3 or not (1.01 * low < best < 0.99 * high):
        return fallback, False
    return best, True


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
