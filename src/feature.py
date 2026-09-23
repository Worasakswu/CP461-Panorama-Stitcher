"""สกัดจุดเด่น (keypoints + descriptors) และจับคู่ด้วย KNN + Lowe's ratio test"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DETECTORS = ("SIFT", "ORB")


@dataclass
class ImageFeatures:
    keypoints: tuple
    descriptors: np.ndarray | None
    points: np.ndarray  # (N, 2) float32 พิกัด (x, y) ของ keypoints
    detector: str

    def __len__(self) -> int:
        return len(self.keypoints)


def create_detector(detector: str, max_features: int):
    if detector == "SIFT":
        return cv2.SIFT_create(nfeatures=max_features)
    if detector == "ORB":
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    raise ValueError(f"ไม่รู้จัก feature detector: {detector}")


def detect_features(
    image: np.ndarray,
    detector: str = "SIFT",
    max_features: int = 4000,
    mask: np.ndarray | None = None,
) -> ImageFeatures:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = create_detector(detector, max_features).detectAndCompute(gray, mask)
    points = np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1, 2)
    return ImageFeatures(tuple(keypoints), descriptors, points, detector)


def match_features(first: ImageFeatures, second: ImageFeatures, ratio: float = 0.75) -> list[cv2.DMatch]:
    """จับคู่ descriptor จาก first → second แล้วคัดด้วย Lowe's ratio test

    ภาพที่ไม่มี feature หรือมีน้อยกว่า 2 จุดจะได้ list ว่าง (โค้ดเดิม crash ในกรณีนี้)
    """
    if (
        first.descriptors is None
        or second.descriptors is None
        or len(first.descriptors) < 2
        or len(second.descriptors) < 2
    ):
        return []

    norm = cv2.NORM_HAMMING if first.detector == "ORB" else cv2.NORM_L2
    knn_matches = cv2.BFMatcher(norm).knnMatch(first.descriptors, second.descriptors, k=2)
    good = [
        pair[0]
        for pair in knn_matches
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]

    # หลายจุดในภาพแรกอาจชี้มาที่จุดเดียวกันในภาพที่สอง เก็บไว้เฉพาะคู่ที่ใกล้ที่สุด
    best_by_train: dict[int, cv2.DMatch] = {}
    for match in good:
        current = best_by_train.get(match.trainIdx)
        if current is None or match.distance < current.distance:
            best_by_train[match.trainIdx] = match
    return sorted(best_by_train.values(), key=lambda match: match.queryIdx)


def matched_points(
    first: ImageFeatures, second: ImageFeatures, matches: list[cv2.DMatch]
) -> tuple[np.ndarray, np.ndarray]:
    """คืนพิกัดของคู่จุด (first_points, second_points) ขนาด (N, 2)"""
    if not matches:
        empty = np.empty((0, 2), np.float32)
        return empty, empty.copy()
    first_points = first.points[[match.queryIdx for match in matches]]
    second_points = second.points[[match.trainIdx for match in matches]]
    return first_points, second_points
