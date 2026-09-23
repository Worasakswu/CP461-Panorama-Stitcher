"""Pipeline หลัก: features → จับคู่ทุกคู่ภาพ → เลือกโครงข่ายภาพ → warp → ปรับแสง → blend → crop"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np

from src.blending import apply_gains, blend, gain_compensation, largest_valid_rectangle, seam_labels
from src.feature import ImageFeatures, detect_features, match_features, matched_points
from src.homography import TransformEstimate, centered, estimate_focal, estimate_transform, transform_is_plausible
from src.image_io import resize_to_max_side
from src.warping import (
    WarpedImage,
    canvas_from_transforms,
    cylindrical_warp,
    to_cylindrical_points,
    translation,
    warp_to_canvas,
)

PROJECTIONS = ("auto", "planar", "cylindrical")
MAX_IMAGES = 10


@dataclass(frozen=True)
class StitchSettings:
    detector: str = "SIFT"
    max_features: int = 4000
    ratio: float = 0.75
    ransac_threshold: float = 4.0
    min_inliers: int = 15
    projection: str = "auto"
    blend: str = "multiband"
    exposure_compensation: bool = True
    crop: bool = True
    max_side: int = 1200
    focal: float | None = None  # px ที่ความละเอียดทำงาน, None = ประมาณจาก homography
    max_canvas_pixels: int = 12_000_000


@dataclass
class PairMatch:
    first: int
    second: int
    matches: list
    first_points: np.ndarray
    second_points: np.ndarray
    estimate: TransformEstimate  # homography ที่แปลงภาพ second → first
    cylindrical: TransformEstimate | None = None  # similarity บนพิกัดทรงกระบอก
    in_tree: bool = False

    @property
    def verified(self) -> bool:
        return self.estimate.ok

    def used_estimate(self, projection: str) -> TransformEstimate:
        if projection == "cylindrical" and self.cylindrical is not None:
            return self.cylindrical
        return self.estimate


class StitchError(ValueError):
    def __init__(self, message: str, pairs: list[PairMatch] | None = None, features: list | None = None):
        super().__init__(message)
        self.pairs = pairs or []
        self.features = features or []


@dataclass
class StitchResult:
    panorama: np.ndarray
    panorama_uncropped: np.ndarray
    mask: np.ndarray
    crop_rect: tuple[int, int, int, int] | None
    canvas_size: tuple[int, int]
    images: list[np.ndarray]
    features: list[ImageFeatures]
    pairs: list[PairMatch]
    included: list[int]  # เรียงจากซ้ายไปขวาบนพาโนรามา
    excluded: list[int]
    reference: int
    projection: str
    focal: float | None
    focal_estimated: bool
    transforms: dict[int, np.ndarray]  # ภาพ → ภาพอ้างอิง
    canvas_transforms: dict[int, np.ndarray]  # ภาพ → canvas ก่อน crop
    canvas_scale: float
    warps: list[WarpedImage]
    gains: np.ndarray | None
    labels: np.ndarray
    settings: StitchSettings
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def pair(self, first: int, second: int) -> PairMatch | None:
        for pair in self.pairs:
            if {pair.first, pair.second} == {first, second}:
                return pair
        return None

    def output_transform(self, index: int) -> np.ndarray | None:
        """เมทริกซ์จากพิกัดภาพ index (ความละเอียดทำงาน) → พิกัดภาพผลลัพธ์ (เฉพาะ planar)"""
        if self.projection != "planar" or index not in self.canvas_transforms:
            return None
        if self.crop_rect is None:
            return self.canvas_transforms[index]
        return translation(-self.crop_rect[0], -self.crop_rect[1]) @ self.canvas_transforms[index]

    def recompose(self, method: str, exposure_compensation: bool | None = None) -> np.ndarray:
        """รวมภาพใหม่ด้วยวิธี blending อื่นจาก warps เดิม (ใช้เปรียบเทียบผล)"""
        use_gains = self.settings.exposure_compensation if exposure_compensation is None else exposure_compensation
        gains = (self.gains if self.gains is not None else gain_compensation(self.warps)) if use_gains else None
        panorama, _ = _blend(apply_gains(self.warps, gains), self.canvas_size, method, self.labels, self.reference)
        return _crop(panorama, self.crop_rect)


def stitch(
    images: list[np.ndarray],
    settings: StitchSettings = StitchSettings(),
    progress: Callable[[float, str], None] | None = None,
) -> StitchResult:
    report = progress or (lambda fraction, message: None)
    if len(images) < 2:
        raise StitchError("ต้องใช้ภาพอย่างน้อย 2 ภาพ")
    if len(images) > MAX_IMAGES:
        raise StitchError(f"รองรับสูงสุด {MAX_IMAGES} ภาพต่อครั้ง")

    timings: dict[str, float] = {}
    warnings: list[str] = []
    started = clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - clock
        clock = now

    working = [resize_to_max_side(image, settings.max_side)[0] for image in images]
    sizes = [(image.shape[1], image.shape[0]) for image in working]

    # 1) หา keypoints + descriptors ของทุกภาพ
    features = []
    for index, image in enumerate(working):
        report(0.02 + 0.33 * index / len(working), f"ตรวจหา {settings.detector} keypoints ในภาพ {index + 1}")
        features.append(detect_features(image, settings.detector, settings.max_features))
    lap("features")

    # 2) จับคู่ทุกคู่ภาพ ไม่ต้องให้ผู้ใช้เรียงลำดับเอง
    pairs = []
    all_pairs = list(combinations(range(len(working)), 2))
    for number, (first, second) in enumerate(all_pairs):
        report(0.35 + 0.35 * number / len(all_pairs), f"จับคู่ภาพ {first + 1} ↔ {second + 1}")
        matches = match_features(features[first], features[second], settings.ratio)
        first_points, second_points = matched_points(features[first], features[second], matches)
        estimate = estimate_transform(
            second_points, first_points, "homography", settings.ransac_threshold,
            settings.min_inliers, source_size=sizes[second],
        )
        pairs.append(PairMatch(first, second, matches, first_points, second_points, estimate))
    lap("matching")

    # 3) เลือกกลุ่มภาพที่เชื่อมกันได้ และเลือกภาพอ้างอิงที่อยู่กลางโครงข่าย
    report(0.72, "คำนวณตำแหน่งของทุกภาพ")
    verified = [pair for pair in pairs if pair.verified]
    component = _largest_component(len(working), verified)
    if len(component) < 2:
        raise StitchError(_no_overlap_message(pairs), pairs, features)

    projection = settings.projection
    focal, focal_estimated = None, False
    if projection in ("auto", "planar"):
        tree = _maximum_spanning_tree(component, verified, "planar")
        reference = _tree_center(component, tree)
        transforms = _compose(reference, tree, "planar")
        problem = _planar_problem(transforms, sizes)
        if problem is None:
            projection = "planar"
        elif projection == "planar":
            raise StitchError(f"ต่อแบบระนาบ (Planar) ไม่ได้: {problem} ลองเลือก Cylindrical projection", pairs, features)
        else:
            warnings.append(f"ใช้ Cylindrical projection อัตโนมัติ เพราะ{problem}")
            projection = "cylindrical"

    sources = [(image, np.full(image.shape[:2], 255, np.uint8)) for image in working]
    if projection == "cylindrical":
        focal, focal_estimated = _focal_length(settings, verified, component, sizes)
        if not focal_estimated:
            warnings.append(f"ประมาณ focal length จากภาพไม่ได้ จึงใช้ค่าเริ่มต้น {focal:.0f} px")
        for pair in verified:
            first_points = to_cylindrical_points(pair.first_points, focal, sizes[pair.first])
            second_points = to_cylindrical_points(pair.second_points, focal, sizes[pair.second])
            pair.cylindrical = estimate_transform(
                second_points, first_points, "similarity", settings.ransac_threshold, settings.min_inliers
            )
            if pair.cylindrical.ok and not _similarity_is_plausible(pair.cylindrical.matrix):
                pair.cylindrical.failure_reason = "สเกลหรือการหมุนบนทรงกระบอกผิดปกติ"
        cylindrical_pairs = [pair for pair in verified if pair.cylindrical.ok]
        component = _largest_component(len(working), cylindrical_pairs, allowed=component)
        if len(component) < 2:
            raise StitchError("ต่อภาพบนทรงกระบอกไม่สำเร็จ ลองปรับ focal length หรือใช้ Planar", pairs, features)
        tree = _maximum_spanning_tree(component, cylindrical_pairs, "cylindrical")
        reference = _tree_center(component, tree)
        transforms = _compose(reference, tree, "cylindrical")
        sources = [cylindrical_warp(image, focal) if index in component else (image, None)
                   for index, (image, _) in enumerate(sources)]

    for pair in tree:
        pair.in_tree = True
    excluded = sorted(set(range(len(working))) - set(component))
    for index in excluded:
        warnings.append(f"ภาพ {index + 1} ไม่ถูกนำมาต่อ เพราะไม่พบส่วนซ้อนทับกับภาพอื่นมากพอ")
    lap("geometry")

    # 4) Warp ทุกภาพลงบน canvas เดียวกัน
    report(0.78, "Warp ภาพลงบน canvas")
    offset, canvas_size, canvas_scale = canvas_from_transforms(
        [transforms[index] for index in component], [sizes[index] for index in component],
        settings.max_canvas_pixels,
    )
    if canvas_scale < 1.0:
        warnings.append(f"พาโนรามาใหญ่เกินหน่วยความจำที่กำหนด จึงย่อผลลัพธ์เหลือ {canvas_scale * 100:.0f}%")
    canvas_transforms = {index: offset @ transforms[index] for index in component}
    warps = [
        warp_to_canvas(index, sources[index][0], sources[index][1], canvas_transforms[index], canvas_size)
        for index in component
    ]
    lap("warping")

    # 5) ปรับความสว่าง/สีของแต่ละภาพให้เข้ากันก่อนรวม
    gains = None
    if settings.exposure_compensation:
        report(0.84, "ปรับแสงระหว่างภาพ (gain compensation)")
        gains = gain_compensation(warps)
    lap("exposure")

    # 6) หา seam แล้ว blend
    report(0.88, f"รวมภาพด้วย {settings.blend} blending")
    labels = seam_labels(warps, canvas_size)
    panorama_uncropped, mask = _blend(apply_gains(warps, gains), canvas_size, settings.blend, labels, reference)
    lap("blending")

    # 7) ตัดขอบดำ
    crop_rect = None
    if settings.crop:
        report(0.96, "ตัดขอบดำ")
        crop_rect = largest_valid_rectangle(mask)
        kept = crop_rect[2] * crop_rect[3] / max(1, np.count_nonzero(mask))
        if kept < 0.5:
            warnings.append(f"ตัดขอบดำแล้วเหลือพื้นที่เพียง {kept * 100:.0f}% ของภาพ ปิด Auto-crop เพื่อดูภาพเต็ม")
    panorama = _crop(panorama_uncropped, crop_rect)
    lap("crop")
    timings["total"] = time.perf_counter() - started
    report(1.0, "เสร็จสิ้น")

    order = sorted(warps, key=lambda warp: warp.x + warp.width / 2)
    return StitchResult(
        panorama=panorama,
        panorama_uncropped=panorama_uncropped,
        mask=mask,
        crop_rect=crop_rect,
        canvas_size=canvas_size,
        images=working,
        features=features,
        pairs=pairs,
        included=[warp.index for warp in order],
        excluded=excluded,
        reference=reference,
        projection=projection,
        focal=focal,
        focal_estimated=focal_estimated,
        transforms=transforms,
        canvas_transforms=canvas_transforms,
        canvas_scale=canvas_scale,
        warps=warps,
        gains=gains,
        labels=labels,
        settings=settings,
        timings=timings,
        warnings=warnings,
    )


def _blend(warps, canvas_size, method, labels, reference):
    if method == "none":
        # วางภาพอ้างอิงไว้บนสุด เหมือนโค้ดเวอร์ชันแรก
        warps = sorted(warps, key=lambda warp: warp.index == reference)
    return blend(warps, canvas_size, method, labels)


def _crop(image: np.ndarray, rect: tuple[int, int, int, int] | None) -> np.ndarray:
    if rect is None:
        return image
    x, y, width, height = rect
    return image[y:y + height, x:x + width].copy()


def _largest_component(count: int, pairs: list[PairMatch], allowed: list[int] | None = None) -> list[int]:
    nodes = set(range(count)) if allowed is None else set(allowed)
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for pair in pairs:
        if pair.first in nodes and pair.second in nodes:
            parent[find(pair.first)] = find(pair.second)
    groups: dict[int, list[int]] = {}
    for node in sorted(nodes):
        groups.setdefault(find(node), []).append(node)
    return max(groups.values(), key=lambda group: (len(group), -group[0]))


def _maximum_spanning_tree(nodes: list[int], pairs: list[PairMatch], projection: str) -> list[PairMatch]:
    """เลือกเส้นเชื่อมระหว่างภาพที่มี inliers มากที่สุดก่อน (Kruskal) ให้ทุกภาพเชื่อมถึงกันด้วยคู่ที่แม่นที่สุด"""
    node_set = set(nodes)
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree = []
    candidates = [pair for pair in pairs if pair.first in node_set and pair.second in node_set]
    for pair in sorted(candidates, key=lambda pair: pair.used_estimate(projection).num_inliers, reverse=True):
        first, second = find(pair.first), find(pair.second)
        if first != second:
            parent[first] = second
            tree.append(pair)
    return tree


def _adjacency(tree: list[PairMatch]) -> dict[int, list[tuple[int, PairMatch]]]:
    adjacency: dict[int, list[tuple[int, PairMatch]]] = {}
    for pair in tree:
        adjacency.setdefault(pair.first, []).append((pair.second, pair))
        adjacency.setdefault(pair.second, []).append((pair.first, pair))
    return adjacency


def _tree_center(nodes: list[int], tree: list[PairMatch]) -> int:
    """ภาพอ้างอิง = ภาพที่อยู่ห่างจากภาพไกลสุดน้อยที่สุด ช่วยลดการยืดของภาพริม"""
    adjacency = _adjacency(tree)

    def eccentricity(start: int) -> int:
        distance = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency.get(node, []):
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        return max(distance.values())

    def strength(node: int) -> int:
        return sum(pair.estimate.num_inliers for _, pair in adjacency.get(node, []))

    return min(nodes, key=lambda node: (eccentricity(node), -strength(node), node))


def _compose(reference: int, tree: list[PairMatch], projection: str) -> dict[int, np.ndarray]:
    """คูณเมทริกซ์ต่อกันตามโครงข่าย เพื่อให้ได้ transform จากทุกภาพ → ภาพอ้างอิง"""
    adjacency = _adjacency(tree)
    transforms = {reference: np.eye(3)}
    queue = deque([reference])
    while queue:
        node = queue.popleft()
        for neighbor, pair in adjacency.get(node, []):
            if neighbor in transforms:
                continue
            matrix = pair.used_estimate(projection).matrix  # second → first
            to_node = matrix if pair.first == node else np.linalg.inv(matrix)
            combined = transforms[node] @ to_node
            transforms[neighbor] = combined / combined[2, 2]
            queue.append(neighbor)
    return transforms


def _planar_problem(transforms: dict[int, np.ndarray], sizes: list[tuple[int, int]]) -> str | None:
    for index, matrix in transforms.items():
        plausible, reason = transform_is_plausible(matrix, *sizes[index], max_area_ratio=8.0, max_depth_ratio=4.0)
        if not plausible:
            return f"ภาพ {index + 1} {reason}"
    return None


def _similarity_is_plausible(matrix: np.ndarray) -> bool:
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return 0.7 <= scale <= 1.4 and angle <= 20.0


def _focal_length(settings, verified, component, sizes) -> tuple[float, bool]:
    if settings.focal:
        return float(settings.focal), True
    members = set(component)
    homographies = [
        centered(pair.estimate.matrix, sizes[pair.second], sizes[pair.first])
        for pair in verified
        if pair.first in members and pair.second in members
    ]
    return estimate_focal(homographies, sizes[component[0]])


def _no_overlap_message(pairs: list[PairMatch]) -> str:
    message = "ไม่พบคู่ภาพที่ซ้อนทับกันมากพอจะต่อได้"
    if pairs:
        best = max(pairs, key=lambda pair: pair.estimate.num_inliers)
        message += (
            f" (คู่ที่ใกล้เคียงที่สุดคือภาพ {best.first + 1} ↔ {best.second + 1}: "
            f"inliers {best.estimate.num_inliers} จาก {best.estimate.num_matches} คู่"
            + (f" — {best.estimate.failure_reason}" if best.estimate.failure_reason else "")
            + ")"
        )
    return message + " ควรถ่ายให้ภาพติดกันซ้อนทับกันราว 30–50% โดยหมุนกล้องอยู่กับที่"
