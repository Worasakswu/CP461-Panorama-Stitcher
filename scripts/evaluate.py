"""ประเมินผลโค้ดเวอร์ชันแรก (baseline, commit 3555092) เทียบกับ pipeline ใหม่

    python scripts/evaluate.py                    # ชุดภาพสังเคราะห์ที่รู้ ground truth
    python scripts/evaluate.py --real test_images # ภาพถ่ายจริง: แต่ละโฟลเดอร์ย่อย = 1 ชุดพาโนรามา

ผลลัพธ์ (ตาราง markdown, CSV และภาพตัวอย่าง) ถูกบันทึกใน evaluation_outputs/ ซึ่งไม่ได้เก็บใน git
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_io import decode_image  # noqa: E402
from src.stitcher import StitchError, StitchSettings, stitch  # noqa: E402
from src.synthetic import make_planar_set, make_rotation_set  # noqa: E402

OUTPUT = ROOT / "evaluation_outputs"


# ---------------------------------------------------------------------------
# Baseline: คัดลอกโค้ดเวอร์ชันแรกมาตรง ๆ (src/feature.py, homography.py, blending.py ที่ commit 3555092)
# ---------------------------------------------------------------------------
def baseline_stitch(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    raw_matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    if len(good_matches) < 4:
        raise ValueError("not enough matches")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    panorama = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    panorama[0:h1, 0:w1] = img1
    return panorama, H


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def corners(width: int, height: int) -> np.ndarray:
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)


def corner_error(estimated: np.ndarray, truth: np.ndarray, size: tuple[int, int]) -> float:
    points = corners(*size)
    difference = cv2.perspectiveTransform(points, estimated) - cv2.perspectiveTransform(points, truth)
    return float(np.linalg.norm(difference, axis=2).mean())


def seam_step(output: np.ndarray, truth: np.ndarray, region: np.ndarray, sigma: float = 3.0) -> float:
    """P99 ของ gradient ของ residual (ผลลัพธ์ − ภาพจริง) หลัง low-pass

    noise และความเบลอจาก interpolation เป็นความถี่สูงจึงถูก blur ทิ้ง เหลือเฉพาะ "ขั้น" ความสว่าง
    ที่รอยต่อ รอยต่อแข็งให้ค่าสูง ส่วนการไล่ระดับกว้าง ๆ ของ multi-band ให้ค่าต่ำ
    """
    residual = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(
        truth, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(residual, (0, 0), sigma)
    magnitude = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3)) / 8
    return float(np.percentile(magnitude[region], 99)) if region.any() else float("nan")


def image_metrics(output: np.ndarray, valid: np.ndarray, truth: np.ndarray, footprints: list[np.ndarray],
                  captured_area: int) -> dict:
    """เทียบภาพผลลัพธ์กับฉากจริงที่ render ลงพิกัดเดียวกัน

    coverage = สัดส่วนพื้นที่ที่กล้องถ่ายไว้ซึ่งปรากฏในผลลัพธ์
    PSNR = ความเหมือนของสีทั้งภาพ (dB, ยิ่งสูงยิ่งดี)
    seam = seam step P99 ในแถบรอบส่วนซ้อนทับ (ยิ่งต่ำยิ่งเนียน)
    """
    union = np.zeros(valid.shape, bool)
    count = np.zeros(valid.shape, np.int32)
    for footprint in footprints:
        union |= footprint
        count += footprint
    # ตัดขอบนอกของพาโนรามาออก 5 px แต่เก็บขอบของแต่ละภาพที่อยู่ข้างใน (ตำแหน่งที่รอยต่อเกิด)
    region = cv2.erode((valid & union).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    seam_band = region & (cv2.dilate((count >= 2).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    difference = output[region].astype(np.float64) - truth[region].astype(np.float64)
    mse = float(np.mean(difference**2)) if region.any() else float("nan")
    return {
        "coverage": float((valid & union).sum() / max(captured_area, 1)),
        "psnr": 10 * np.log10(255**2 / mse) if mse > 0 else float("inf"),
        "seam": seam_step(output, truth, seam_band),
    }


def footprint_masks(to_output: list[np.ndarray], sizes: list[tuple[int, int]], canvas: tuple[int, int]):
    """พื้นที่ที่แต่ละภาพถ่ายไว้ในพิกัดผลลัพธ์ คืน (masks ภายใน canvas, พื้นที่รวมทั้งหมดรวมส่วนที่หลุดนอก canvas)"""
    polygons = [cv2.perspectiveTransform(corners(*size), matrix).reshape(-1, 2) for matrix, size in zip(to_output, sizes)]
    points = np.concatenate(polygons + [np.float32([[0, 0], canvas])])
    left, top = np.floor(points.min(axis=0)).astype(int)
    right, bottom = np.ceil(points.max(axis=0)).astype(int)
    union = np.zeros((bottom - top + 1, right - left + 1), np.uint8)
    masks = []
    for polygon in polygons:
        mask = np.zeros_like(union)
        cv2.fillPoly(mask, [np.round(polygon - [left, top]).astype(np.int32)], 1)
        union |= mask
        masks.append(mask[-top:-top + canvas[1], -left:-left + canvas[0]] > 0)
    return masks, int(union.sum())


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    key: str
    title: str
    count: int
    kwargs: dict = field(default_factory=dict)
    order: str = "sorted"  # sorted | reversed | shuffled
    distractor: bool = False


SCENARIOS = [
    Scenario("A", "2 ภาพ ซ้าย→ขวา ซ้อนกัน 40%", 2, dict(overlap=0.40, perspective=0.03)),
    Scenario("B", "2 ภาพ อัปโหลดสลับ (ขวา→ซ้าย)", 2, dict(overlap=0.40, perspective=0.03), order="reversed"),
    Scenario("C", "2 ภาพ แสงต่างกัน ±20% + vignetting", 2, dict(overlap=0.40, gain_range=0.2, vignetting=0.3)),
    Scenario("D", "2 ภาพ ซ้อนกันเพียง 25%", 2, dict(overlap=0.25, perspective=0.03)),
    Scenario("E", "3 ภาพ มุมเอียงมาก + แสงต่างกัน", 3, dict(perspective=0.06, gain_range=0.1, vignetting=0.2)),
    Scenario("F", "5 ภาพ สลับลำดับ + แสงต่างกัน", 5, dict(gain_range=0.2, vignetting=0.3), order="shuffled"),
    Scenario("G", "4 ภาพ + ภาพที่ไม่เกี่ยวข้อง 1 ภาพ", 4, dict(gain_range=0.1, vignetting=0.2), order="shuffled",
             distractor=True),
]


def build_case(scenario: Scenario, seed: int):
    synthetic = make_planar_set(count=scenario.count, seed=seed, **scenario.kwargs)
    rng = np.random.default_rng(seed)
    order = list(range(scenario.count))
    if scenario.order == "reversed":
        order.reverse()
    elif scenario.order == "shuffled":
        while order == sorted(order):
            order = list(rng.permutation(scenario.count))
    images = [synthetic.images[index] for index in order]
    truth = [synthetic.view_to_scene[index] for index in order]
    if scenario.distractor:
        position = int(rng.integers(0, len(images) + 1))
        images.insert(position, make_planar_set(count=1, seed=seed + 5000).images[0])
        truth.insert(position, None)
        order.insert(position, -1)
    return synthetic, images, truth, order


def evaluate_improved(synthetic, images, truth, order, settings: StitchSettings) -> dict:
    started = time.perf_counter()
    try:
        result = stitch(images, settings)
    except StitchError:
        return {"success": False, "runtime": time.perf_counter() - started}
    runtime = time.perf_counter() - started
    expected = [index for index, matrix in enumerate(truth) if matrix is not None]
    reference = result.reference
    errors = []
    for index in result.included:
        if index == reference or truth[index] is None:
            continue
        ground_truth = np.linalg.inv(truth[reference]) @ truth[index]
        size = (images[index].shape[1], images[index].shape[0])
        errors.append(corner_error(result.transforms[index], ground_truth, size))
    mean_error = float(np.mean(errors)) if errors else float("nan")
    ordered = [order[index] for index in result.included] == sorted(order[index] for index in expected)
    row = {
        "success": sorted(result.included) == expected and mean_error < 2.0,
        "corner_error": mean_error,
        "ordered": ordered,
        "runtime": runtime,
        "result": result,
    }
    if result.projection == "planar":
        canvas = result.canvas_size
        scene_to_output = result.output_transform(reference) @ np.linalg.inv(truth[reference])
        ground_truth_image = cv2.warpPerspective(synthetic.scene, scene_to_output, canvas, flags=cv2.INTER_LINEAR)
        footprints, captured = footprint_masks(
            [scene_to_output @ truth[index] for index in expected],
            [(images[index].shape[1], images[index].shape[0]) for index in expected],
            canvas,
        )
        valid = result.mask > 0
        row["variants"] = {}
        for name, method, exposure in [
            ("overlay", "none", False),
            ("feather", "feather", False),
            ("multiband", "multiband", False),
            ("multiband+gain", "multiband", True),
        ]:
            output = result.recompose(method, exposure)
            row["variants"][name] = image_metrics(output, valid, ground_truth_image, footprints, captured)
        row.update(row["variants"]["multiband+gain"])
        row["success"] = row["success"] and row["coverage"] >= 0.98
    return row


def evaluate_baseline(synthetic, images, truth) -> dict:
    started = time.perf_counter()
    try:
        panorama, homography = baseline_stitch(images[0], images[1])
    except Exception:  # โค้ดเวอร์ชันแรกไม่มี error handling
        return {"success": False, "runtime": time.perf_counter() - started}
    runtime = time.perf_counter() - started
    size = (images[1].shape[1], images[1].shape[0])
    error = corner_error(homography, np.linalg.inv(truth[0]) @ truth[1], size)
    canvas = (panorama.shape[1], panorama.shape[0])
    scene_to_output = np.linalg.inv(truth[0])
    ground_truth_image = cv2.warpPerspective(synthetic.scene, scene_to_output, canvas, flags=cv2.INTER_LINEAR)
    footprints, captured = footprint_masks(
        [scene_to_output @ matrix for matrix in truth], [(image.shape[1], image.shape[0]) for image in images], canvas
    )
    metrics = image_metrics(panorama, panorama.max(axis=2) > 0, ground_truth_image, footprints, captured)
    return {
        "success": error < 2.0 and metrics["coverage"] >= 0.98,
        "corner_error": error,
        "runtime": runtime,
        "panorama": panorama,
        **metrics,
    }


def mean(values) -> float:
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def fmt(value, digits=2, suffix="") -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def run_synthetic(seeds: int) -> str:
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "examples").mkdir(exist_ok=True)
    settings = StitchSettings(crop=False)
    rows, lines = [], []
    summary = []
    ablation: dict[str, list[dict]] = {}
    for scenario in SCENARIOS:
        improved_rows, baseline_rows = [], []
        for seed in range(seeds):
            case_seed = 100 * ord(scenario.key) + seed
            synthetic, images, truth, order = build_case(scenario, case_seed)
            improved = evaluate_improved(synthetic, images, truth, order, settings)
            improved_rows.append(improved)
            for name, metrics in improved.get("variants", {}).items():
                photometric = scenario.kwargs.get("gain_range", 0) > 0
                ablation.setdefault(name, []).append({**metrics, "photometric": photometric})
            baseline = evaluate_baseline(synthetic, images, truth) if scenario.count == 2 else None
            if baseline is not None:
                baseline_rows.append(baseline)
            rows.append({
                "scenario": scenario.key, "seed": case_seed,
                "improved_success": improved["success"], "improved_corner_error": improved.get("corner_error"),
                "improved_coverage": improved.get("coverage"), "improved_psnr": improved.get("psnr"),
                "improved_seam": improved.get("seam"), "improved_runtime": improved["runtime"],
                "baseline_success": None if baseline is None else baseline["success"],
                "baseline_corner_error": None if baseline is None else baseline.get("corner_error"),
                "baseline_coverage": None if baseline is None else baseline.get("coverage"),
                "baseline_psnr": None if baseline is None else baseline.get("psnr"),
                "baseline_seam": None if baseline is None else baseline.get("seam"),
                "baseline_runtime": None if baseline is None else baseline["runtime"],
            })
            if seed == 0 and "result" in improved:
                result = improved["result"]
                cv2.imwrite(str(OUTPUT / "examples" / f"{scenario.key}_improved.jpg"), result.recompose("multiband", True))
                if baseline is not None and "panorama" in baseline:
                    cv2.imwrite(str(OUTPUT / "examples" / f"{scenario.key}_baseline.jpg"), baseline["panorama"])
        summary.append((scenario, improved_rows, baseline_rows))
        print(f"  scenario {scenario.key} done", flush=True)

    with open(OUTPUT / "synthetic_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines.append(f"### ผลต่อสถานการณ์ ({seeds} ชุดภาพสุ่มต่อสถานการณ์)\n")
    lines.append("| สถานการณ์ | ภาพ | สำเร็จ (เดิม) | สำเร็จ (ใหม่) | Corner error px (เดิม → ใหม่) | Coverage (เดิม → ใหม่) | PSNR dB (เดิม → ใหม่) | Seam step P99 (เดิม → ใหม่) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for scenario, improved_rows, baseline_rows in summary:
        images = scenario.count + (1 if scenario.distractor else 0)
        improved_success = sum(row["success"] for row in improved_rows)
        if baseline_rows:
            baseline_success = f"{sum(row['success'] for row in baseline_rows)}/{len(baseline_rows)}"
            error = f"{fmt(mean(r.get('corner_error') for r in baseline_rows))} → {fmt(mean(r.get('corner_error') for r in improved_rows))}"
            coverage = f"{fmt(100 * mean(r.get('coverage') for r in baseline_rows), 1, '%')} → {fmt(100 * mean(r.get('coverage') for r in improved_rows), 1, '%')}"
            psnr = f"{fmt(mean(r.get('psnr') for r in baseline_rows))} → {fmt(mean(r.get('psnr') for r in improved_rows))}"
            seam = f"{fmt(mean(r.get('seam') for r in baseline_rows))} → {fmt(mean(r.get('seam') for r in improved_rows))}"
        else:
            baseline_success = "รองรับแค่ 2 ภาพ"
            error = f"— → {fmt(mean(r.get('corner_error') for r in improved_rows))}"
            coverage = f"— → {fmt(100 * mean(r.get('coverage') for r in improved_rows), 1, '%')}"
            psnr = f"— → {fmt(mean(r.get('psnr') for r in improved_rows))}"
            seam = f"— → {fmt(mean(r.get('seam') for r in improved_rows))}"
        lines.append(
            f"| {scenario.key}. {scenario.title} | {images} | {baseline_success} | {improved_success}/{len(improved_rows)} "
            f"| {error} | {coverage} | {psnr} | {seam} |"
        )
    ordered = [row["ordered"] for scenario, improved_rows, _ in summary if scenario.order == "shuffled"
               for row in improved_rows if "ordered" in row]
    lines.append(f"\nลำดับซ้าย→ขวาที่ระบบหาเองถูกต้อง (สถานการณ์ที่สลับลำดับ F, G): {sum(ordered)}/{len(ordered)} ชุด\n")

    lines.append("### Ablation ของ blending (สถานการณ์ A–G, ภาพที่ warp ชุดเดียวกัน)\n")
    lines.append("| วิธีรวมภาพ | PSNR (dB) ↑ | Seam step P99 ↓ แสงเท่ากัน (A, B, D) | Seam step P99 ↓ แสงต่างกัน (C, E, F, G) |")
    lines.append("|---|---:|---:|---:|")
    labels = {
        "overlay": "วางทับ (แบบเวอร์ชันแรก)",
        "feather": "Feather",
        "multiband": "Multi-band",
        "multiband+gain": "Multi-band + Exposure compensation (ค่าเริ่มต้น)",
    }
    for name, label in labels.items():
        values = ablation.get(name, [])
        same = [v for v in values if not v["photometric"]]
        different = [v for v in values if v["photometric"]]
        lines.append(
            f"| {label} | {fmt(mean(v['psnr'] for v in values))} | {fmt(mean(v['seam'] for v in same), 3)} "
            f"| {fmt(mean(v['seam'] for v in different), 3)} |"
        )

    runtime_new = mean(row["improved_runtime"] for row in rows if row["baseline_runtime"] is not None)
    runtime_old = mean(row["baseline_runtime"] for row in rows if row["baseline_runtime"] is not None)
    lines.append(f"\nเวลาเฉลี่ยกรณี 2 ภาพ 640×480: เดิม {runtime_old:.2f} s, ใหม่ {runtime_new:.2f} s "
                 "(ใหม่รวม exposure compensation + multi-band blending)\n")

    # กล้องหมุนมุมกว้าง
    lines.append("### กล้องหมุนมุมกว้าง (5 ภาพ, หมุนครั้งละ 30°, มุมรับภาพ 60° รวม ~180°)\n")
    rotation_rows = []
    for seed in range(max(3, seeds // 2)):
        synthetic = make_rotation_set(count=5, yaw_step=30, seed=900 + seed)
        started = time.perf_counter()
        try:
            result = stitch(synthetic.images, StitchSettings())
            rotation_rows.append({
                "success": sorted(result.included) == list(range(5)) and result.projection == "cylindrical",
                "focal_error": abs(result.focal - synthetic.focal) / synthetic.focal,
                "runtime": time.perf_counter() - started,
            })
            if seed == 0:
                cv2.imwrite(str(OUTPUT / "examples" / "rotation_cylindrical.jpg"), result.panorama)
        except StitchError:
            rotation_rows.append({"success": False, "focal_error": float("nan"), "runtime": time.perf_counter() - started})
        try:
            stitch(synthetic.images, StitchSettings(projection="planar"))
            rotation_rows[-1]["planar_failed"] = False
        except StitchError:
            rotation_rows[-1]["planar_failed"] = True
    lines.append("| ชุดภาพ | ต่อครบ 5 ภาพ (Auto → Cylindrical) | Planar อย่างเดียวถูกปฏิเสธ | Focal error เฉลี่ย | เวลาเฉลี่ย |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {len(rotation_rows)} | {sum(r['success'] for r in rotation_rows)}/{len(rotation_rows)} "
        f"| {sum(r['planar_failed'] for r in rotation_rows)}/{len(rotation_rows)} "
        f"| {fmt(100 * mean(r['focal_error'] for r in rotation_rows), 1, '%')} | {fmt(mean(r['runtime'] for r in rotation_rows), 2, ' s')} |"
    )

    # SIFT vs ORB
    lines.append("\n### SIFT เทียบ ORB (สถานการณ์ E และ F)\n")
    lines.append("| Detector | สำเร็จ | Corner error (px) | เวลาเฉลี่ย (s) |")
    lines.append("|---|---:|---:|---:|")
    for detector in ("SIFT", "ORB"):
        detector_rows = []
        for scenario in [s for s in SCENARIOS if s.key in "EF"]:
            for seed in range(seeds):
                synthetic, images, truth, order = build_case(scenario, 100 * ord(scenario.key) + seed)
                row = evaluate_improved(synthetic, images, truth, order, StitchSettings(detector=detector, crop=False))
                detector_rows.append(row)
        lines.append(
            f"| {detector} | {sum(r['success'] for r in detector_rows)}/{len(detector_rows)} "
            f"| {fmt(mean(r.get('corner_error') for r in detector_rows))} | {fmt(mean(r['runtime'] for r in detector_rows))} |"
        )

    text = "\n".join(lines)
    (OUTPUT / "synthetic_summary.md").write_text(text, encoding="utf-8")
    return text


def run_real(folder: Path) -> str:
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "real").mkdir(exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    sets = [path for path in sorted(folder.iterdir()) if path.is_dir()]
    if any(path.suffix.lower() in extensions for path in folder.iterdir()):
        sets.insert(0, folder)
    lines = [
        "| ชุดภาพ | ภาพ | ต่อได้ | Projection | Inliers เฉลี่ย (คู่ที่ใช้) | Inlier ratio | RMSE (px) | เวลา (s) | หมายเหตุ |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for path in sets:
        files = sorted(file for file in path.iterdir() if file.suffix.lower() in extensions)
        if len(files) < 2:
            continue
        images = [decode_image(file.read_bytes(), 1200)[0] for file in files]
        started = time.perf_counter()
        try:
            result = stitch(images, StitchSettings())
        except StitchError as error:
            lines.append(f"| {path.name} | {len(images)} | 0 | — | — | — | — | {time.perf_counter() - started:.2f} | {error} |")
            continue
        tree = [pair.used_estimate(result.projection) for pair in result.pairs if pair.in_tree]
        cv2.imwrite(str(OUTPUT / "real" / f"{path.name}.jpg"), result.panorama)
        notes = "; ".join(result.warnings) or "-"
        projection = result.projection + (f" (f = {result.focal:.0f} px)" if result.focal else "")
        lines.append(
            f"| {path.name} | {len(images)} | {len(result.included)} | {projection} "
            f"| {mean(e.num_inliers for e in tree):.0f} | {100 * mean(e.inlier_ratio for e in tree):.0f}% "
            f"| {mean(e.reprojection_rmse for e in tree):.2f} | {result.timings['total']:.2f} | {notes} |"
        )
        if len(images) >= 2:
            try:
                baseline, _ = baseline_stitch(images[0], images[1])
                cv2.imwrite(str(OUTPUT / "real" / f"{path.name}_baseline_first_two.jpg"), baseline)
            except Exception:
                pass
    text = "\n".join(lines)
    (OUTPUT / "real_summary.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, help="โฟลเดอร์ภาพจริง (แต่ละโฟลเดอร์ย่อยคือ 1 ชุด)")
    parser.add_argument("--seeds", type=int, default=10, help="จำนวนชุดภาพสุ่มต่อสถานการณ์")
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run_real(arguments.real) if arguments.real else run_synthetic(arguments.seeds))
