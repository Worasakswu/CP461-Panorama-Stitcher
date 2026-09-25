"""หน้าเว็บ Streamlit ของ Automatic Panorama Stitcher (CP461 Introduction to Computer Vision)"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.image_io import decode_image, encode_image
from src.stitcher import MAX_IMAGES, StitchError, StitchSettings, stitch
from src.synthetic import make_demo_set
from src.visualize import draw_keypoints, draw_layout, draw_matches, seam_close_up

st.set_page_config(page_title="Automatic Panorama Stitcher", page_icon="📷", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stFileUploaderDropzone"] {border: 1.5px dashed #4f6bed; border-radius: 16px;}
    .step {display:inline-block; color:#3148c8; background:#eef1ff; border-radius:999px;
           padding:.3rem .75rem; margin:0 .35rem .45rem 0; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECTION_LABELS = {
    "auto": "อัตโนมัติ (Planar → Cylindrical เมื่อมุมกว้าง)",
    "planar": "Planar (Homography บนระนาบภาพอ้างอิง)",
    "cylindrical": "Cylindrical (ฉายลงทรงกระบอก)",
}
BLEND_LABELS = {
    "multiband": "Multi-band (Laplacian pyramid)",
    "feather": "Feather (ถ่วงน้ำหนักตามระยะจากขอบ)",
    "none": "ไม่ blend (วางทับแบบเวอร์ชันแรก)",
}
SHORT_BLEND_LABELS = {"none": "วางทับ (เวอร์ชันแรก)", "feather": "Feather", "multiband": "Multi-band"}
UPLOAD_SOURCE = "อัปโหลดภาพของฉัน"
SAMPLE_SOURCE = "ใช้ชุดภาพตัวอย่าง"
SYNTHETIC_SAMPLE = "ภาพจำลอง (สังเคราะห์)"
SAMPLE_ROOT = Path(__file__).resolve().parent / "test_images"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
TIMING_LABELS = {
    "features": "ตรวจหา keypoints + descriptors",
    "matching": "จับคู่ KNN + ratio test + RANSAC ทุกคู่ภาพ",
    "geometry": "เลือกโครงข่ายภาพ + รวม transform",
    "warping": "Warp ภาพลง canvas",
    "exposure": "Exposure compensation",
    "blending": "Seam + Blending",
    "crop": "Auto-crop",
    "total": "รวมทั้งหมด",
}

st.title("📷 Automatic Panorama Stitcher")
st.caption("CP461 Introduction to Computer Vision · ต่อภาพหลายภาพเป็นพาโนรามาไร้รอยต่อโดยอัตโนมัติ")
st.markdown(
    f'<span class="step">1 อัปโหลด 2–{MAX_IMAGES} ภาพ (ลำดับใดก็ได้)</span>'
    '<span class="step">2 SIFT/ORB + Lowe\'s ratio test</span>'
    '<span class="step">3 RANSAC Homography</span>'
    '<span class="step">4 Warping + Exposure compensation</span>'
    '<span class="step">5 Multi-band blending + Auto-crop</span>',
    unsafe_allow_html=True,
)


def show_bgr(image: np.ndarray, caption: str | None = None) -> None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


@st.cache_data(show_spinner=False, max_entries=40)
def load_image(data: bytes, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    return decode_image(data, max_side)


@st.cache_data(show_spinner=False)
def demo_images() -> list[np.ndarray]:
    return make_demo_set()


def sample_sets() -> dict[str, list[Path]]:
    """ชุดภาพตัวอย่างใน test_images/ (โฟลเดอร์ละ 1 ชุด ต้องมีอย่างน้อย 2 ภาพ)"""
    sets = {}
    if SAMPLE_ROOT.is_dir():
        for folder in sorted(path for path in SAMPLE_ROOT.iterdir() if path.is_dir()):
            files = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if len(files) >= 2:
                sets[folder.name] = files
    return sets


def shuffled_order(count: int) -> list[int]:
    order = list(range(count))
    generator = random.Random(461)
    while count > 1 and order == sorted(order):
        generator.shuffle(order)
    return order


source = st.radio("แหล่งภาพ", [UPLOAD_SOURCE, SAMPLE_SOURCE], horizontal=True, label_visibility="collapsed")
uploaded_files = []
samples = sample_sets()
if source == UPLOAD_SOURCE:
    uploaded_files = st.file_uploader(
        "ลากภาพมาวาง หรือกดเพื่อเลือกหลายไฟล์พร้อมกัน",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="ภาพติดกันควรซ้อนทับกันราว 30–50% ไม่ต้องเรียงลำดับ ระบบหาลำดับให้เอง",
    ) or []
    with st.expander("เคล็ดลับการเลือกภาพให้ต่อได้สวย"):
        st.markdown(
            "- ใช้ภาพชุดที่ถ่ายจากจุดเดียวกันโดย **หมุนกล้อง** ไปทางซ้าย/ขวา ไม่ได้เดินขยับ (ลด parallax)\n"
            "- ถ้าหาภาพจากอินเทอร์เน็ต ให้เลือกชุดที่ตั้งใจถ่ายไว้ทำพาโนรามา (ภาพต่อเนื่องจากกล้องตัวเดียว)\n"
            "- ภาพที่อยู่ติดกันควรซ้อนทับกันประมาณ 30–50%\n"
            "- ฉากที่มีรายละเอียด (ตึก ต้นไม้ ป้าย) ต่อได้ดีกว่าท้องฟ้าหรือผนังเรียบ"
        )
else:
    sample_choice, shuffle_column, distractor_column = st.columns([2, 1, 1], vertical_alignment="bottom")
    sample_name = sample_choice.selectbox(
        "ชุดภาพตัวอย่าง", list(samples) + [SYNTHETIC_SAMPLE],
        help="ภาพจริงจากโฟลเดอร์ test_images/ ใน repo หรือภาพจำลองที่สร้างขึ้นเอง",
    )
    shuffle_samples = shuffle_column.toggle("สลับลำดับภาพ", help="ทดสอบว่าระบบหาลำดับซ้าย→ขวาได้เอง")
    add_distractor = distractor_column.toggle(
        "เพิ่มภาพแปลกปลอม", disabled=not samples, help="แทรกภาพจากชุดอื่น 1 ภาพ ทดสอบว่าระบบตัดภาพนั้นทิ้งได้"
    )
preview_area = st.container()

with st.expander("⚙️ ตั้งค่าขั้นสูง (สำหรับสาธิตผลของแต่ละขั้นตอน)"):
    first, second, third = st.columns(3)
    detector = first.selectbox(
        "Feature detector", ["SIFT", "ORB"],
        help="SIFT ทนต่อการเปลี่ยนสเกล/มุมมองได้ดี (ค่าเริ่มต้น) ส่วน ORB เป็น binary descriptor ที่เร็วกว่า",
    )
    ratio = second.slider("Lowe's ratio threshold", 0.50, 0.90, 0.75, 0.05, help="ค่ายิ่งต่ำยิ่งคัดคู่จุดเข้มงวด")
    ransac_threshold = third.slider("RANSAC reprojection threshold (px)", 1.0, 10.0, 4.0, 0.5)
    first, second, third = st.columns(3)
    projection = first.selectbox("Projection", list(PROJECTION_LABELS), format_func=PROJECTION_LABELS.get)
    blend_method = second.selectbox("Blending", list(BLEND_LABELS), format_func=BLEND_LABELS.get)
    max_side = third.select_slider(
        "ความละเอียดที่ใช้ประมวลผล (ด้านยาว, px)", [800, 1000, 1200, 1600, 2000], value=1200,
        help="ภาพที่ใหญ่กว่านี้จะถูกย่อก่อนประมวลผล เพื่อให้เร็วและไม่เกินหน่วยความจำของ Streamlit Cloud",
    )
    first, second, third = st.columns(3)
    exposure = first.toggle("Exposure compensation", value=True, help="ปรับ gain ของแต่ละภาพให้ความสว่างในส่วนซ้อนทับเท่ากัน")
    crop = second.toggle("Auto-crop ขอบดำ", value=True)
    max_features = third.select_slider("จำนวน keypoints สูงสุดต่อภาพ", [1000, 2000, 3000, 4000, 6000, 8000], value=4000)
    focal = 0
    if projection == "cylindrical":
        focal = st.number_input("Focal length (px ที่ความละเอียดประมวลผล, 0 = ประมาณจาก homography)", 0, 20000, 0, 50)

settings = StitchSettings(
    detector=detector,
    max_features=max_features,
    ratio=ratio,
    ransac_threshold=ransac_threshold,
    projection=projection,
    blend=blend_method,
    exposure_compensation=exposure,
    crop=crop,
    max_side=max_side,
    focal=float(focal) or None,
)

images: list[np.ndarray] = []
names: list[str] = []
digests: list[str] = []
if source == UPLOAD_SOURCE:
    if not uploaded_files:
        st.info(f"เริ่มจากอัปโหลดภาพที่ซ้อนทับกัน 2–{MAX_IMAGES} ภาพ หรือเลือก “{SAMPLE_SOURCE}” เพื่อทดลองทันที")
        st.stop()
    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        try:
            image, original_size = load_image(data, max_side)
        except ValueError as error:
            st.error(f"{uploaded.name}: {error}")
            continue
        images.append(image)
        names.append(f"{uploaded.name} ({original_size[0]}×{original_size[1]})")
        digests.append(hashlib.sha1(data).hexdigest())
else:
    entries = []  # (ภาพ, ชื่อที่แสดง, key สำหรับ signature)
    if sample_name == SYNTHETIC_SAMPLE:
        entries = [(image, f"ภาพจำลอง {index + 1}", f"synthetic-{index}") for index, image in enumerate(demo_images())]
        st.caption("ภาพจำลอง 4 ภาพของฉากเดียวกันจากหลายมุม แสงแต่ละภาพไม่เท่ากัน มีขอบภาพมืด และสลับลำดับไว้แล้ว")
    else:
        for path in samples[sample_name]:
            image, original_size = load_image(path.read_bytes(), max_side)
            entries.append((image, f"{path.name} ({original_size[0]}×{original_size[1]})", f"{sample_name}/{path.name}"))
    if shuffle_samples:
        entries = [entries[index] for index in shuffled_order(len(entries))]
    if add_distractor:
        other = next((name for name in samples if name != sample_name), None)
        if other is not None:
            path = samples[other][len(samples[other]) // 2]
            image, _ = load_image(path.read_bytes(), max_side)
            entries.insert(len(entries) // 2, (image, f"{other}/{path.name} · ภาพแปลกปลอม", f"{other}/{path.name}"))
    images = [entry[0] for entry in entries]
    names = [entry[1] for entry in entries]
    digests = [entry[2] for entry in entries]

if len(images) > MAX_IMAGES:
    st.warning(f"ใช้เฉพาะ {MAX_IMAGES} ภาพแรก (อัปโหลดมา {len(images)} ภาพ)")
    images, names, digests = images[:MAX_IMAGES], names[:MAX_IMAGES], digests[:MAX_IMAGES]

with preview_area:
    per_row = min(5, max(1, len(images)))
    for start in range(0, len(images), per_row):
        columns = st.columns(per_row)
        for column, index in zip(columns, range(start, min(start + per_row, len(images)))):
            with column:
                show_bgr(images[index], f"ภาพ {index + 1} · {names[index]}")

if len(images) < 2:
    st.warning("ต้องมีภาพที่อ่านได้อย่างน้อย 2 ภาพ")
    st.stop()

signature = hashlib.sha1((repr(digests) + repr(settings)).encode()).hexdigest()
if st.button("🧩 สร้างภาพพาโนรามา", type="primary", width="stretch"):
    progress_bar = st.progress(0.0, text="เริ่มประมวลผล")
    try:
        result = stitch(images, settings, progress=lambda fraction, message: progress_bar.progress(min(fraction, 1.0), text=message))
        failure = None
    except StitchError as error:
        result, failure = None, error
    progress_bar.empty()
    st.session_state["stitch"] = {"signature": signature, "result": result, "error": failure, "names": names}

state = st.session_state.get("stitch")
if state is None:
    st.stop()
if state["signature"] != signature:
    st.info("ภาพหรือการตั้งค่าเปลี่ยนไปจากผลล่าสุด กด “สร้างภาพพาโนรามา” อีกครั้ง")
    st.stop()


def pair_rows(pairs, projection: str = "planar") -> list[dict]:
    rows = []
    for pair in sorted(pairs, key=lambda pair: (-pair.in_tree, -pair.estimate.num_inliers)):
        estimate = pair.used_estimate(projection)
        rows.append({
            "คู่ภาพ": f"{pair.first + 1} ↔ {pair.second + 1}",
            "Good matches": pair.estimate.num_matches,
            "RANSAC inliers": estimate.num_inliers,
            "Inlier ratio": f"{estimate.inlier_ratio * 100:.0f}%",
            "RMSE (px)": "—" if not np.isfinite(estimate.reprojection_rmse) else f"{estimate.reprojection_rmse:.2f}",
            "ผลตรวจ": "ผ่าน" if estimate.ok else (estimate.failure_reason or "ไม่ผ่าน"),
            "ใช้ต่อภาพ": "✓" if pair.in_tree else "",
        })
    return rows


if state["error"] is not None:
    failure = state["error"]
    st.error(str(failure))
    if failure.pairs:
        st.markdown("**ผลการจับคู่ทุกคู่ภาพ**")
        st.dataframe(pair_rows(failure.pairs), hide_index=True)
    st.stop()

result = state["result"]
names = state["names"]
width, height = result.panorama.shape[1], result.panorama.shape[0]

st.subheader("ผลลัพธ์")
metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("ภาพที่ต่อได้", f"{len(result.included)}/{len(result.images)}", border=True)
metric_two.metric("Projection", result.projection.capitalize(), border=True)
metric_three.metric("ขนาดพาโนรามา", f"{width}×{height}", border=True)
metric_four.metric("เวลาประมวลผล", f"{result.timings['total']:.1f} วินาที", border=True)

order_text = " → ".join(f"ภาพ {index + 1}" for index in result.included)
st.success(
    f"ต่อภาพสำเร็จ · ลำดับจากซ้ายไปขวา: {order_text} · ภาพอ้างอิง: ภาพ {result.reference + 1} · "
    f"{BLEND_LABELS[result.settings.blend]}"
)
for warning in result.warnings:
    st.warning(warning)
show_bgr(result.panorama)

if "downloads" not in state:
    state["downloads"] = (encode_image(result.panorama, ".png"), encode_image(result.panorama, ".jpg"))
download_png, download_jpg = st.columns(2)
# on_click="ignore": ดาวน์โหลดได้โดยไม่ต้อง rerun ทั้งหน้า
download_png.download_button(
    "ดาวน์โหลด PNG (lossless)", state["downloads"][0], "panorama.png", "image/png", on_click="ignore", width="stretch"
)
download_jpg.download_button(
    "ดาวน์โหลด JPG", state["downloads"][1], "panorama.jpg", "image/jpeg", on_click="ignore", width="stretch",
    type="primary",
)

# on_change="rerun" ทำให้แท็บจำตำแหน่งที่เลือกไว้ และข้ามการวาดเนื้อหาของแท็บที่ไม่ได้เปิด
matching_tab, layout_tab, blending_tab, technical_tab = st.tabs(
    ["🔍 Feature Matching", "🧭 การวางภาพ (Homography)", "🎨 เปรียบเทียบ Blending", "📊 ข้อมูลทางเทคนิค"],
    key="result_tab",
    on_change="rerun",
)


def matching_content() -> None:
    st.markdown(
        f"ระบบหา **{result.settings.detector}** keypoints ในทุกภาพ จับคู่ descriptor ด้วย KNN (k=2) "
        f"คัดด้วย Lowe's ratio < {result.settings.ratio:.2f} แล้วใช้ **RANSAC** "
        f"(threshold {result.settings.ransac_threshold:.1f} px) หา Homography และตัด outliers ของ **ทุกคู่ภาพ** "
        "คู่ที่ผ่านการตรวจจะถูกใช้สร้างโครงข่ายภาพ (maximum spanning tree ตามจำนวน inliers)"
    )
    keypoint_columns = st.columns(min(5, len(result.images)))
    for index, feature in enumerate(result.features):
        keypoint_columns[index % len(keypoint_columns)].metric(f"Keypoints ภาพ {index + 1}", f"{len(feature):,}")
    st.dataframe(pair_rows(result.pairs, result.projection), hide_index=True)

    pairs = sorted(result.pairs, key=lambda pair: (-pair.in_tree, -pair.estimate.num_inliers))
    selected = st.selectbox(
        "ดูคู่จุดของคู่ภาพ",
        range(len(pairs)),
        format_func=lambda number: (
            f"ภาพ {pairs[number].first + 1} ↔ ภาพ {pairs[number].second + 1} · "
            f"inliers {pairs[number].estimate.num_inliers}/{pairs[number].estimate.num_matches}"
            + (" · ใช้ต่อภาพ" if pairs[number].in_tree else "")
        ),
    )
    pair = pairs[selected]
    show_outliers = st.toggle("แสดง outliers (เส้นสีแดง)", value=True)
    show_bgr(
        draw_matches(
            result.images[pair.first], result.images[pair.second], pair.first_points, pair.second_points,
            pair.estimate.inlier_mask, show_outliers,
        ),
        "เส้นสีเขียว = RANSAC inliers ที่ใช้คำนวณ Homography · เส้นสีแดง = คู่ที่ผ่าน ratio test แต่ไม่สอดคล้องกับ Homography",
    )
    pair_one, pair_two, pair_three, pair_four = st.columns(4)
    pair_one.metric("Good matches", pair.estimate.num_matches)
    pair_two.metric("RANSAC inliers", pair.estimate.num_inliers)
    pair_three.metric("Inlier ratio", f"{pair.estimate.inlier_ratio * 100:.1f}%")
    pair_four.metric(
        "Reprojection RMSE",
        "—" if not np.isfinite(pair.estimate.reprojection_rmse) else f"{pair.estimate.reprojection_rmse:.2f} px",
    )
    if pair.estimate.failure_reason:
        st.info(f"คู่นี้ไม่ถูกใช้: {pair.estimate.failure_reason}")
    if pair.estimate.matrix is not None:
        with st.expander(f"Homography ที่แปลงภาพ {pair.second + 1} → ภาพ {pair.first + 1}"):
            st.code(np.array2string(pair.estimate.matrix, precision=5, suppress_small=True))

    with st.expander("ดู keypoints ของแต่ละภาพ"):
        keypoint_image = st.selectbox(
            "ภาพ", range(len(result.images)), format_func=lambda index: f"ภาพ {index + 1} · {names[index]}"
        )
        show_bgr(
            draw_keypoints(result.images[keypoint_image], result.features[keypoint_image]),
            "600 keypoints ที่แรงที่สุด วงกลมแสดงสเกล เส้นแสดงทิศทาง (orientation)",
        )


def layout_content() -> None:
    show_bgr(
        draw_layout(result),
        "สีแสดงว่าส่วนไหนของพาโนรามามาจากภาพใด · เส้นขาว = seam ที่ใช้ blend · กรอบเหลือง = บริเวณหลัง auto-crop",
    )
    if result.projection == "planar":
        st.markdown(
            f"**Planar projection:** ทุกภาพถูก warp ลงบนระนาบของ **ภาพ {result.reference + 1}** "
            "(ภาพที่อยู่กลางโครงข่าย) ด้วย Homography ที่คูณต่อกันตามเส้นทางในโครงข่ายภาพ"
        )
    else:
        focal_source = "ประมาณจาก homography" if result.focal_estimated else "ค่าเริ่มต้น"
        st.markdown(
            f"**Cylindrical projection:** ฉายทุกภาพลงทรงกระบอกรัศมี f = **{result.focal:.0f} px** ({focal_source}) "
            "แล้วหา similarity transform ด้วย RANSAC บนพิกัดทรงกระบอก เหมาะกับพาโนรามามุมกว้างที่ระนาบเดียวรองรับไม่ได้"
        )
    with st.expander("เมทริกซ์ transform ของแต่ละภาพ → ภาพอ้างอิง"):
        for index in result.included:
            st.markdown(f"**ภาพ {index + 1}**" + (" (ภาพอ้างอิง)" if index == result.reference else ""))
            st.code(np.array2string(result.transforms[index], precision=5, suppress_small=True))


def blending_content() -> None:
    st.markdown(
        "เปรียบเทียบการรวมภาพ 3 แบบจากภาพที่ warp แล้วชุดเดียวกัน ภาพเล็กด้านบนคือการซูมเข้าไปที่ seam "
        f"(Exposure compensation: {'เปิด' if result.settings.exposure_compensation else 'ปิด'})"
    )
    if state.get("comparison") is None:
        with st.spinner("กำลัง blend ใหม่ทั้ง 3 แบบ…"):
            state["comparison"] = {method: result.recompose(method) for method in reversed(BLEND_LABELS)}
    window = seam_close_up(result)
    columns = st.columns(len(state["comparison"]))
    for column, (method, image) in zip(columns, state["comparison"].items()):
        with column:
            st.markdown(f"**{SHORT_BLEND_LABELS[method]}**")
            if window is not None:
                x, y, side, _ = window
                show_bgr(image[y:y + side, x:x + side], "ซูมที่ seam")
            show_bgr(image, "ภาพเต็ม")


def technical_content() -> None:
    st.markdown("**เวลาที่ใช้แต่ละขั้นตอน**")
    st.dataframe(
        [{"ขั้นตอน": TIMING_LABELS.get(name, name), "เวลา (วินาที)": round(value, 3)} for name, value in result.timings.items()],
        hide_index=True,
    )
    if result.gains is not None:
        st.markdown("**Gain ที่ใช้ปรับแสงแต่ละภาพ (B, G, R)**")
        st.dataframe(
            [
                {"ภาพ": warp.index + 1, "B": round(float(gain[0]), 3), "G": round(float(gain[1]), 3), "R": round(float(gain[2]), 3)}
                for warp, gain in zip(result.warps, result.gains)
            ],
            hide_index=True,
        )
    st.markdown("**ขนาดภาพที่ใช้ประมวลผล**")
    st.dataframe(
        [{"ภาพ": index + 1, "ไฟล์": names[index], "ขนาดที่ใช้": f"{image.shape[1]}×{image.shape[0]}"}
         for index, image in enumerate(result.images)],
        hide_index=True,
    )
    uncropped, union_mask = st.columns(2)
    with uncropped:
        show_bgr(result.panorama_uncropped, "ก่อน auto-crop")
    with union_mask:
        st.image(result.mask, caption="Mask บริเวณที่มีข้อมูลภาพ", width="stretch")
    st.caption(f"การตั้งค่าที่ใช้: {result.settings}")


for tab, render in [
    (matching_tab, matching_content),
    (layout_tab, layout_content),
    (blending_tab, blending_content),
    (technical_tab, technical_content),
]:
    if tab.open:
        with tab:
            render()
