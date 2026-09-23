"""รันหน้าเว็บ Streamlit จริงด้วย AppTest (ไม่ต้องเปิดเบราว์เซอร์)"""

import io
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.image_io import encode_image
from src.synthetic import make_planar_set

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_empty_upload_shows_start_prompt():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.info


def test_demo_images_produce_panorama_and_comparison():
    app = AppTest.from_file(APP).run(timeout=30)
    app.radio[0].set_value("ใช้ภาพตัวอย่าง (สังเคราะห์)").run(timeout=60)
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    assert any(block.value.startswith("ต่อภาพสำเร็จ") for block in app.success)
    assert any(metric.label == "ภาพที่ต่อได้" and metric.value == "4/4" for metric in app.metric)

    # เปิดแท็บเปรียบเทียบ blending (แท็บแบบ stateful อ่าน/เขียนผ่าน session_state ได้)
    app.session_state["result_tab"] = "🎨 เปรียบเทียบ Blending"
    app.run(timeout=120)
    assert not app.exception
    headings = [block.value for block in app.markdown]
    assert "**Multi-band**" in headings
    assert "**วางทับ (เวอร์ชันแรก)**" in headings


def test_uploaded_files_are_stitched():
    synthetic = make_planar_set(count=2, seed=5)
    uploads = []
    for index, image in enumerate(reversed(synthetic.images)):
        buffer = io.BytesIO(encode_image(image, ".jpg"))
        buffer.name = f"photo-{index}.jpg"
        uploads.append(buffer)
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(APP).run(timeout=30)
        app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    assert any(metric.label == "ภาพที่ต่อได้" and metric.value == "2/2" for metric in app.metric)


def test_unrelated_uploads_show_error_with_pair_table():
    images = [make_planar_set(count=1, seed=seed).images[0] for seed in (61, 62)]
    uploads = []
    for index, image in enumerate(images):
        buffer = io.BytesIO(encode_image(image, ".png"))
        buffer.name = f"unrelated-{index}.png"
        uploads.append(buffer)
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(APP).run(timeout=30)
        app.button[0].click().run(timeout=120)
    assert not app.exception
    assert app.error
    assert "ซ้อนทับ" in app.error[0].value
