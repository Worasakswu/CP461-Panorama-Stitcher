#  ระบบต่อภาพพาโนรามาอัตโนมัติ (Automatic Panorama Stitcher)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cp461-panorama-stitcher-h2tewyfx9y38z45nk7bckd.streamlit.app/)

เว็บแอปพลิเคชัน Computer Vision สำหรับต่อภาพถ่ายที่มีส่วนซ้อนทับกัน **2–8 ภาพ** ให้กลายเป็นภาพพาโนรามามุมกว้างไร้รอยต่อโดยอัตโนมัติ พัฒนาขึ้นสำหรับรายวิชา **CP461 Introduction to Computer Vision**

🔗 **ลิงก์เข้าใช้งานระบบ:** [cp461-panorama-stitcher.streamlit.app](https://cp461-panorama-stitcher-h2tewyfx9y38z45nk7bckd.streamlit.app/)

---

##  ขั้นตอนการทำงาน

1. อัปโหลดภาพ 2–8 ภาพ **ลำดับใดก็ได้** หรือเลือก “ใช้ชุดภาพตัวอย่าง” จาก `test_images/` (มีตัวเลือกสลับลำดับภาพและเพิ่มภาพแปลกปลอมไว้ทดสอบ edge case)
2. หมุนภาพตาม EXIF และย่อให้ด้านยาวไม่เกิน 1,200 px (ปรับได้)
3. ตรวจหา **SIFT** (ค่าเริ่มต้น) หรือ **ORB** keypoints และสร้าง descriptors ของทุกภาพ
4. จับคู่ descriptor ของ **ทุกคู่ภาพ** ด้วย KNN (k = 2) แล้วคัดด้วย **Lowe's ratio test**
5. หา **Homography ด้วย RANSAC** ตัด outliers และตรวจความน่าเชื่อถือ (จำนวน inliers, เกณฑ์ของ Brown & Lowe, ภาพไม่กลับด้าน/ไม่ยืดผิดปกติ)
6. สร้างโครงข่ายภาพจากคู่ที่ผ่านการตรวจ (maximum spanning tree ตามจำนวน inliers) เลือกภาพกลางโครงข่ายเป็นภาพอ้างอิง แล้วคูณ Homography ต่อกันไปถึงทุกภาพ ภาพที่ไม่เกี่ยวข้องถูกตัดออกพร้อมแจ้งเตือน
7. **Warp** ทุกภาพลงระนาบเดียวกัน (Planar) หรือถ้ามุมกว้างเกินจะเปลี่ยนเป็น **Cylindrical projection** อัตโนมัติ โดยประมาณ focal length จาก Homography
8. **Exposure compensation** ปรับ gain ของแต่ละภาพให้แสงในส่วนซ้อนทับเท่ากัน
9. วาง seam กลางส่วนซ้อนทับ แล้วรวมภาพด้วย **Multi-band blending (Laplacian pyramid)**
10. **Auto-crop** ตัดขอบดำ แล้วดาวน์โหลดเป็น PNG/JPG

หน้าเว็บแสดงหลักฐานของทุกขั้นตอน: จำนวน keypoints, ตาราง good matches/inliers/RMSE ของทุกคู่ภาพ, ภาพเส้นคู่จุด inlier (เขียว) และ outlier (แดง), เมทริกซ์ Homography, แผนผังว่าส่วนไหนของพาโนรามามาจากภาพใด, ภาพเปรียบเทียบวิธี blending แบบซูมที่ seam และเวลาที่ใช้แต่ละขั้นตอน

ดู [ผลประเมินก่อน–หลังและข้อจำกัด](EVALUATION.md) และ [แนวทางเขียนรายงาน/นำเสนอ](PROJECT_REPORT_GUIDE.md)

---

##  เทคโนโลยีที่ใช้ (Tech Stack)

* **ภาษาหลัก:** Python 3.10+
* **Computer Vision & Math:** OpenCV (`opencv-python-headless`), NumPy, Pillow
* **Frontend & Cloud Deployment:** Streamlit, Streamlit Community Cloud
* **Testing:** pytest + Streamlit AppTest

---

##  โครงสร้างโฟลเดอร์ใน Repository

```text
CP461-Panorama-Stitcher/
├── app.py                  # หน้าเว็บ Streamlit
├── requirements.txt        # ไลบรารีที่ระบบต้องใช้
├── requirements-dev.txt    # ไลบรารีสำหรับรันชุดทดสอบ
├── EVALUATION.md           # ผลประเมินเทียบเวอร์ชันแรก
├── PROJECT_REPORT_GUIDE.md # แนวทางรายงานและการนำเสนอ
├── .streamlit/config.toml  # theme และขนาดไฟล์อัปโหลด
├── src/                    # โมดูลประมวลผล Computer Vision
│   ├── image_io.py         # อ่านภาพ + EXIF orientation, ย่อภาพ, บันทึก PNG/JPG
│   ├── feature.py          # SIFT/ORB, KNN matching, Lowe's ratio test
│   ├── homography.py       # RANSAC, ตรวจความน่าเชื่อถือ, ประมาณ focal length
│   ├── warping.py          # Cylindrical projection, canvas, perspective warping
│   ├── blending.py         # Gain compensation, seam, feather/multi-band blending, auto-crop
│   ├── stitcher.py         # Pipeline หลักสำหรับหลายภาพ
│   ├── visualize.py        # ภาพประกอบบนหน้าเว็บ
│   └── synthetic.py        # ภาพสังเคราะห์ที่รู้ ground truth (ทดสอบ/ประเมินผล/เดโม)
├── scripts/evaluate.py     # สคริปต์ประเมินผล (ภาพสังเคราะห์ หรือภาพถ่ายจริง)
├── tests/                  # ชุดทดสอบอัตโนมัติ
├── notebooks/
└── test_images/            # ชุดภาพทดสอบจากอินเทอร์เน็ต (โฟลเดอร์ละ 1 ชุด) ใช้เป็นภาพตัวอย่างบนเว็บด้วย
```

##  ขั้นตอนการติดตั้งและรันบนเครื่องตัวเอง (Local)

```bash
# 1. Clone Repository ลงเครื่อง
git clone https://github.com/Worasakswu/CP461-Panorama-Stitcher.git
cd CP461-Panorama-Stitcher

# 2. สร้าง virtual environment และติดตั้งไลบรารี
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

# 3. สั่งรัน Web App แล้วเปิด http://localhost:8501
python -m streamlit run app.py
```

##  ทดสอบและประเมินผล

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                              # 30 tests: pipeline + ภาพจริง 3 ชุด + หน้าเว็บจริง
python scripts/evaluate.py                       # เทียบเวอร์ชันแรกกับเวอร์ชันใหม่บนภาพสังเคราะห์
python scripts/evaluate.py --real test_images    # ประเมินภาพถ่ายจริง (test_images/<ชื่อชุด>/*.jpg)
```

ผลประเมินถูกบันทึกใน `evaluation_outputs/` (ไม่ได้เก็บใน git)

##  ชุดภาพทดสอบ (`test_images/`)

ภาพทุกชุดรวบรวมจากอินเทอร์เน็ต โดยแต่ละชุดถ่ายจากจุดเดียวกันด้วยกล้องตัวเดียว (ข้อมูลกล้องอ่านจาก EXIF) เพิ่มชุดใหม่ได้โดยสร้างโฟลเดอร์ที่มีอย่างน้อย 2 ภาพ แล้วเว็บจะแสดงเป็นตัวเลือกให้เอง

| ชุดภาพ | ภาพ | ขนาด | กล้อง | ลักษณะเด่น | แหล่งที่มา |
|---|---:|---|---|---|---|
| building | 5 | 640×480 | iPhone 4S | อาคารกระจก/อิฐ, perspective ชัด | _โปรดระบุ_ |
| cafe | 5 | 1920×1080 | Pantech IM-A890S | ร้านกาแฟ, วัตถุใกล้กล้อง, แสงในร่ม/กลางแจ้งต่างกันมาก | _โปรดระบุ_ |
| city | 8 | 2048×1360 | Nikon D90 | ถนนหมู่บ้าน, หมุนกว้าง ~140° (ใช้ cylindrical) | _โปรดระบุ_ |
| class | 4 | 1600×1200 | Pantech IM-A890S | ห้องเรียน, มีคนอยู่ใกล้กล้อง (parallax) | _โปรดระบุ_ |
| house | 5 | 968×648 | Pentax K200D | บ้านและถนน, ย้อนแสง | _โปรดระบุ_ |

ภาพเหล่านี้ใช้เพื่อการศึกษาในรายวิชาเท่านั้น ลิขสิทธิ์เป็นของเจ้าของภาพ กรุณาใส่ลิงก์แหล่งที่มาในตารางด้านบน

## Deploy บน Streamlit Community Cloud

แอป deploy จาก branch `main` การ push commit ใหม่ขึ้น branch นี้ Streamlit จะ deploy เวอร์ชันล่าสุดให้อัตโนมัติ (Main file path: `app.py`, ไลบรารีติดตั้งจาก `requirements.txt`)
