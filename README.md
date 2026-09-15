#  ระบบต่อภาพพาโนรามาอัตโนมัติ (Automatic Panorama Stitcher)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cp461-panorama-stitcher-h2tewyfx9y38z45nk7bckd.streamlit.app/)

เว็บแอปพลิเคชัน Computer Vision สำหรับต่อภาพถ่ายที่มีส่วนซ้อนทับกันให้กลายเป็นภาพพาโนรามามุมกว้างไร้รอยต่อโดยอัตโนมัติ พัฒนาขึ้นสำหรับรายวิชา **CP461 Introduction to Computer Vision**

🔗 **ลิงก์เข้าใช้งานระบบ:** [cp461-panorama-stitcher.streamlit.app](https://cp461-panorama-stitcher-h2tewyfx9y38z45nk7bckd.streamlit.app/)

---

##  ฟีเจอร์หลักของระบบ

* **Feature Detection & Matching:** สกัดจุดเด่นของภาพด้วย SIFT (Scale-Invariant Feature Transform) และคัดกรองจุดจับคู่ด้วย Lowe's Ratio Test
* **Homography Estimation:** คำนวณเมทริกซ์การแปลงภาพ (Planar Homography) ด้วยเทคนิค RANSAC เพื่อตัดจุดจับคู่ที่ผิดพลาด (Outliers) ออก
* **Image Blending:** ปรับมุมมองภาพ (Perspective Warping) และรวมภาพเข้าด้วยกันอย่างราบรื่น
* **Interactive Web Interface:** หน้าเว็บใช้งานง่ายด้วย Streamlit พร้อมแสดงภาพขั้นตอนการจับคู่จุด (Inliers) ให้เห็นแบบเรียลไทม์

---

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

* **ภาษาหลัก:** Python 3.10+
* **Computer Vision & Math:** OpenCV (`opencv-python-headless`), NumPy
* **Frontend & Cloud Deployment:** Streamlit, Streamlit Community Cloud

---

## โครงสร้างโฟลเดอร์ใน Repository
CP461-Panorama-Stitcher/
├── app.py              # ไฟล์หลักสำหรับรันหน้าเว็บ Streamlit UI และรวม Pipeline
├── requirements.txt    # รายการไลบรารีที่ระบบต้องใช้
├── README.md           # เอกสารอธิบายโปรเจกต์
└── src/                # โมดูลประมวลผล Computer Vision
├── feature.py      # ฟังก์ชันสกัดจุดเด่นและจับคู่ภาพ 
├── homography.py   # ฟังก์ชันคำนวณ Homography และ RANSAC 
└── blending.py     # ฟังก์ชันปรับมุมและรวมภาพ 

##  ขั้นตอนการติดตั้งและรันบนเครื่องตัวเอง (Local)

```bash
# 1. Clone Repository ลงเครื่อง
git clone [https://github.com/Worasakswu/CP461-Panorama-Stitcher.git](https://github.com/Worasakswu/CP461-Panorama-Stitcher.git)
cd CP461-Panorama-Stitcher

# 2. ติดตั้ง ไลบรารี ที่จำเป็น
pip install -r requirements.txt

# 3. สั่งรัน Web App
streamlit run app.py
