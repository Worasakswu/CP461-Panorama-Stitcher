# Project Report Guide

## 1. Project title

**Automatic Panorama Stitcher: Multi-image Feature Matching, Homography Warping and Seamless Blending**

## 2. Problem statement

กล้องทั่วไปถ่ายภาพได้มุมรับภาพจำกัด ถ้าต้องการภาพทิวทัศน์หรืออาคารแบบมุมกว้าง ผู้ใช้ต้องถ่ายหลายภาพที่ซ้อนทับกันแล้วนำมาต่อ การต่อด้วยมือทำได้ยาก เพราะแต่ละภาพถ่ายจากมุมต่างกัน (ต้องปรับ perspective) แสงไม่เท่ากัน (auto-exposure, vignetting) และผู้ใช้อาจไม่ได้เรียงลำดับภาพ หรือมีภาพที่ไม่เกี่ยวข้องปนมา

โปรเจกต์นี้สร้างเว็บแอปที่รับภาพ 2–10 ภาพลำดับใดก็ได้ หาความสัมพันธ์ระหว่างภาพเองด้วย feature matching + RANSAC Homography จัดวางทุกภาพลงบนระนาบ (หรือทรงกระบอกเมื่อมุมกว้าง) แล้วรวมภาพให้ไร้รอยต่อด้วย exposure compensation และ multi-band blending

## 3. Objectives

1. ตรวจหา keypoints และสร้าง descriptors ด้วย SIFT (และ ORB เพื่อเปรียบเทียบ)
2. จับคู่ descriptor ด้วย KNN + Lowe's ratio test สำหรับ **ทุกคู่ภาพ**
3. ประมาณ Homography ด้วย RANSAC ตัด outliers และปฏิเสธคู่ภาพที่ไม่ได้ซ้อนทับกันจริง
4. หาลำดับและตำแหน่งของทุกภาพโดยอัตโนมัติ (image graph + maximum spanning tree) และตัดภาพที่ไม่เกี่ยวข้องออก
5. Warp ภาพลงระนาบอ้างอิง และเปลี่ยนเป็น cylindrical projection เมื่อมุมกว้างเกินระนาบเดียว
6. ปรับแสงระหว่างภาพ (gain compensation) และรวมภาพด้วย multi-band blending ให้ไม่เห็นรอยต่อ
7. ตัดขอบดำอัตโนมัติ แสดงหลักฐานของแต่ละขั้นตอนบน Streamlit และดาวน์โหลดผลลัพธ์ได้
8. ประเมินผลเชิงปริมาณเทียบกับโค้ดเวอร์ชันแรก ([EVALUATION.md](EVALUATION.md))

## 4. System architecture

```mermaid
flowchart TD
    A[Upload 2–10 images, any order] --> B[Decode + EXIF orientation + resize]
    B --> C[SIFT or ORB keypoints + descriptors]
    C --> D[KNN matching + Lowe ratio test<br/>for every image pair]
    D --> E[RANSAC Homography + verification<br/>inliers, Brown–Lowe test, plausibility]
    E --> F[Image graph: largest connected set<br/>maximum spanning tree by inliers]
    F --> G[Reference = tree center<br/>compose Homographies along the tree]
    G -->|Plausible on one plane| H[Planar warping]
    G -->|Too wide / horizon crossing| I[Estimate focal from H<br/>Cylindrical projection + similarity RANSAC]
    I --> H2[Cylindrical warping]
    H --> J[Gain compensation]
    H2 --> J
    J --> K[Seam: distance-transform Voronoi]
    K --> L[Multi-band blending<br/>Laplacian pyramid]
    L --> M[Auto-crop largest valid rectangle]
    M --> N[Show panorama, matches, layout,<br/>blend comparison, download PNG/JPG]
    F -.->|Unrelated images| X[Excluded + warning]
```

## 5. Core methods

### 5.1 Feature extraction and matching

SIFT (ค่าเริ่มต้น) หา keypoints ที่ทนต่อการเปลี่ยนสเกลและการหมุน พร้อม descriptor 128 มิติ ORB ใช้ FAST + BRIEF แบบ binary (เร็วกว่าแต่ตำแหน่งจุดแม่นน้อยกว่า) สำหรับแต่ละ descriptor ในภาพ $A$ หาเพื่อนบ้านใกล้ที่สุด 2 ตัวในภาพ $B$ ($d_1 \le d_2$) และเก็บคู่นั้นเมื่อ

$$
\frac{d_1}{d_2} < \tau, \qquad \tau = 0.75 \;(\text{ปรับได้})
$$

จากนั้นถ้าหลายจุดในภาพ $A$ ชี้มาที่จุดเดียวกันในภาพ $B$ จะเก็บเฉพาะคู่ที่ descriptor ใกล้ที่สุด ระบบจับคู่ **ทุกคู่ภาพ** จึงไม่ต้องให้ผู้ใช้เรียงลำดับ

### 5.2 Homography and RANSAC

Homography แปลงจุดบนภาพหนึ่งไปยังอีกภาพ เมื่อฉากเป็นระนาบหรือกล้องหมุนรอบจุดเดิม:

$$
s\begin{bmatrix}x' \\ y' \\ 1\end{bmatrix} =
\mathbf{H}\begin{bmatrix}x \\ y \\ 1\end{bmatrix},
\qquad \mathbf{H}\in\mathbb{R}^{3\times3},\; 8 \text{ DOF}
$$

RANSAC สุ่ม 4 คู่จุด แก้ $\mathbf{H}$ ด้วย DLT นับ inliers ที่ reprojection error $\lVert \mathbf{x}' - \pi(\mathbf{H}\mathbf{x}) \rVert <$ threshold (ค่าเริ่มต้น 4 px) ทำซ้ำจนได้ความมั่นใจ $p = 0.995$ ต้องใช้จำนวนรอบประมาณ $N = \log(1-p) / \log(1-w^4)$ เมื่อ $w$ คือสัดส่วน inliers แล้วปรับ $\mathbf{H}$ ด้วย Levenberg–Marquardt บน inliers ทั้งหมด (`cv2.findHomography`)

คู่ภาพจะถูกใช้เมื่อผ่านทุกข้อ:

- inliers ≥ 15 และ inliers > 8 + 0.3 × จำนวน good matches (เกณฑ์ของ Brown & Lowe, 2007) กันคู่ภาพที่ไม่ได้ซ้อนทับกันจริง
- เมทริกซ์มีค่า finite, มุมภาพไม่ถูกฉายเลยเส้นขอบฟ้า ($w > 0$), ไม่กลับด้าน/พับ (สี่เหลี่ยมนูนทิศเดิม), พื้นที่เปลี่ยนไม่เกิน 8 เท่า และความลึกของมุมภาพต่างกันไม่เกิน 6 เท่า

### 5.3 Image graph, reference image and multi-image composition

แต่ละภาพคือ node และคู่ที่ผ่านการตรวจคือ edge ที่มีน้ำหนักเท่ากับจำนวน inliers ระบบทำงานดังนี้:

1. เลือกกลุ่มภาพที่เชื่อมกันได้ใหญ่ที่สุด ภาพนอกกลุ่มถูกตัดออกพร้อมแจ้งเตือน
2. สร้าง maximum spanning tree (Kruskal) ให้ทุกภาพเชื่อมกันผ่านคู่ที่แม่นที่สุด
3. เลือกภาพอ้างอิงเป็น tree center (eccentricity ต่ำสุด) ภาพริมจึงถูกยืดน้อยที่สุด
4. หา transform ของภาพ $k$ ไปภาพอ้างอิงโดยคูณ Homography ตามเส้นทางในต้นไม้ $\mathbf{H}_{k\to r} = \mathbf{H}_{p\to r}\,\mathbf{H}_{k\to p}$
5. เรียงลำดับซ้าย→ขวาจากตำแหน่งบน canvas (ใช้แสดงผล)

### 5.4 Planar vs cylindrical projection

ถ้า transform ที่คูณต่อกันทำให้ภาพใดข้ามเส้นขอบฟ้าหรือยืดผิดปกติ (เช่นกล้องหมุนรวมเกิน ~120°) โหมด Auto จะเปลี่ยนเป็น cylindrical projection:

- ประมาณ focal length $f$ จาก Homography ของกล้องที่หมุนรอบจุดเดิม $\mathbf{H} = \mathbf{K}\mathbf{R}\mathbf{K}^{-1}$ (สูตรของ Szeliski & Shum, 1997 แบบเดียวกับ `cv::detail::focalsFromHomography`) แล้วใช้ค่ามัธยฐานจากทุกคู่
- ฉายภาพลงทรงกระบอก: $\theta = \arctan(x/f)$, $h = y/\sqrt{x^2+f^2}$, $(u, v) = (f\theta + c_x,\; f h + c_y)$
- แปลงพิกัด keypoints ตามสูตรเดียวกัน (ไม่ต้องหา feature ใหม่) แล้วใช้ RANSAC หา similarity transform (4 DOF) บนพิกัดทรงกระบอก

### 5.5 Warping

หาขอบเขต canvas จากมุมของทุกภาพหลังแปลง เลื่อนให้พิกัดเป็นบวก แล้ว `warpPerspective` แต่ละภาพลงเฉพาะกรอบ (ROI) ที่ภาพครอบคลุมเพื่อประหยัดหน่วยความจำ ถ้า canvas เกิน 12 MP จะย่อผลลัพธ์ลงแทนการ error

### 5.6 Exposure (gain) compensation

หา gain $g_i$ ต่อช่องสีของแต่ละภาพจาก least squares (Brown & Lowe, 2007):

$$
e = \sum_{i}\sum_{j \ne i} N_{ij}\left[\frac{(g_i \bar I_{ij} - g_j \bar I_{ji})^2}{\sigma_N^2} + \frac{(1-g_i)^2}{\sigma_g^2}\right]
$$

$N_{ij}$ คือจำนวนพิกเซลที่ซ้อนกัน และ $\bar I_{ij}$ คือสีเฉลี่ยของภาพ $i$ ในส่วนที่ซ้อนกับภาพ $j$ ใช้ $\sigma_N = 10$ และ $\sigma_g = 0.3$ (หลวมกว่า 0.1 ในบทความ เพราะมือถือปรับแสงต่างกันได้เกิน 10%) แล้ว normalize ให้ gain เฉลี่ยเป็น 1

### 5.7 Seam and multi-band blending

- **Seam:** แต่ละพิกเซลเป็นของภาพที่พิกเซลนั้นอยู่ลึกจากขอบภาพมากที่สุด (distance transform) seam จึงอยู่กลางส่วนซ้อนทับ
- **Multi-band blending (Burt & Adelson, 1983):** สร้าง Laplacian pyramid ของแต่ละภาพ $L^i_k = G^i_k - \operatorname{expand}(G^i_{k+1})$ และ Gaussian pyramid ของ seam mask $W^i_k$ แล้วรวมทีละชั้น

$$
L_k = \frac{\sum_i W^i_k L^i_k}{\sum_i W^i_k}, \qquad
\text{panorama} = L_0 + \operatorname{expand}(L_1 + \operatorname{expand}(L_2 + \dots))
$$

  รายละเอียดความถี่สูงถูกผสมในช่วงแคบ (ภาพคม ไม่มีเงาซ้อน) ส่วนความถี่ต่ำ (แสง/สี) ถูกผสมในช่วงกว้าง รอยต่อจึงกลืนกัน ก่อนสร้าง pyramid ระบบเติมสีจากขอบภาพออกไปในบริเวณที่ไม่มีข้อมูล (push–pull) และขยาย seam label ไปถึงขอบ canvas เพื่อไม่ให้ขอบพาโนรามามืด
- **Feather** (distance-weighted average) และ **วางทับ** (แบบเวอร์ชันแรก) ยังเลือกได้ เพื่อใช้เปรียบเทียบ

### 5.8 Auto-crop

หาสี่เหลี่ยมที่ใหญ่ที่สุดที่อยู่ในบริเวณที่มีข้อมูลทั้งหมด (maximal rectangle ด้วย histogram + stack บน mask ที่ย่อขนาด) แล้วตรวจซ้ำที่ความละเอียดเต็มว่าไม่มีพิกเซลดำเหลืออยู่

## 6. Evaluation

ผลเต็มอยู่ที่ [EVALUATION.md](EVALUATION.md) (ภาพสังเคราะห์ที่รู้ ground truth 70 ชุด + กล้องหมุนมุมกว้าง 5 ชุด) รันซ้ำได้ด้วย `python scripts/evaluate.py`

สำหรับภาพถ่ายจริงของกลุ่ม ให้เก็บภาพไว้ใน `test_images/<ชื่อชุด>/` แล้วรัน `python scripts/evaluate.py --real test_images` ควรถ่ายให้ครอบคลุมปัจจัยต่อไปนี้:

| Factor | Suggested values |
|---|---|
| จำนวนภาพ | 2, 3–5, 6–10 |
| ส่วนซ้อนทับ | ~50%, ~30%, ~15% |
| มุมหมุนรวม | < 90°, 90–150°, > 150° (ควรเปลี่ยนเป็น cylindrical) |
| ฉาก | อาคาร/ป้าย (feature เยอะ), ต้นไม้, ท้องฟ้า/ผนังเรียบ (feature น้อย) |
| แสง | กลางวัน, ในอาคาร, ย้อนแสง, auto-exposure ต่างกันมาก |
| สิ่งรบกวน | คน/รถเคลื่อนที่, ขยับตัวระหว่างถ่าย (parallax), ภาพที่ไม่เกี่ยวข้องปนมา |

บันทึกค่า: จำนวน keypoints, good matches, RANSAC inliers/inlier ratio, reprojection RMSE, ภาพที่ถูกตัดออก, projection ที่ใช้, focal ที่ประมาณได้, เวลาแต่ละขั้นตอน และตรวจด้วยตาว่ามีรอยต่อ/เงาซ้อน/เส้นตรงหักหรือไม่ ภาพจริงไม่มี ground truth จึงรายงานเป็นตัวเลขความสอดคล้อง (inliers, RMSE) และผลเชิงคุณภาพ แยกจากผลบนภาพสังเคราะห์

### 6.1 ชุดภาพ edge case ที่ต้องถ่ายเอง (ใช้ใน live demo)

Rubric ให้ 1.0 คะแนนกับ “live edge-case demonstration with custom test images” จึงต้องใช้ **ภาพที่กลุ่มถ่ายเอง** ไม่ใช่ภาพตัวอย่างสังเคราะห์บนเว็บ ถ่ายด้วยมือถือได้เลย ตั้งชื่อโฟลเดอร์ตามรหัสด้านล่างใน `test_images/` แล้ว **ลองกับเว็บจริงให้ครบทุกชุดก่อนอัดวิดีโอ**

“ผลที่ควรเห็น” คือพฤติกรรมที่ระบบออกแบบไว้ ถ้าภาพจริงให้ผลต่างออกไป ให้บันทึกไว้เป็นข้อจำกัดและเล่าในวิดีโอตามจริง การอธิบายว่าระบบล้มเหลวเพราะอะไรก็นับเป็น edge case ที่ดี

| รหัส | วิธีถ่าย | ผลที่ควรเห็นบนเว็บ | ชี้ให้ดูตรงไหน |
|---|---|---|---|
| E1 ปกติ | ตึก/ลานที่มีรายละเอียด 3–5 ภาพ ยืนที่เดิมแล้วหมุนกล้อง ซ้อนกัน 30–50% | ต่อครบ, Planar, ไม่เห็นรอยต่อ | ผลลัพธ์ + แท็บการวางภาพ |
| E2 สลับลำดับ | ใช้ชุด E1 แต่เลือกไฟล์สลับลำดับตอนอัปโหลด | ข้อความ “ลำดับจากซ้ายไปขวา” เรียงถูก | กล่องข้อความสีเขียว + แผนผังเลขภาพ |
| E3 ภาพแปลกปลอม | ชุด E1 + ภาพจากที่อื่น 1 ภาพ | ต่อเฉพาะภาพที่เกี่ยวข้อง + คำเตือนว่าภาพนั้นถูกตัดออก | คำเตือนสีเหลือง + ตารางคู่ภาพ (inliers ต่ำ, “ไม่ผ่าน”) |
| E4 ซ้อนกันน้อย | 2 ภาพที่ซ้อนกันแค่ ~10–15% | ต่อได้ถ้า inliers พอ หรือขึ้น error พร้อมเหตุผล ไม่สร้างภาพมั่ว | ตารางคู่ภาพ / ข้อความ error |
| E5 ไม่มี feature | ผนังเรียบหรือท้องฟ้าล้วน 2 ภาพ | ไม่ crash, ขึ้น error ว่าไม่พบส่วนซ้อนทับพอ | ข้อความ error + keypoints/matches ต่ำ |
| E6 แสงต่างกัน | ภาพหนึ่งหันเข้าหาหน้าต่าง/ท้องฟ้า อีกภาพหันเข้าที่ร่ม (ให้ auto-exposure ต่างกัน) | Gain ของแต่ละภาพไม่เท่ากัน, รอยต่อกลืน | ปิด/เปิด Exposure compensation ในตั้งค่าขั้นสูง + แท็บเปรียบเทียบ Blending + ตาราง gain |
| E7 มุมกว้าง | 6–8 ภาพ หมุนรวม ~150–180° | คำเตือน “ใช้ Cylindrical projection อัตโนมัติ”, ต่อครบ | ตั้งค่า Projection = Planar แล้วกดใหม่ → error ที่อธิบายเหตุผล |
| E8 Parallax / คนเดิน | เดินขยับระหว่างถ่าย หรือมีคนเดินผ่านส่วนซ้อนทับ | อาจเห็นเงาซ้อน/วัตถุถูกตัดที่ seam | ใช้อธิบายข้อจำกัดของ homography (ข้อ 9) |
| E9 ภาพแนวตั้ง | ถือมือถือแนวตั้ง 3 ภาพ | ภาพหมุนถูกทิศ (EXIF) และต่อได้ | ภาพ preview ตอนอัปโหลด |

ใช้ชุดเหล่านี้กับการตั้งค่าขั้นสูงด้วย เช่น สลับ SIFT/ORB บน E1 หรือปรับ Lowe's ratio 0.6 กับ 0.9 แล้วดูจำนวน good matches และเส้นสีแดงเปลี่ยน จากนั้นรัน `python scripts/evaluate.py --real test_images` เพื่อเก็บตัวเลขของภาพจริงไว้ใส่รายงาน

## 7. Failure handling

- ภาพที่อ่านไม่ได้ → แจ้งชื่อไฟล์และข้ามไป ภาพที่อ่านได้ยังใช้ต่อ
- ภาพไม่มี feature (เช่นผนังเรียบ) → ไม่ crash ได้ 0 matches และแจ้งผลในตารางคู่ภาพ
- คู่ภาพที่ไม่ซ้อนทับ → RANSAC verification ปฏิเสธ พร้อมเหตุผลในตาราง
- ภาพที่ไม่เกี่ยวข้อง → ถูกตัดออกจากโครงข่ายพร้อมคำเตือน ส่วนภาพอื่นยังต่อได้
- ไม่มีคู่ภาพใดใช้ได้ → แสดงข้อความ error ที่บอกคู่ที่ใกล้เคียงที่สุดและคำแนะนำการถ่ายภาพ พร้อมตารางผลการจับคู่
- มุมกว้างเกินระนาบเดียว → Auto เปลี่ยนเป็น cylindrical แต่ถ้าบังคับ Planar จะแจ้ง error แทนการสร้างภาพที่บิด
- focal ประมาณไม่ได้ → ใช้ค่าเริ่มต้นและแจ้งเตือน หรือให้ผู้ใช้กรอกเอง
- พาโนรามาใหญ่เกินหน่วยความจำ → ย่อผลลัพธ์ลงพร้อมแจ้งเปอร์เซ็นต์
- Auto-crop ตัดพื้นที่ทิ้งเกินครึ่ง → แจ้งให้ปิด auto-crop เพื่อดูภาพเต็ม

## 8. Responsibility split (5 members)

งานกำหนดให้กลุ่มละ **5 คนพอดี** และจะหักคะแนนการนำเสนอถ้าสมาชิกคนใดไม่มีส่วนร่วม จึงควรให้ทุกคนได้พูดในวิดีโอใกล้เคียงกัน (คนละประมาณ 1:50 นาที) และรับผิดชอบส่วนของโค้ดที่ตัวเองอธิบาย

| บทบาท | งานหลัก (ไฟล์) | ส่วนที่นำเสนอ | ช่วงเวลาในวิดีโอ | ผู้รับผิดชอบ |
|---|---|---|---|---|
| 1. Requirements + ชุดภาพทดสอบ | ถ่ายภาพ edge case ตามข้อ 6.1, `test_images/` | Problem, objectives, pipeline, E1 | 0:00–1:50 | |
| 2. Features + matching | `src/feature.py`, `src/visualize.py` | SIFT/ORB, ratio test, ภาพคู่จุด | 1:50–3:40 | |
| 3. Geometry | `src/homography.py`, `src/stitcher.py` | RANSAC, verification, image graph, E2–E5 | 3:40–5:30 | |
| 4. Warping + blending | `src/warping.py`, `src/blending.py` | Cylindrical, gain compensation, multi-band, E6–E7 | 5:30–7:20 | |
| 5. Web + evaluation | `app.py`, `scripts/evaluate.py`, `tests/` | Engineering, deploy, ผลประเมิน, ข้อจำกัด (E8), สรุป | 7:20–9:30 | |

## 9. Limitations and future work

- ผลเชิงปริมาณทั้งหมดตอนนี้มาจากภาพสังเคราะห์ ควรเพิ่มชุดภาพถ่ายจริงของกลุ่มพร้อมรายงานผล
- ฉากที่มี parallax (วัตถุใกล้กล้อง, เดินขยับระหว่างถ่าย) ทำให้ homography ไม่พอดีทั้งภาพ แก้ได้ด้วย seam finding แบบ graph-cut หรือ local warping (APAP)
- วัตถุเคลื่อนที่อาจเกิดเงาซ้อน (ghosting) → ใช้ seam ที่หลบบริเวณที่ภาพต่างกันมาก
- ยังไม่มี bundle adjustment: ทดลอง refinement ของ homography 8 DOF ทุกภาพพร้อมกันแล้วไม่ช่วย (ดู EVALUATION.md) ขั้นต่อไปคือ bundle adjustment ของการหมุนกล้อง + focal (≈4 DOF ต่อภาพ) แบบ Brown & Lowe
- Cylindrical projection รองรับการหมุนแนวนอนเป็นหลัก ยังไม่รองรับพาโนรามา 360° ที่ต่อปิดวง หรือหลายแถว (spherical)
- ยังไม่มี wave correction และการแก้ lens distortion
- รองรับสูงสุด 10 ภาพต่อครั้ง และย่อภาพเหลือด้านยาว 1,200 px โดยค่าเริ่มต้นเพื่อให้ทำงานบน Streamlit Community Cloud ได้

## 10. Rubric mapping and demo video

### 10.1 เกณฑ์ให้คะแนน (10 คะแนน) กับหลักฐานในโปรเจกต์

| เกณฑ์ (คะแนน) | สิ่งที่อาจารย์ดู | หลักฐานในโปรเจกต์ | แสดงตอนไหน |
|---|---|---|---|
| Algorithmic Correctness & Robustness (4) | SIFT/ORB, descriptor ratio test | Keypoints ต่อภาพ, ภาพ keypoints (สเกล + orientation), ตาราง good matches, ปรับ ratio ได้ | ผู้พูด 2 |
| | RANSAC | เส้น inlier เขียว / outlier แดง, inlier ratio, RMSE, เมทริกซ์ Homography | ผู้พูด 3 |
| | Failure cases / outliers | Verification ปฏิเสธคู่ที่ไม่ซ้อนทับ, ตัดภาพแปลกปลอม, error ที่อธิบายเหตุผล, Planar → Cylindrical | ผู้พูด 3–4 (E3–E5, E7) |
| | Seamless output | Gain compensation + seam กลางส่วนซ้อนทับ + multi-band + auto-crop, แท็บเปรียบเทียบ Blending | ผู้พูด 4 (E6) |
| Engineering & UI Implementation (3) | โครงสร้างโค้ด, abstraction | แยกโมดูลใน `src/` ตามขั้นตอน, `StitchSettings` / `StitchResult`, ชุดทดสอบ 24 ข้อ, สคริปต์ประเมินผล | ผู้พูด 5 |
| | User experience + tier | Tier 3: Streamlit Cloud URL สาธารณะ, อัปโหลดหลายไฟล์, progress bar, คำเตือนภาษาไทย, ดาวน์โหลด PNG/JPG | ผู้พูด 1 และ 5 |
| 10-Minute Presentation & Demo (3) | อธิบาย pipeline ชัดเจน (1.5) | แผนภาพข้อ 4 + อธิบายทีละขั้นตามลำดับ pipeline | ผู้พูด 1–5 |
| | Live edge-case demo ด้วยภาพของกลุ่มเอง (1.0) | ชุดภาพ E1–E9 ในข้อ 6.1 | ผู้พูด 1, 3, 4, 5 |
| | ไม่เกิน 10 นาที + ทุกคนมีส่วนร่วม (0.5) | ตารางเวลาในข้อ 10.2 รวม 9:30 (เผื่อ 30 วินาที) | ทั้งกลุ่ม |

### 10.2 แผนวิดีโอ (ต้องไม่เกิน 10:00 เด็ดขาด)

| เวลา | ผู้พูด | เนื้อหา | สิ่งที่แสดงบนจอ |
|---|---|---|---|
| 0:00–1:50 | 1 | ปัญหา, วัตถุประสงค์, ภาพรวม pipeline, บอกว่าเป็น Tier 3 | แผนภาพข้อ 4 → เปิดลิงก์เว็บจริง → อัปโหลด E1 → ผลลัพธ์ |
| 1:50–3:40 | 2 | SIFT vs ORB, descriptor, KNN + ratio test | Keypoints ของภาพ E1, ตาราง good matches, ปรับ ratio 0.6/0.9, สลับ ORB |
| 3:40–5:30 | 3 | RANSAC, verification, image graph, ภาพอ้างอิง | เส้นเขียว/แดง, เมทริกซ์ H → E2 (สลับลำดับ) → E3 (ภาพแปลกปลอม) → E5 (error) |
| 5:30–7:20 | 4 | Planar vs cylindrical, gain compensation, seam, multi-band | แท็บการวางภาพ → E7 (มุมกว้าง) → E6 + แท็บเปรียบเทียบ Blending |
| 7:20–9:10 | 5 | โครงสร้างโค้ด, tests, deploy, ผลประเมิน, ข้อจำกัด | GitHub repo, ผล pytest, ตารางใน EVALUATION.md, E8 (parallax) |
| 9:10–9:30 | 5 (หรือทุกคน) | สรุป | สไลด์สรุป + รายชื่อสมาชิก |

เคล็ดลับ: อัดแยกทีละช่วงแล้วตัดต่อ จะคุมเวลาได้ง่ายกว่าอัดรวดเดียว เว็บบน Streamlit Community Cloud จะหลับเมื่อไม่มีคนใช้สักพัก ให้เปิดปลุกไว้ก่อนอัด และกดสร้างพาโนรามาทุกชุดไว้หนึ่งรอบ เพื่อให้รู้ว่าแต่ละชุดใช้เวลาเท่าไร

### 10.3 ประเด็นที่ควรพูดในเสียงบรรยาย (technical decisions)

| การตัดสินใจ | เหตุผล | หลักฐาน |
|---|---|---|
| SIFT เป็นค่าเริ่มต้น, ORB เป็นตัวเลือก | SIFT ทนต่อสเกล/การหมุน และระบุตำแหน่งจุดแบบ sub-pixel | EVALUATION.md (ภาพสังเคราะห์ 3–5 ภาพ): SIFT สำเร็จ 20/20, corner error 0.54 px; ORB 4/20, 6.37 px |
| Lowe's ratio 0.75 + ตัดคู่ที่ชี้จุดซ้ำ | ค่ามาตรฐานจาก Lowe (2004) ตัดคู่กำกวมในลวดลายซ้ำ | ปรับ ratio บนเว็บแล้วดูเส้นแดงเปลี่ยน |
| RANSAC threshold 4 px, confidence 0.995 + verification | ต้องทนต่อ outliers และปฏิเสธคู่ภาพที่ไม่ได้ซ้อนทับจริง (inliers > 8 + 0.3·matches ตาม Brown & Lowe) | E3, E4, E5 |
| จับคู่ทุกคู่ภาพ + maximum spanning tree + ภาพอ้างอิงกลางโครงข่าย | ผู้ใช้ไม่ต้องเรียงลำดับ ใช้คู่ที่แม่นที่สุด และคูณเมทริกซ์ต่อกันน้อยที่สุด ภาพริมจึงยืดน้อย | E2, E3 |
| Auto Planar → Cylindrical + ประมาณ focal จาก H | ระนาบเดียวรองรับมุมกว้างมากไม่ได้ (ภาพเลยเส้นขอบฟ้า) | E7; focal error 1.0% บนภาพสังเคราะห์ |
| Gain compensation (σg = 0.3) | มือถือปรับแสงแต่ละภาพต่างกัน | E6, ตาราง gain ในแท็บข้อมูลทางเทคนิค |
| Seam กลางส่วนซ้อนทับ + multi-band แทน feather/วางทับ | วางทับเห็นขอบแข็ง, feather เบลอ/เงาซ้อนเมื่อ align ไม่สมบูรณ์, multi-band ผสมความถี่ต่ำกว้างแต่เก็บรายละเอียดคม | แท็บเปรียบเทียบ Blending; seam step ลด 41% เมื่อแสงต่างกัน |
| ย่อภาพเหลือ 1,200 px + จำกัด canvas 12 MP | ให้ทำงานได้ใน RAM จำกัดของ Streamlit Community Cloud | เวลาแต่ละขั้นในแท็บข้อมูลทางเทคนิค |
| ทดลอง global refinement แล้วไม่ใช้ | ทดลองแล้วไม่ช่วยให้แม่นขึ้น — แสดงว่าตัดสินใจจากข้อมูล | EVALUATION.md หัวข้อ “สิ่งที่ทดลองแล้วไม่ได้นำมาใช้” |

### 10.4 Checklist ก่อนส่ง

- [ ] ลิงก์ GitHub repo (public, มี `requirements.txt` และวิธีรันใน README)
- [ ] ลิงก์เว็บจริง: ลองเปิดในหน้าต่าง incognito ว่าเข้าได้โดยไม่ต้อง login
- [ ] ถ่ายชุดภาพ E1–E9 ใส่ `test_images/` และลองกับเว็บจริงครบทุกชุด
- [ ] เติมชื่อผู้รับผิดชอบในตารางข้อ 8 ให้ครบ 5 คน
- [ ] วิดีโอยาวไม่เกิน 10:00 (เช็กความยาวไฟล์จริงหลังตัดต่อ) มีเสียงบรรยายและ live demo
- [ ] อัปโหลดวิดีโอเป็น YouTube (Unlisted) หรือ Google Drive (ตั้งค่า “ทุกคนที่มีลิงก์ดูได้”)
- [ ] ทุกคนได้พูดในวิดีโอ และมีรายชื่อสมาชิกในสไลด์แรก/สุดท้าย

## 11. References

- M. Brown and D. G. Lowe, “Automatic Panoramic Image Stitching using Invariant Features,” *IJCV*, 74(1), 2007.
- D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” *IJCV*, 60(2), 2004.
- E. Rublee et al., “ORB: An efficient alternative to SIFT or SURF,” *ICCV*, 2011.
- M. A. Fischler and R. C. Bolles, “Random Sample Consensus,” *Communications of the ACM*, 24(6), 1981.
- P. J. Burt and E. H. Adelson, “A Multiresolution Spline with Application to Image Mosaics,” *ACM Transactions on Graphics*, 2(4), 1983.
- R. Szeliski and H.-Y. Shum, “Creating Full View Panoramic Image Mosaics and Environment Maps,” *SIGGRAPH*, 1997.
- R. Szeliski, “Image Alignment and Stitching: A Tutorial,” *Foundations and Trends in Computer Graphics and Vision*, 2(1), 2006.
- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004.
