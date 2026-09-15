import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import โมดูลจากโฟลเดอร์ src/
from src.feature import extract_and_match
from src.homography import compute_homography
from src.blending import blend_images

st.set_page_config(page_title="Automatic Panorama Stitcher", layout="wide")

st.title("📷 Automatic Panorama Stitcher")
st.write("CP461 Introduction to Computer Vision Project")

# 1. ส่วนอัปโหลดรูปภาพ
st.sidebar.header("Upload Images")
uploaded_file1 = st.sidebar.file_uploader("Upload Image 1 (Left)", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.sidebar.file_uploader("Upload Image 2 (Right)", type=["jpg", "png", "jpeg"])

if uploaded_file1 and uploaded_file2:
    # แปลงไฟล์เป็นรูปแบบ OpenCV Image
    file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
    
    file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

    # แสดงภาพต้นฉบับ 2 ภาพข้างกัน
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Image 1 (Left)", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Image 2 (Right)", use_column_width=True)

    # ปุ่มประมวลผล
    if st.button("Generate Panorama"):
        with st.spinner("Processing Panorama Stitching..."):
            # เรียกใช้ Pipeline (เมื่อ Member 1-3 เขียนเสร็จแล้ว)
            kp1, kp2, matches, vis_matches = extract_and_match(img1, img2)
            H, mask = compute_homography(kp1, kp2, matches)
            result = blend_images(img1, img2, H)

            # แสดงภาพจับคู่จุด Inliers และภาพผลลัพธ์ Panorama
            st.subheader("Feature Matching (Inliers)")
            st.image(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.subheader("Final Panorama Result")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
