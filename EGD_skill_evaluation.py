import streamlit as st
import os
import cv2
import numpy as np
from collections import deque
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import svm
from math import atan2, degrees
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials, storage

# Constants
TEMP_DIR = "temp_files"
GREEN_LOWER = np.array([35, 80, 50], np.uint8)
GREEN_UPPER = np.array([100, 255, 255], np.uint8)
MIN_VIDEO_DURATION = 150  # 2.5 minutes
MAX_VIDEO_DURATION = 330  # 5.5 minutes
A4_WIDTH = 2480
A4_HEIGHT = 3508
IMAGES_PER_ROW = 8
PADDING = 20
FONT_SIZE = 50

def initialize_firebase():
    """Firebase 초기화 함수"""
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"],
            "universe_domain": st.secrets["universe_domain"]
        })
        firebase_admin.initialize_app(cred, {"storageBucket": "amcgi-bulletin.appspot.com"})
    return storage.bucket('amcgi-bulletin.appspot.com')

def process_video_frame(frame):
    """비디오 프레임 처리 함수"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    contours, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        g = max(contours, key=cv2.contourArea)
        ga = cv2.contourArea(g)
    else:
        g = []
        ga = 0
    
    return g, ga

def analyze_video(file_path):
    """비디오 분석 함수"""
    camera = cv2.VideoCapture(file_path)
    if not camera.isOpened():
        st.error(f"동영상 파일을 열 수 없습니다: {file_path}")
        return None
    
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = camera.get(cv2.CAP_PROP_FPS)
    duration = length / frame_rate
    
    st.write(f'동영상 길이: {int(duration // 60)} 분  {int(duration % 60)} 초')
    
    if duration < MIN_VIDEO_DURATION or duration > MAX_VIDEO_DURATION:
        st.error(f"동영상 길이가 {int(duration // 60)}분 {int(duration % 60)}초로 2분 30초에서 5분 30초 사이의 범위를 벗어납니다.")
        camera.release()
        return None
    
    points_data = analyze_frames(camera, length)
    camera.release()
    return points_data, duration

def analyze_frames(camera, length):
    """프레임별 분석 함수"""
    pts = deque()
    angle_g = []
    distance_g = []
    
    st.write("프레임 분석 시작")
    progress_bar = st.progress(0)
    
    for frame_count in range(length):
        progress = (frame_count + 1) / length
        progress_bar.progress(progress)
        
        ret, frame = camera.read()
        if not ret:
            break
            
        g, ga = process_video_frame(frame)
        
        if ga > 500:
            u = np.array(g)
        else:
            u = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[2, 1]], [[2, 2]], [[1, 2]], [[0, 2]], [[0, 1]]])
        
        M = cv2.moments(u)
        px = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        py = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        ((cx, cy), radius) = cv2.minEnclosingCircle(u)
        
        pts.append([
            frame_count + 1,
            2 if ga > 500 else 3,
            abs(px),
            abs(py),
            int(radius)
        ])
        
        # Calculate angles and distances
        if len(pts) > 1:
            prev_point = pts[-2]
            curr_point = pts[-1]
            
            if (prev_point[1] != 3 and curr_point[1] != 3) and (prev_point[1] == 2 and curr_point[1] == 2):
                a = curr_point[2] - prev_point[2]  # x difference
                b = curr_point[3] - prev_point[3]  # y difference
                angle_g.append(degrees(atan2(a, b)))
                rr = prev_point[4]  # radius
                if rr != 0:
                    delta_g = (np.sqrt((a * a) + (b * b))) / rr
                    distance_g.append(delta_g)
                    if frame_count % 30 == 0:  # 30프레임마다 중간값 출력
                        st.write(f"Frame {frame_count}: delta_g = {delta_g:.4f}, rr = {rr}")
            else:
                distance_g.append(0)
    
    # 디버깅을 위한 중간값 출력
    valid_distances = [d for d in distance_g if d < 6]
    st.write(f"총 거리 측정값 개수: {len(distance_g)}")
    st.write(f"유효한 거리 측정값 개수: {len(valid_distances)}")
    
    mean_g = np.mean([ggg for ggg in distance_g if ggg < 6])
    std_g = np.std([ggg for ggg in distance_g if ggg < 6])
    x_test = np.array([[mean_g, std_g]])

    # 결과의 일관성을 위해 랜덤 시드 설정
    np.random.seed(42)

    # 기준 데이터 로드 (x_train.csv가 없으면 생성)
    if not os.path.exists('x_train.csv'):
        # 초기 기준 데이터 설정
        x_train = np.array([
            [0.3, 0.2],    # 예시 기준값 1
            [0.4, 0.25],   # 예시 기준값 2
            [0.35, 0.22]   # 예시 기준값 3
        ])
        np.savetxt('x_train.csv', x_train, delimiter=',')
    
    # 기존 훈련 데이터 로드
    x_train = np.loadtxt('x_train.csv', delimiter=',')

    # 고정된 정규화 범위 사용
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # SVM 모델 생성 (고정된 파라미터 사용)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(x_train_scaled)

    # 예측
    y_pred_test = clf.predict(x_test_scaled)
    decision_value = clf.decision_function(x_test_scaled)[0]
    st.write(f"결정 값: {decision_value:.4f}")
    
    if y_pred_test == 1:
        str3 = 'pass.'
        st.success('EGD 수행이 적절하게 진행되어 검사 과정 평가에서는 합격입니다.')
    else:
        str3 = 'failure.'
        st.error('EGD 수행이 적절하게 진행되지 못했습니다. 검사 과정 평가에서 불합격입니다.')
    
    return str3, str(round(decision_value, 4))

def process_frame_data(frame_count, contour, area):
    """프레임 데이터 처리 함수"""
    frame_data = [frame_count + 1]
    
    if area > 500:
        u = np.array(contour)
        frame_data.append(2)
    else:
        u = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[2, 1]], [[2, 2]], [[1, 2]], [[0, 2]], [[0, 1]]])
        frame_data.append(3)
    
    M = cv2.moments(u)
    px = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    py = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    
    ((cx, cy), radius) = cv2.minEnclosingCircle(u)
    frame_data.extend([abs(px), abs(py), int(radius)])
    
    return frame_data

def create_result_image(bmp_files, name_endo, current_date, duration, str3, str4):
    """결과 이미지 생성 함수"""
    result_image = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), 'white')
    draw = ImageDraw.Draw(result_image)
    
    single_width = (A4_WIDTH - (PADDING * (IMAGES_PER_ROW + 1))) // IMAGES_PER_ROW
    x, y = PADDING, PADDING
    
    for idx, bmp_file in enumerate(bmp_files):
        img = Image.open(bmp_file)
        img.thumbnail((single_width, single_width))
        result_image.paste(img, (x, y))
        
        x += single_width + PADDING
        if (idx + 1) % IMAGES_PER_ROW == 0:
            x = PADDING
            y += single_width + PADDING
    
    add_text_to_image(draw, len(bmp_files), duration, str3, str4)
    
    temp_result_path = os.path.join(TEMP_DIR, f'{name_endo}_{current_date}.png')
    result_image.save(temp_result_path)
    return temp_result_path

def add_text_to_image(draw, photo_count, duration, str3, str4):
    """이미지에 텍스트 추가 함수"""
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except OSError:
        try:
            font_path = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            font = ImageFont.truetype(font_path, FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()
            st.warning("시스템 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    
    video_length = f"{int(duration // 60)} min {int(duration % 60)} sec"
    text = f"photo number: {photo_count}\nduration: {video_length}\nresult: {str3}\nstr4: {str4}"
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_height = text_bbox[3] - text_bbox[1]
    
    draw.text((PADDING, A4_HEIGHT - text_height - PADDING), text, 
              fill=(0, 0, 0), font=font, align="left")

def cleanup_temp_files():
    """임시 파일 정리 함수"""
    for file_path in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file_path))
    os.rmdir(TEMP_DIR)

def main():
    st.set_page_config(page_title="EGD_skill_evaluation")
    bucket = initialize_firebase()
    
    st.markdown("<h1>EGD_skill_evaluation</h1>", unsafe_allow_html=True)
    st.markdown("이 페이지는 EGD simulator을 대상으로 한 EGD 검사 수행의 적절성을 평가하는 페이지 입니다.")
    st.divider()
    
    name_endo = st.text_input("본인의 성명을 한글로 입력해 주세요 (예: F1홍길동, R3아무개):")
    
    st.divider()
    st.subheader("- 파일 업로드 및 파악 과정-")
    
    uploaded_files = st.file_uploader("분석할 파일들을 탐색기에서 찾아 모두 선택해주세요", 
                                    accept_multiple_files=True,
                                    type=['avi', 'bmp', 'mp4'])
    
    if uploaded_files and name_endo:
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # 파일 분류 및 처리
        avi_files = []
        bmp_files = []
        has_bmp = False
        
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.lower().endswith(('.avi', '.mp4')):
                avi_files.append(temp_path)
            elif uploaded_file.name.lower().endswith('.bmp'):
                has_bmp = True
                bmp_files.append(temp_path)
        
        st.divider()
        st.subheader("- 동영상 분석 과정 -")
        
        # 한국 시간대로 현재 시간 설정
        kst = pytz.timezone('Asia/Seoul')
        current_date = datetime.now(kst).strftime("%Y%m%d")
        
        for file_path in avi_files:
            str3, str4 = analyze_video(file_path)
        
        if has_bmp:
            st.divider()
            st.subheader("- 이미지 저장 과정 -")
            temp_result_path = create_result_image(bmp_files, name_endo, current_date, 
                                                 0, str3, str4)
            
            # Firebase Storage에 업로드 (파일명에 시간 포함)
            result_blob = bucket.blob(f'EGD_skill_evaluation/test_results/{name_endo}_{current_date}.png')
            result_blob.upload_from_filename(temp_result_path)
            
            st.success(f"이미지가 저장되었습니다: {name_endo}_{current_date}.png")
            st.image(temp_result_path, use_container_width=True)
        
        # 임시 파일 정리
        cleanup_temp_files()
        st.divider()
        st.success("평가가 완료되었습니다.")
    elif uploaded_files and not name_endo:
        st.error("이름이 입력되지 않았습니다.")

if __name__ == "__main__":
    main()