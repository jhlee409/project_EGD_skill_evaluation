import streamlit as st
import os
import cv2
import numpy as np
import sys
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

# 버전 정보 출력
st.write(f"Python 버전: {sys.version}")
st.write(f"OpenCV 버전: {cv2.__version__}")
st.write(f"NumPy 버전: {np.__version__}")

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

# 고정된 훈련 데이터
FIXED_TRAIN_DATA = np.array([
    [0.2, 0.15],  # 예시 기준값 1
    [0.3, 0.18],  # 예시 기준값 2
    [0.25, 0.16]  # 예시 기준값 3
])

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
    # 프레임 크기 통일
    frame = cv2.resize(frame, (640, 480))
    
    # 노이즈 제거
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
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
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)  # 고정된 프레임 레이트
    
    if not camera.isOpened():
        st.error(f"동영상 파일을 열 수 없습니다: {file_path}")
        return None
    
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = camera.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / frame_rate  # 프레임 간 시간 간격
    duration = length / frame_rate
    
    if duration < MIN_VIDEO_DURATION or duration > MAX_VIDEO_DURATION:
        st.error(f"동영상 길이가 {int(duration // 60)} min {int(duration % 60)} sec로 2분 30초에서 5분 30초 사이의 범위를 벗어납니다.")
        camera.release()
        return None
    
    points_data = analyze_frames(camera, length, frame_time)
    camera.release()
    return points_data, duration

def analyze_frames(camera, length, frame_time):
    """프레임별 분석 함수"""
    np.seterr(all='raise')
    np.set_printoptions(precision=10)
    
    pts = deque(maxlen=32)  # 위치 데이터를 저장할 큐
    angle_g = np.array([])
    distance_g = np.array([])
    prev_center = None
    prev_time = 0
    
    for frame_count in range(length):
        ret, frame = camera.read()
        if not ret:
            break
            
        # HSV 색공간으로 변환하여 녹색 버튼 검출 향상
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 노이즈 제거
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 가장 큰 윤곽선 선택
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > 500:  # 최소 크기 조건
                # 중심점 계산
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    current_center = np.array([center_x, center_y])
                    
                    # 속도 계산
                    current_time = frame_count * frame_time
                    if prev_center is not None:
                        # 픽셀 단위 거리를 실제 거리로 변환 (예: 1픽셀 = 1mm로 가정)
                        distance = np.linalg.norm(current_center - prev_center)
                        time_diff = current_time - prev_time
                        if time_diff > 0:
                            velocity = distance / time_diff
                            # 비정상적으로 큰 속도값 필터링 (노이즈 제거)
                            if velocity < 1000:  # 적절한 임계값 설정
                                distance_g = np.append(distance_g, velocity)
                    
                    prev_center = current_center
                    prev_time = current_time
                    pts.appendleft(current_center)
        
        # 각도 계산
        if len(pts) >= 3:
            for i in range(1, len(pts) - 1):
                if pts[i - 1] is not None and pts[i] is not None and pts[i + 1] is not None:
                    angle = calculate_angle(pts[i-1], pts[i], pts[i+1])
                    if not np.isnan(angle):
                        angle_g = np.append(angle_g, angle)
    
    # 통계 계산
    mean_velocity = np.mean(distance_g) if len(distance_g) > 0 else 0
    std_velocity = np.std(distance_g) if len(distance_g) > 0 else 0
    mean_angle = np.mean(angle_g) if len(angle_g) > 0 else 0
    std_angle = np.std(angle_g) if len(angle_g) > 0 else 0
    
    # 판단 로직
    # 1. 속도 기반 평가
    velocity_score = evaluate_velocity(mean_velocity, std_velocity)
    # 2. 각도 기반 평가
    angle_score = evaluate_angle(mean_angle, std_angle)
    # 3. 종합 평가 (속도:각도 = 65:35로 조정)
    final_score = 0.65 * velocity_score + 0.35 * angle_score
    
    # 판단 결과 생성
    decision_score = final_score
    margin = 0.1  # 여유 범위를 10%로 확대
    
    # 합격 기준을 65%로 조정 (더 현실적인 기준)
    if decision_score > 0.65:
        str3 = 'pass'
        st.write('EGD 수행이 적절하게 진행되어 검사 과정 평가에서는 합격입니다.')
        if decision_score > 0.85:  # 우수 수행 기준 추가
            st.write('특히 우수한 수행을 보여주었습니다!')
    else:
        str3 = 'failure'
        st.write('EGD 수행이 적절하게 진행되지 못했습니다. 검사 과정 평가에서 불합격입니다.')
        if decision_score < 0.4:  # 매우 미흡한 경우 추가 피드백
            st.write('기본적인 조작 기술의 향상이 필요합니다.')
    
    str4 = str(round(decision_score, 4))
    st.write(f"판단 점수: {str4} (합격 기준: 0.65)")
    
    # 세부 평가 결과 표시 (백분율로 변환하여 표시)
    st.write(f"속도 평가 점수: {round(velocity_score * 100, 1)}%")
    st.write(f"각도 평가 점수: {round(angle_score * 100, 1)}%")
    
    # 개선이 필요한 영역 피드백 제공
    if velocity_score < 0.6:
        st.write("※ 내시경 조작 속도 조절이 필요합니다.")
    if angle_score < 0.6:
        st.write("※ 내시경 회전 동작의 안정성 향상이 필요합니다.")
    
    return str3, str4

def evaluate_velocity(mean_velocity, std_velocity):
    """속도 기반 평가 함수"""
    # 속도 범위 설정 (내시경 움직임 특성 반영)
    optimal_mean_velocity = 100  # 적정 평균 속도 (너무 빠르지 않게 조정)
    optimal_std_velocity = 30    # 안정적인 움직임을 위한 표준편차
    
    # 속도 점수 계산 방식 개선
    # 평균 속도가 너무 빠르거나 느린 경우 감점
    if mean_velocity < optimal_mean_velocity:
        mean_score = mean_velocity / optimal_mean_velocity
    else:
        mean_score = max(0, 1.0 - (mean_velocity - optimal_mean_velocity) / (optimal_mean_velocity * 2))
    
    # 표준편차 점수 계산 (안정성 평가)
    # 표준편차가 작을수록 더 안정적인 움직임
    std_score = max(0, 1.0 - (std_velocity / optimal_std_velocity))
    
    # 최종 속도 점수 계산 (평균과 표준편차를 8:2 비율로 조정)
    # 평균 속도의 중요도를 높임
    return 0.8 * mean_score + 0.2 * std_score

def evaluate_angle(mean_angle, std_angle):
    """각도 기반 평가 함수"""
    # 각도 범위 설정 (내시경 회전 특성 반영)
    optimal_mean_angle = 30    # 부드러운 회전을 위한 평균 각도
    optimal_std_angle = 20     # 다양한 각도 변화 허용
    max_allowed_angle = 90     # 최대 허용 각도
    
    # 평균 각도 점수 계산
    # 각도가 너무 크거나 작은 경우 감점
    if mean_angle <= optimal_mean_angle:
        mean_score = mean_angle / optimal_mean_angle
    else:
        mean_score = max(0, 1.0 - (mean_angle - optimal_mean_angle) / (max_allowed_angle - optimal_mean_angle))
    
    # 표준편차 점수 계산
    # 적절한 범위 내의 각도 변화는 허용
    if std_angle <= optimal_std_angle:
        std_score = 1.0
    else:
        std_score = max(0, 1.0 - (std_angle - optimal_std_angle) / optimal_std_angle)
    
    # 최종 각도 점수 계산 (평균과 표준편차를 7:3 비율로 조정)
    return 0.7 * mean_score + 0.3 * std_score

def calculate_angle(p1, p2, p3):
    """세 점 사이의 각도 계산"""
    if p1 is None or p2 is None or p3 is None:
        return np.nan
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 수치 안정성을 위한 클리핑
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)

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

def create_result_image(bmp_files, name_endo, current_date, duration, str3):
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
    
    kst = pytz.timezone('Asia/Seoul')
    current_date = datetime.now(kst).strftime("%Y%m%d")
    
    add_text_to_image(draw, len(bmp_files), duration, str3)
 
    temp_result_path = os.path.join(TEMP_DIR, f'{name_endo}_{current_date}.png') 
    result_image.save(temp_result_path)
    return temp_result_path

def add_text_to_image(draw, photo_count, duration, str3):
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
    
    result_text = str3.rstrip('.') if isinstance(str3, str) else str3
    
    text = (
        f"photo number: {photo_count}\n"
        f"duration: {video_length}\n"
        f"result: {result_text}"
    )
    
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
    st.subheader("- 파일 업로드 및 파악 과정 -")
    
    uploaded_files = st.file_uploader("분석할 파일들을 탐색기에서 찾아 모두 선택해주세요", 
                                      accept_multiple_files=True,
                                      type=['avi', 'bmp', 'mp4'])
    
    if uploaded_files and name_endo:
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        kst = pytz.timezone('Asia/Seoul')
        current_date = datetime.now(kst).strftime("%Y%m%d")
        
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
        
        duration = None
        str3 = None
        str4 = None
        
        for file_path in avi_files:
            result = analyze_video(file_path)
            if result:
                str3, str4 = result
                duration = result[1]
        
        if has_bmp and duration is not None:
            st.divider()
            st.subheader("- 이미지 저장 과정 -")
            
            temp_result_path = create_result_image(bmp_files, name_endo, current_date, duration, str3)
            
            result_blob = bucket.blob(f'EGD_skill_evaluation/test_results/{name_endo}_{current_date}.png')
            result_blob.upload_from_filename(temp_result_path)
            
            st.success(f"이미지가 저장되었습니다: {name_endo}_{current_date}.png")
            st.image(temp_result_path, use_container_width=True)
        
        cleanup_temp_files()
        st.divider()
        st.success("평가가 완료되었습니다.")
    elif uploaded_files and not name_endo:
        st.error("이름이 입력되지 않았습니다.")

if __name__ == "__main__":
    main()