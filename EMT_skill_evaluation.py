import streamlit as st
import os
import cv2
import numpy as np
from collections import deque
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import svm
from math import atan2, degrees
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, storage

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

st.set_page_config(page_title="EMT_skill_evaluation")
bucket = initialize_firebase()

st.markdown("<h1>EMT_skill_evaluation</h1>", unsafe_allow_html=True)
st.markdown("이 페이지는 EGD simulator을 대상으로 한 EMT 검사 수행의 적절성을 평가하는 페이지입니다.")
st.markdown("합격 판정이 나오면 추가로 파일을 올리지 마세요. 올릴 때마다 이전기록이 삭제됩니다.")
st.write("---")

user_name = st.text_input("본인의 성명을 한글로 입력해 주세요 (예: 홍길동):")
position = st.selectbox("Select Position", ["", "Staff", "F1", "F2", "R3", "Student"])  

st.write("---")

def is_korean(text):
    # 한글 유니코드 범위: AC00-D7A3 (가-힣)
    return all('\uAC00' <= char <= '\uD7A3' for char in text if char.strip())

# 입력값 검증
is_valid = True
if not user_name:
    st.error("한글 이름을 입력해 주세요 (예; 이진혁)")
    is_valid = False
if position == "Select Position" or not position:
    st.error("position을 선택해 주세요")
    is_valid = False
elif not is_korean(user_name):
    st.error("한글 이름을 입력해 주세요 (예; 이진혁)")
    is_valid = False

st.write("---")

if is_valid:

    st.subheader("합격 동영상 예시")
    st.write("EMT 합격한 동영상 예시를 올립니다. 잘보고 어떤 점에서 초심자와 차이가 나는지 연구해 보세요.")
    try:
        bucket = storage.bucket('amcgi-bulletin.appspot.com')
        demonstration_blob = bucket.blob('EMT_skill_evaluation/EMT_pass_demo/EMT_pass_demo.avi')
        if demonstration_blob.exists():
            demonstration_url = demonstration_blob.generate_signed_url(expiration=timedelta(minutes=15))
            if st.download_button(
                label="동영상 다운로드",
                data=demonstration_blob.download_as_bytes(),
                file_name="EMT_pass_demo.avi",
                mime="video/avi"
            ):
                st.write("")
        else:
            st.error("EMT 합격 동영상 파일을 찾을 수 없습니다.")

    except Exception as e:
        st.error(f"EMT 합격 동영상 파일 다운로드 중 오류가 발생했습니다: {e}")

    st.write("---")

    st.subheader("- 파일 업로드 및 파악 과정 -")

    uploaded_files = st.file_uploader("분석할 파일들(avi, mp4, bmp)을 탐색기에서 찾아 모두 선택해주세요", 
                                    accept_multiple_files=True,
                                    type=['avi', 'bmp', 'mp4'])

    # 파일의 업로드 및 파악
    if uploaded_files:
        st.write(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
        if not user_name:
            st.error("이름이 입력되지 않았습니다.")
        else:
            # 임시 디렉토리 생성
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)

            # 파일 분류
            has_bmp = False
            avi_files = []
            bmp_files = []

            # 업로드된 파일 저장 및 분류
            total_files = len(uploaded_files)
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith('.avi') or uploaded_file.name.endswith('.mp4'):
                    avi_files.append(temp_path)
                elif uploaded_file.name.endswith('.bmp'):
                    has_bmp = True
                    bmp_files.append(temp_path)

            st.write(f"avi 파일 수 : {len([file for file in avi_files if file.endswith('.avi')])} , MP4 파일 수 : {len([file for file in avi_files if file.endswith('.mp4')])} , BMP 파일 수: {len(bmp_files)}")

            # BMP 이미지 수 확인
            if not (62 <= len(bmp_files) <= 66):
                st.error("사진의 숫자가 62장에서 66을 벗어납니다. 다시 시도해 주세요")
                st.stop()

            # AVI 파일 처리
            total_avi_files = len(avi_files)
            for file_path in avi_files:
                camera = cv2.VideoCapture(file_path)
                if not camera.isOpened():
                    st.error("동영상 파일을 열 수 없습니다.")
                    continue

                length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_rate = camera.get(cv2.CAP_PROP_FPS)
                duration = length / frame_rate

                st.write(f"---\n동영상 길이: {int(duration // 60)} 분 {int(duration % 60)} 초")
                if not (300 <= duration <= 330):
                    st.error("동영상의 길이가 5분에서 5분30초를 벗어납니다. 다시 시도해 주세요")
                    st.stop()

                st.write(f"비디오 정보 : 총 프레임 수 = {length} , 프레임 레이트 = {frame_rate:.2f}")
                progress_container = st.empty()
                progress_container.progress(0)


                try:
                    # 프레임 처리를 위한 변수 초기화
                    pts = []
                    angle_g = np.array([])
                    distance_g = np.array([])
                    frame_count = 0

                    # 진행률 표시를 위한 컨테이너 생성
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    while True:
                        ret, frame = camera.read()
                        if not ret:
                            break

                        # 프레임 카운트 증가
                        frame_count += 1

                        try:
                            # 프레임 분석
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            
                            # 색상 범위 설정 및 마스크 생성
                            green_lower = np.array([50, 80, 50], np.uint8)
                            green_upper = np.array([60, 255, 255], np.uint8)
                            green = cv2.inRange(hsv, green_lower, green_upper)

                            # 윤곽선 검출
                            contours, hierarchy = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if len(contours) > 0:
                                # 가장 큰 윤곽선 찾기
                                c = max(contours, key=cv2.contourArea)
                                ga = cv2.contourArea(c)
                                
                                if ga > 500:
                                    u = c
                                    pts.extend([frame_count, 2])
                                else:
                                    u = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[2, 1]], [[2, 2]], [[1, 2]], [[0, 2]], [[0, 1]]])
                                    pts.extend([frame_count, 3])

                                # 중심점 계산
                                M = cv2.moments(u)
                                if M["m00"] != 0:
                                    px = abs(int(M["m10"] / M["m00"]))
                                    py = abs(int(M["m01"] / M["m00"]))
                                else:
                                    px, py = 0, 0

                                pts.extend([px, py])

                                # 최소 외접원 계산
                                ((cx, cy), radius) = cv2.minEnclosingCircle(u)
                                pts.append(int(radius))

                            # 진행률 업데이트 (5프레임마다)
                            if frame_count % 5 == 0:
                                progress = frame_count / length
                                progress_container.progress(progress)

                        except Exception as e:
                            st.write(f"\n[ERROR] 프레임 {frame_count} 처리 중 오류 발생 : {str(e)}")
                            continue

                    # 진행률 표시 컨테이너 제거
                    progress_bar.empty()
                    progress_text.empty()

                    st.write(f"처리된 총 프레임 수 :  {frame_count}")
                    st.write(f"수집된 데이터 포인트 수 : {len(pts)}")
                    st.write("\n-> 분석 완료")

                except Exception as e:
                    st.write(f"\n[ERROR] 비디오 처리 중 치명적 오류 발생 : {str(e)}")
                finally:
                    # 분석 완료 후 정리
                    camera.release()

                k = list(pts)
                array_k = np.array(k)

                frame_no = array_k[0::5]
                timesteps = len(frame_no)
                frame_no2 = np.reshape(frame_no, (timesteps, 1))

                color = array_k[1::5]
                color2 = np.reshape(color, (timesteps, 1))

                x_value = array_k[2::5]
                x_value2 = np.reshape(x_value, (timesteps, 1))

                y_value = array_k[3::5]
                y_value2 = np.reshape(y_value, (timesteps, 1))

                radius2 = array_k[4::5]
                radius3 = np.reshape(radius2, (timesteps, 1))

                points = np.hstack([frame_no2, color2, x_value2, y_value2, radius3])

                for i in range(timesteps - 1):
                    if (points[i][1] != 3 and points[i + 1][1] != 3) and (points[i][1] == 2 and points[i + 1][1] == 2):
                        a = points[i + 1][2] - points[i][2]
                        b = points[i + 1][3] - points[i][3]
                        angle_g = np.append(angle_g, degrees(atan2(a, b)))
                        rr = points[i][4]
                        delta_g = (np.sqrt((a * a) + (b * b))) / rr
                        distance_g = np.append(distance_g, delta_g)
                    else:
                        distance_g = np.append(distance_g, 0)

                mean_g = np.mean([ggg for ggg in distance_g if ggg < 6])
                std_g = np.std([ggg for ggg in distance_g if ggg < 6])
                x_test = np.array([[mean_g, std_g]])

                # 결과의 일관성을 위해 랜덤 시드 설정
                np.random.seed(42)

                # 기준 데이터 로드 (x_train.csv가 없으면 생성)
                if not os.path.exists('x_train.csv'):
                    # 초기 기준 데이터 설정
                    x_train = np.array([
                        [0.2, 0.15],  # 예시 기준값 1
                        [0.3, 0.18],  # 예시 기준값 2
                        [0.25, 0.16]  # 예시 기준값 3
                    ])
                    np.savetxt('x_train.csv', x_train, delimiter=',')
                
                # 기존 훈련 데이터 로드
                x_train = np.loadtxt('x_train.csv', delimiter=',')

                # 고정된 정규화 범위 사용
                scaler = MinMaxScaler(feature_range=(0, 1))
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)

                # SVM 모델 생성
                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf.fit(x_train_scaled)

                st.write("---")
                st.subheader("-최종 판정-")

                y_pred_test = clf.predict(x_test_scaled)
                str4 = str(round(clf.decision_function(x_test_scaled)[0], 4))
                st.write(f"판단 점수: {str4}")
                if y_pred_test == 1:
                    str3 = 'Pass'
                    st.write('EGD 수행이 적절하게 진행되어 EMT 과정에서 합격하셨습니다. 수고하셨습니다.')
                else:
                    str3 = 'Fail.'
                    st.write('EGD 수행이 적절하게 진행되지 못해 불합격입니다. 다시 도전해 주세요.')


            # BMP 파일 처리 (한 번만 실행)
            if has_bmp and duration is not None:
                # A4 크기 설정 (300 DPI 기준)
                a4_width = 2480
                a4_height = 3508
                images_per_row = 8
                padding = 20

                # A4 크기의 빈 이미지 생성
                result_image = Image.new('RGB', (a4_width, a4_height), 'white')
                draw = ImageDraw.Draw(result_image)

                # 각 이미지의 크기 계산
                single_width = (a4_width - (padding * (images_per_row + 1))) // images_per_row

                # 이미지 배치
                x, y = padding, padding
                for idx, bmp_file in enumerate(bmp_files):
                    img = Image.open(bmp_file)
                    img.thumbnail((single_width, single_width))
                    result_image.paste(img, (x, y))
                    
                    x += single_width + padding
                    if (idx + 1) % images_per_row == 0:
                        x = padding
                        y += single_width + padding
                
                # 텍스트 추가
                font_size = 70  # 폰트 크기를 100으로 수정
                text_color = (0, 0, 0)  # 검은색

                try:
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 볼드체 폰트 경로
                    font = ImageFont.truetype(font_path, font_size)
                except OSError:
                    try:
                        font_path = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"  # 볼드체 폰트 경로
                        font = ImageFont.truetype(font_path, font_size)
                    except OSError:
                        # Windows 시스템 폰트 경로 시도
                        try:
                            font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 맑은 고딕 폰트
                            font = ImageFont.truetype(font_path, font_size)
                        except OSError:
                            font = ImageFont.load_default()
                            st.write("시스템 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

                # 동영상 길이 저장
                video_length = f"{int(duration // 60)} min {int(duration % 60)} sec"

                # 추가할 텍스트
                text = f"Photo number: {len(bmp_files)}\nDuration: {video_length}\nResult: {str3}\nSVM_value: {str4}"

                # 텍스트 크기 계산
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # 텍스트 위치 계산 (왼쪽 정렬, 맨 아래 줄)
                x = padding  # 왼쪽 정렬
                y = a4_height - text_height - padding * 2  # 여백을 더 확보

                # 텍스트 그리기
                draw.text((x, y), text, fill=text_color, font=font, align="left")

                st.divider()
                st.subheader("- 이미지 전송 과정 -")
                
                # 임시 디렉토리 생성
                os.makedirs('EMT_skill_evaluation/test_results', exist_ok=True)
                
                # 결과 이미지 저장
                # current_time = datetime.now().strftime('%Y%m%d')
                temp_image_path = f'EMT_skill_evaluation/test_results/{user_name}.png'
                result_image.save(temp_image_path)
                
                try:
                    if str3 == "Pass":
                        # Firebase Storage에 업로드
                        result_blob = bucket.blob(f'EMT_skill_evaluation/test_results/{user_name}.png')
                        result_blob.upload_from_filename(temp_image_path)
                        st.success(f"이미지가 성공적으로 전송되었습니다: {user_name}.png")
                    else:
                        st.warning("평가 결과가 'fail'이므로 업로드하지 않습니다.")
                    
                    st.image(temp_image_path, use_container_width=True)
                except Exception as e:
                    st.error(f"전송 도중 오류 발생: {str(e)}")
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                        
        st.divider() 
        st.success("평가가 완료되었습니다.")