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
import firebase_admin
from firebase_admin import credentials, storage

st.set_page_config(page_title="EGD_skill_evaluation")

# Firebase 초기화
if not firebase_admin._apps:
    # Streamlit Secrets에서 Firebase 설정 정보 로드
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

# Firebase Storage 버킷 참조
bucket_name = 'amcgi-bulletin.appspot.com'
bucket = storage.bucket(bucket_name)  # 항상 사용할 수 있도록 초기화

st.header("EGD_skill_evaluation")
st.markdown("이 페이지는 EGD simulator을 대상으로 한 EGD 검사 수행의 적절성을 평가하는 페이지 입니다.")
st.divider()

name_endo = st.text_input("본인의 성명을 한글로 입력해 주세요:")

# 파일 업로더
uploaded_files = st.file_uploader("분석할 파일들을 선택해주세요", 
                                    accept_multiple_files=True,
                                    type=['avi', 'bmp', 'mp4'])

# 파일의 업로드 및 파악
if uploaded_files:
    if not name_endo:
        st.error("이름을 입력해 주세요.")
    else:
        st.write("파일 업로드 및 파악 중...")  # 파일 업로드 중 메시지
        progress_text = st.empty()  # 진행률 텍스트 초기화

        # 임시 디렉토리 생성
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        # 파일 분류
        has_bmp = False
        avi_files = []
        bmp_files = []

        # 업로드된 파일 저장 및 분류
        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.avi') or uploaded_file.name.endswith('.mp4'):
                avi_files.append(temp_path)
            elif uploaded_file.name.endswith('.bmp'):
                has_bmp = True
                bmp_files.append(temp_path)

            # 진행률 계산 및 표시
            progress = int(((idx + 1) / total_files) * 100)
            progress_text.text(f"파일 업로드 진행률: {progress}%")

        st.success("파일 업로드 및 파악이 완료되었습니다. 지금부터는 동영상 파일을 분석하겠습니다.")

        # AVI 파일 처리
        total_avi_files = len(avi_files)
        processed_files = 0

        for file_path in avi_files:
            camera = cv2.VideoCapture(file_path)
            if not camera.isOpened():
                st.error(f"동영상 파일을 열 수 없습니다: {file_path}")
                continue

            length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = camera.get(cv2.CAP_PROP_FPS)
            duration = length / frame_rate

            st.write(f'동영상 길이: {int(duration // 60)}분 {int(duration % 60)}초')

            # 동영상 길이 체크
            if duration < 180 or duration > 300:  # 3분(180초)에서 5분(300초) 사이 체크
                st.error(f"동영상 길이가 {int(duration // 60)}분 {int(duration % 60)}초로 3분에서 5분 사이의 범위를 벗어납니다. 더이상 분석은 진행되지 않습니다.")
                break  # 분석 중단

            # 진행률 계산 및 표시
            for frame_count in range(length):
                progress = int(((frame_count + 1) / length) * 100)
                progress_text.text(f"동영상 분석 진행률: {progress}%")

                ret, frame = camera.read()
                if not ret:
                    break

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                green_lower = np.array([35, 80, 50], np.uint8)
                green_upper = np.array([100, 255, 255], np.uint8)
                green = cv2.inRange(hsv, green_lower, green_upper)

                contours3, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours3:
                    g = max(contours3, key=cv2.contourArea)
                    ga = cv2.contourArea(g)
                else:
                    g = []
                    ga = 0

                pts = deque()
                ii = 1
                angle_g = []
                distance_g = []

                while ret:
                    pts.append(ii)
                    ii += 1

                    if ga > 500:
                        u = np.array(g)
                        pts.append(2)
                    else:
                        u = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[2, 1]], [[2, 2]], [[1, 2]], [[0, 2]], [[0, 1]]])
                        pts.append(3)

                    M = cv2.moments(u)
                    if M["m00"] != 0:
                        px = abs(int(M["m10"] / M["m00"]))
                        py = abs(int(M["m01"] / M["m00"]))
                    else:
                        px, py = 0, 0

                    pts.append(px)
                    pts.append(py)

                    ((cx, cy), radius) = cv2.minEnclosingCircle(u)
                    center = (int(cx), int(cy))
                    radius = int(radius)
                    pts.append(radius)

                    if radius > 8:
                        cv2.circle(frame, center, 30, (0, 0, 255), -1)

                    ret, frame = camera.read()

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
                x_train = np.array([[mean_g, std_g]])

                # CSV 파일에 결과 저장
                with open('x_train.csv', 'a', newline='') as w:
                    writer = csv.writer(w)
                    writer.writerows(x_train)

                # 데이터 전처리 및 모델 예측
                series2 = pd.read_csv('x_train.csv', header=None)
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp.fit(series2)
                series = imp.transform(series2)

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler = scaler.fit(series)
                normalized = scaler.transform(series)

                x_train = normalized[0:-1]
                x_test = normalized[-1]
                x_test = np.reshape(x_test, (1, -1))

                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf.fit(x_train)

                y_pred_test = clf.predict(x_test)
                if y_pred_test == 1:
                    print('\n EGD 수행이 적절하게 진행되어 1단계 합격입니다.\n')
                    str3 = 'pass.'
                else:
                    print('\nEGD 수행이 적절하게 진행되지 못했습니다. 1단계 불합격입니다. 다시 시도해 주세요 \n')
                    str3 = 'failure.'
                str4 = str(round(clf.decision_function(x_test)[0], 4))

                st.write(str4)

                if y_pred_test == 1:
                    st.success('EGD 수행이 적절하게 진행되어 1단계 합격입니다.')
                else:
                    st.error('EGD 수행이 적절하게 진행되지 못했습니다. 1단계 불합격입니다.')

        # BMP 파일 처리 (한 번만 실행)
        if has_bmp:
            progress_text.text("이미지 분석 중...")
            
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

                progress = int(((idx + 1) / len(bmp_files)) * 100)
                progress_text.text(f"이미지 분석 진행률: {progress}%")

            # 현재 날짜 가져오기
            current_date = datetime.now().strftime("%Y%m%d")
            
            # 결과 이미지 임시 저장
            temp_result_path = os.path.join(temp_dir, f'{name_endo}_{current_date}.png')
            result_image.save(temp_result_path)

            # 텍스트 추가
            text_position = (padding, single_width + padding * 2)  # 두 번째 줄에 위치
            text_color = (0, 0, 0)  # 검은색
            font_size = 12
            font_path = "C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc"  # Malgun Gothic 폰트의 절대 경로

            # 폰트 로드
            try:
                font = ImageFont.truetype(font_path, font_size)  # Noto Sans CJK 폰트 사용
            except OSError:
                st.error("폰트를 로드할 수 없습니다. 경로를 확인하세요.")

            # 추가할 텍스트
            text = f"Name: {name_endo}\n사진 수: {len(bmp_files)}\n시간: {datetime.now().strftime('%H:%M:%S')}\nstr3: {str3}\nstr4: {str4}"
            draw.text(text_position, text, fill=text_color, font=font)

            # Firebase Storage에 업로드
            result_blob = bucket.blob(f'EGD_skill_evaluation/test_results/{name_endo}_{current_date}.png')
            result_blob.upload_from_filename(temp_result_path)

            progress_text.text("이미지 분석 완료!")
            st.success(f"이미지 분석 결과가 저장되었습니다: {name_endo}_{current_date}.png")

            # 최종 결과 이미지 보여주기
            st.image(temp_result_path, caption='최종 분석 결과', use_column_width=True)  # 결과 이미지 표시

        # 임시 파일 정리
        for file_path in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file_path))
        os.rmdir(temp_dir)

        st.success("분석이 완료되었습니다.")