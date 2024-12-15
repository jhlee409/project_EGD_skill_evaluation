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
from datetime import datetime
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

# 파일의 업로드 및 파악
if uploaded_files:
    st.write(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
    if not name_endo:
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
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.avi'):
                avi_files.append(temp_path)
            elif uploaded_file.name.endswith('.mp4'):
                avi_files.append(temp_path)
            elif uploaded_file.name.endswith('.bmp'):
                has_bmp = True
                bmp_files.append(temp_path)

        st.write(f"AVI 파일 수: {len([file for file in avi_files if file.endswith('.avi')])}, MP4 파일 수: {len([file for file in avi_files if file.endswith('.mp4')])}, BMP 파일 수: {len(bmp_files)}")

        # AVI 파일 처리
        total_avi_files = len(avi_files)
        processed_files = 0

        for file_path in avi_files:
            camera = cv2.VideoCapture(file_path)
            if not camera.isOpened():
                st.error("동영상 파일을 열 수 없습니다.")
                continue

            length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = camera.get(cv2.CAP_PROP_FPS)
            duration = length / frame_rate

            st.write(f'동영상 길이: {int(duration // 60)} 분  {int(duration % 60)} 초')

            # 동영상 길이 체크
            if duration < 120 or duration > 330:  # 2분(120초)에서 5분(300초) 사이 체크
                st.error(f"동영상 길이가 {int(duration // 60)}분 {int(duration % 60)}초로 2분에서 5분 30초 사이의 범위를 벗어납니다. 더이상 분석은 진행되지 않습니다.")
                break  # 분석 중단

            st.write("\n[DEBUG] 비디오 분석 시작...")
            st.write(f"[DEBUG] 비디오 정보: 총 프레임 수={length}, 프레임 레이트={frame_rate:.2f}")

            try:
                # 동영상 분석 창 생성
                cv2.namedWindow('Video Analysis', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Video Analysis', 800, 600)

                # 프레임 처리를 위한 변수 초기화
                pts = []
                angle_g = np.array([])
                distance_g = np.array([])
                frame_count = 0
                last_time = time.time()

                while True:
                    ret, frame = camera.read()
                    if not ret:
                        st.write("[DEBUG] 더 이상 읽을 프레임이 없습니다.")
                        break

                    # 프레임 처리 시간 계산
                    current_time = time.time()
                    fps = 1 / (current_time - last_time)
                    last_time = current_time

                    # 프레임 카운트 증가
                    frame_count += 1

                    try:
                        # 프레임 분석
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        # 색상 범위 설정 및 마스크 생성
                        green_lower = np.array([35, 80, 50], np.uint8)
                        green_upper = np.array([100, 255, 255], np.uint8)
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
                                st.write(f"[DEBUG] 프레임 {frame_count}: 중심점 계산 실패")

                            pts.extend([px, py])

                            # 최소 외접원 계산
                            ((cx, cy), radius) = cv2.minEnclosingCircle(u)
                            center = (int(cx), int(cy))
                            radius = int(radius)
                            pts.append(radius)

                            # 원 그리기
                            if radius > 8:
                                cv2.circle(frame, center, 30, (0, 0, 255), -1)

                        # 디버그 정보 표시
                        debug_info = f'Frame: {frame_count}/{length} | FPS: {fps:.1f}'
                        cv2.putText(frame, debug_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # 분석 진행률 표시
                        progress = f'Progress: {(frame_count) / length * 100:.1f}%'
                        cv2.putText(frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # 프레임 표시
                        cv2.imshow('Video Analysis', frame)
                        
                        # 진행률 표시
                        st.write(f'[INFO] 분석 진행률: {(frame_count) / length * 100:.1f}% | FPS: {fps:.1f}')

                        # 키 입력 처리
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC 키를 누르면 분석 중단
                            st.write("\n[DEBUG] 사용자가 ESC 키를 눌러 분석을 중단했습니다.")
                            break

                    except Exception as e:
                        st.write(f"\n[ERROR] 프레임 {frame_count} 처리 중 오류 발생: {str(e)}")
                        continue

                st.write("\n[DEBUG] 분석 완료")
                st.write(f"[DEBUG] 처리된 총 프레임 수: {frame_count}")
                st.write(f"[DEBUG] 수집된 데이터 포인트 수: {len(pts)}")

            except Exception as e:
                st.write(f"\n[ERROR] 비디오 처리 중 치명적 오류 발생: {str(e)}")
            finally:
                # 분석 완료 후 정리
                camera.release()
                cv2.destroyAllWindows()
                st.write("[DEBUG] 리소스 정리 완료")

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

            y_pred_test = clf.predict(x_test_scaled)
            if y_pred_test == 1:
                str3 = 'pass.'
                st.write('EGD 수행이 적절하게 진행되어 검사 과정 평가에서는 합격입니다.')
            else:
                str3 = 'failure.'
                st.write('EGD 수행이 적절하게 진행되지 못했습니다. 검사 과정 평가에서 불합격입니다.')
            str4 = str(round(clf.decision_function(x_test_scaled)[0], 4))
            st.write(f"판단 점수: {str4}")

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
            text = f"photo number: {len(bmp_files)}\nduration: {video_length}\nresult: {str3}\nstr4: {str4}"

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
            st.subheader("- 이미지 저장 과정 -")
            
            # 임시 디렉토리 생성
            os.makedirs('EGD_skill_evaluation/test_results', exist_ok=True)
            
            # 결과 이미지 저장
            temp_image_path = f'EGD_skill_evaluation/test_results/{name_endo}_{current_date}.png'
            result_image.save(temp_image_path)
            
            try:
                # Firebase Storage에 업로드
                result_blob = bucket.blob(f'EGD_skill_evaluation/test_results/{name_endo}_{current_date}.png')
                result_blob.upload_from_filename(temp_image_path)
                
                st.success(f"이미지가 Firebase Storage에 성공적으로 저장되었습니다: {name_endo}_{current_date}.png")
                st.image(temp_image_path, use_container_width=True)
            except Exception as e:
                st.error(f"Firebase Storage 업로드 중 오류 발생: {str(e)}")
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    cleanup_temp_files()
    st.divider()
    st.success("평가가 완료되었습니다.")