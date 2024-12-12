import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

 # Check if Firebase app has already been initialized
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
    firebase_admin.initialize_app(cred)


bucket = storage.bucket('amcgi-bulletin.appspot.com') 

# Streamlit UI
st.title("EGD Skill Evaluation")
name_endo = st.text_input("본인의 성명을 한글로 입력해 주세요:")

if st.button("분석 시작"):
    if name_endo:
        st.write(f"분석을 시작합니다, {name_endo}님.")

        # Firebase에서 파일 다운로드
        blobs = bucket.list_blobs(prefix='EGD_skill_evaluation/test/')
        for blob in blobs:
            # 파일 경로를 디렉토리와 파일 이름으로 설정
            file_name = os.path.basename(blob.name)  # blob의 이름을 가져옵니다.
            file_path = os.path.join('temp', 'EGD_skill_evaluation', 'test', file_name)  # 전체 파일 경로를 만듭니다.
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 디렉토리가 없으면 생성합니다.
            blob.download_to_filename(file_path)  # 파일을 다운로드합니다.
            st.write(f"다운로드 완료: {blob.name}")

            # EGD_skill_evaluation.py의 처리 과정 적용
            if file_path.endswith('.avi'):
                st.write(f"동영상 파일 분석: {file_path}")
                camera = cv2.VideoCapture(file_path)
                if not camera.isOpened():
                    st.error(f"동영상 파일을 열 수 없습니다: {file_path}")
                    continue

                length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_rate = camera.get(cv2.CAP_PROP_FPS)
                duration = length / frame_rate

                st.write(f'동영상 길이(초): {int(duration)}')
                st.write(f'동영상 frame 수: {length}')

                if length < 8000 or length > 13000:
                    st.error("불합격: 권장 검사 시간 범위를 벗어났습니다.")
                    break

                ret, frame = camera.read()
                pts = deque()
                ii = 1
                angle_g = []
                distance_g = []

                while ret:
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
                        if delta_g > 6:
                            st.write(f"빠른 움직임 감지: 프레임 {points[i][0]}")
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
                str4 = str(round(clf.decision_function(x_test)[0], 4))

                if y_pred_test == 1:
                    st.success('EGD 수행이 적절하게 진행되어 1단계 합격입니다.')
                else:
                    st.error('EGD 수행이 적절하게 진행되지 못했습니다. 1단계 불합격입니다.')

            elif file_path.endswith('.bmp'):
                st.write(f"이미지 파일 분석: {file_path}")
                img = cv2.imread(file_path)
                # 이미지 처리 코드 추가 가능

        st.success("분석이 완료되었습니다.")
    else:
        st.error("이름을 입력해 주세요.")