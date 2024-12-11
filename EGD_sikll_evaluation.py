from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import csv
import math
import os
from math import atan2, degrees
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
from shutil import copyfile
from sklearn.impute import SimpleImputer

# 변수 초기화jh
pts = deque()
px = 0
py = 0
distance_r = []
distance_b = []
distance_g = []
x_train = []
fast2 = []
ii = 1
angle_g = []
length = 0
mean_b = 0
std_b = 0
mean_g = 0
std_g = 0

name_endo = input("\n본인의 성명을 한글로 입력해 주세요 : ")

blue_lower = np.array([90, 50, 50], np.uint8)
blue_upper = np.array([130, 255, 255], np.uint8)

# 이 green 색의 값은 HSV 색 공간에서의 값입니다.
green_lower = np.array([35, 50, 50], np.uint8)
green_upper = np.array([85, 255, 255], np.uint8)

dirname = r'test'
for (path, dir, files) in os.walk(dirname):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        image_path = os.path.join(path, filename)

        if ext == '.mp4':
            camera = cv2.VideoCapture(image_path)
            
            # 동영상 파일이 제대로 열렸는지 확인
            if not camera.isOpened():
                print(f"동영상 파일을 열 수 없습니다: {image_path}")
                continue  # 다음 파일로 넘어갑니다.

            length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = camera.get(cv2.CAP_PROP_FPS)  # 프레임 속도 가져오기
            duration = length / frame_rate  # 동영상 길이(초)

            print('\n동영상 길이(초):', int(duration))
            print('\n동영상 frame 수(권장 frame 값 ; 7200 - 7920) :', length)

            # 동영상 길이 체크 조건 수정
            if length < 7200 or length > 7920:
                print("\n불합격: 권장 검사 시간 범위(7200 - 7920 프레임)를 벗어났습니다. 다시 하십시오\n")
                break
            else:
                ret, frame = camera.read()

            hsv_values = []

            # 초기 중심점과 윤곽선 저장
            initial_center = None
            initial_contour = None

            while ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                green = cv2.inRange(hsv, green_lower, green_upper)

                contours3, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours3:
                    g = max(contours3, key=cv2.contourArea)
                    ga = cv2.contourArea(g)

                    # 초기 윤곽선과 중심점 설정
                    if initial_contour is None:
                        initial_contour = g
                        M = cv2.moments(g)
                        if M["m00"] != 0:
                            initial_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        else:
                            initial_center = (0, 0)

                    # 윤곽선 유사성 비교
                    similarity = cv2.matchShapes(initial_contour, g, cv2.CONTOURS_MATCH_I1, 0.0)
                    if similarity < 0.1:  # 유사성이 높으면 같은 물체로 간주
                        M = cv2.moments(g)
                        if M["m00"] != 0:
                            px = abs(int(M["m10"] / M["m00"]))
                            py = abs(int(M["m01"] / M["m00"]))
                        else:
                            px, py = 0, 0

                        # 초기 중심점과의 거리 계산
                        distance = math.sqrt((px - initial_center[0])**2 + (py - initial_center[1])**2)
                        print(f"중심점 이동 거리: {distance}")

                        # 중심점과 반지름 저장
                        pts.append(px)
                        pts.append(py)
                        ((cx, cy), radius) = cv2.minEnclosingCircle(g)
                        pts.append(radius)

                        if radius > 8:
                            cv2.circle(frame, (px, py), 30, (0, 0, 255), -1)

                # 프레임을 새로운 윈도우에 표시
                cv2.imshow('Video Analysis', frame)

                # 'q' 키를 눌러서 종료할 수 있도록 설정
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                ret, frame = camera.read()

            camera.release()
            cv2.destroyAllWindows()

            k = list(pts)
            array_k = np.array(k)

            frame_no = array_k[0::5]
            timesteps = len(frame_no)
            frame_no2 = np.reshape(frame_no, (timesteps, 1))

            color = array_k[1::5]
            color2 = np.reshape(color, (timesteps, 1))

            x_value = array_k[2::5]
            timesteps = len(x_value)
            x_value2 = np.reshape(x_value, (timesteps, 1))

            y_value = array_k[3::5]
            y_value2 = np.reshape(y_value, (timesteps, 1))

            radius2 = array_k[4::5]
            radius3 = np.reshape(radius2, (timesteps, 1))

            # 가장 작은 배열의 크기로 timesteps 설정
            min_timesteps = min(len(frame_no2), len(color2), len(x_value2), len(y_value2), len(radius3))

            # 배열 크기 조정
            frame_no2 = frame_no2[:min_timesteps]
            color2 = color2[:min_timesteps]
            x_value2 = x_value2[:min_timesteps]
            y_value2 = y_value2[:min_timesteps]
            radius3 = radius3[:min_timesteps]

            points = np.hstack([frame_no2, color2, x_value2, y_value2, radius3])

            y_value4 = []
            for i in range(timesteps - 1):
                if i + 12 < timesteps:
                    if (points[i][3] != 1) and (points[i + 5][3] != 1) and (points[i + 12][3] != 1):
                        y_value3 = points[i][3]
                        y_value4 = np.append(y_value4, y_value3)
                    else:
                        y_value4 = np.append(y_value4, 0)
                else:
                    y_value4 = np.append(y_value4, 0)

            for i in range(timesteps - 1):
                if (points[i][1] != 3 and points[i + 1][1] != 3) and (points[i][1] == 1 and points[i + 1][1] == 1):
                    a = points[i + 1][2] - points[i][2]
                    b = points[i + 1][3] - points[i][3]
                    rr = points[i][4]
                    delta_b = (math.sqrt((a * a) + (b * b))) / rr
                    distance_b = np.append(distance_b, delta_b)
                else:
                    distance_b = np.append(distance_b, 0)

                if (points[i][1] != 3 and points[i + 1][1] != 3) and (points[i][1] == 2 and points[i + 1][1] == 2):
                    a = points[i + 1][2] - points[i][2]
                    b = points[i + 1][3] - points[i][3]
                    angle_g = np.append(angle_g, degrees(atan2(a, b)))
                    rr = points[i][4]
                    delta_g = (math.sqrt((a * a) + (b * b))) / rr
                    if delta_g > 6:
                        fast2 = np.append(fast2, points[i][0])
                    distance_g = np.append(distance_g, delta_g)
                else:
                    distance_g = np.append(distance_g, 0)

            angle_gg = [iiii for iiii in angle_g if (iiii == 0)]
            angle_ggg = [abs(iiiii) for iiiii in angle_g if iiiii != 0]
            mean_ggg = np.mean(angle_ggg)
            std_ggg = np.std(angle_ggg)
            print('\njerky movement 횟수(숫자가 클 수록 흔들 사진이 찍힐 능성이 높습니다. 권장 20 이하)) :  ', len(fast2))
            distance = [iii for iii in distance_g if iii != 0]
            steps = len(distance)
            xx = np.arange(0.0, steps, 1)

            distance_bb = [bbb for bbb in distance_b if bbb < 6]
            mean_b = np.mean(distance_bb)
            std_b = np.std(distance_bb)
            if mean_b == 0:
                print('\n불합격입니다. 십이지장 2nd portion을 관찰하지 않았습니다. 다시 시도해 주세요')
            mean_b = round(mean_b, 4)

            distance_gg = [ggg for ggg in distance_g if ggg < 6]
            mean_g = np.mean(distance_gg)
            std_g = np.std(distance_gg)
            x_train = []
            x_train.append([mean_g, std_g])

            x_train = np.resize(x_train, (1, 2))
            x_train = np.array(x_train)

    copyfile('x_train_expert.csv', 'x_train.csv')

    with open('x_train.csv', 'a', newline='') as w:
        writer = csv.writer(w)
        writer.writerows(x_train)

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
        print('\n EGD 수행이 적절하게 진행되어 1단계 합격입니다.\n')
        print('2단계; 사진 촬영 평가')
        print(':test 폴더에 있는 test.png를 expert photo.pptx와 비교해 보고 만족스러우면 파일을 제출해 주세요')
        str3 = 'pass.'
    else:
        print('\nEGD 수행이 적절하게 진행되지 못했습니다. 1단계 불합격입니다. 다시 시도해 주세요 \n')
        str3 = 'failure.'

w = 12
h = 12
fig = plt.figure(figsize=(12, 12))
columns = 9
rows = 9
q = 1
i = 1

for (path, dir, files) in os.walk(dirname):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        image_path = os.path.join(path, filename)

        if ext == '.bmp':
            img = mpimg.imread(image_path)
            width = np.size(img, 1)
            if width > 410:
                im1 = Image.open(image_path)
                im2 = im1.crop((170, 40, 570, 440))  # 400x400
            else:
                im1 = Image.open(image_path)
                im2 = im1
            image_name = "test_" + "%02d" % i + ext
            new_image_path = os.path.join(path, image_name)
            im2.save(new_image_path)
            if image_path == new_image_path:
                pass
            else:
                os.remove(image_path)
            i = i + 1

img_list = glob.glob(os.path.join(dirname, '*.bmp'))
print("\ntotal image number : ", len(img_list))
if len(img_list) < 62 or len(img_list) > 66:
    print('\n불합격; 권장 사진 수(62 - 66장)를 벗어났습니다. 다시 도전해 보세요\n')

for ii in img_list:
    img = mpimg.imread(ii)
    plt.figure(1)
    plt.subplot(rows, columns, q, aspect='equal')
    plt.axis('off')
    
    # 이미지 크기를 두 배로 조정
    img_resized = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    plt.imshow(img_resized)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    # 이미지 이미지 내부 밑부분에 추가
    image_name = os.path.basename(ii)
    plt.text(0.5, 0.05, image_name, fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    
    q = q + 1

plt.savefig('test_result.png')

image4 = cv2.imread('test_result.png')
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

str_total = str(name_endo) + " " + str(length) + "   " + str(len(fast2)) + "   " + str(mean_b) + "   " + str4 + "   " + str3 + "   " + str(len(img_list))
height, width, _ = image4.shape
pt = (20, height - 20)

# PIL을 사용하여 한글 텍스트 추가
image_pil = Image.fromarray(image4)
draw = ImageDraw.Draw(image_pil)
font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)  # 적절한 한글 글꼴 경로로 수정
draw.text(pt, str_total, font=font, fill=(0, 0, 0))

# 다시 OpenCV 형식으로 변환
image4 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
plt.imsave('test_result.png', image4)
plt.show()

# try:
#     files = os.listdir(dirname)
#     for file in files:
#         os.remove(os.path.join(dirname, file))
# except PermissionError as e:
#     print(f"파일 삭제 중 오류 발생: {e}")