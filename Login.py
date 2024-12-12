import streamlit as st
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage
import os

st.set_page_config(page_title="EGD_skill_evaluation", layout="wide")

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

# Streamlit 페이지 설정
st.title("EGD_skill_evaluation")
st.header("Login page")
st.markdown(
    '''
    1. 이 게시판은 서울 아산병원 GI 상부 EGD_skill_evaluation을 위한 웹페이지입니다.
    1. 로그인 한 후 왼쪽 메뉴에서 EGD_skill_evaluation을 선택한 후 이용해 주세요.
    1. 최종결과는 자동적으로 제출됩니다.
    '''
)
st.divider()

# 사용자 입력
ID = st.text_input("ID")
password = st.text_input("Password", type="password")

# 로그인 버튼
if st.button("Login"):
    if ID == "amcgi" and password == "3180":
        st.success(f"로그인에 성공하셨습니다. 이제 왼쪽의 메뉴를 이용하실 수 있습니다.")
        st.session_state['logged_in'] = True
        st.session_state['user_ID'] = ID
    else:
        st.error("로그인에 실패했습니다. ID 또는 비밀번호를 확인하세요.")

# 로그아웃 버튼
if "logged_in" in st.session_state and st.session_state['logged_in']:
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.success("로그아웃 되었습니다.")
