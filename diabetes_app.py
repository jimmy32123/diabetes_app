import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------
# 1. 모델 및 스케일러 로드 (C드라이브 절대 경로 강제 지정)
# -------------------------------------------------------------
@st.cache_resource
def load_ml_objects():
    # 주소창에 뜨는 절대 경로 그대로 지정합니다.
    scaler_path = r"C:\Diabetes\scaler.pkl"
    model_path = r"C:\Diabetes\diabetes_pred_model (1).pkl"
    
    # 파일이 해당 경로에 진짜 있는지 확인 후 로드
    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return scaler, model
    return None, None

scaler, log_model_eng = load_ml_objects()

# -------------------------------------------------------------
# 2. 페이지 기본 설정 및 스타일 정의
# -------------------------------------------------------------
st.set_page_config(page_title="당뇨병 AI 예측 대시보드", layout="centered", page_icon="🩺")

st.title("🩺 당뇨병 발병 위험도 예측 AI")
st.caption("Machine Learning기반 건강 지표 분석 및 예측 시스템")

# 안내 문구 상자
st.info("💡 **이용 안내:** 아래의 건강 지표들을 입력하신 후, 하단의 **[분석 시작하기]** 버튼을 눌러주세요.")

# 모델 로드 상태 검증
if scaler is None or log_model_eng is None:
    st.error("❌ **시스템 오류:** `C:\\DIABETES` 폴더 내에서 `scaler.pkl` 또는 `diabetes_pred_model(1).pkl` 파일을 찾을 수 없습니다.")
    st.info("💡 **체크리스트:** 파일 확장자가 혹시 `pkl.txt`로 숨겨져 있지는 않은지, 혹은 파일명에 대소문자나 오타가 없는지 확인해보세요.")
    st.stop()

st.markdown("---")

# -------------------------------------------------------------
# 3. 데이터 입력 영역 (Form 형태로 깔끔하게 묶음)
# -------------------------------------------------------------
st.subheader("📋 건강 지표 데이터 입력")

with st.form(key="health_data_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        preg = st.number_input("🤰 임신 횟수", min_value=0, value=0, step=1)
        skin = st.number_input("💪 피부 두께", min_value=0.0, value=20.0, step=1.0)
        dpf = st.number_input("🧬 가족력 지수", min_value=0.0, value=0.5, step=0.01)
        
    with c2:
        glucose = st.number_input("🩸 공복 혈당 수치", min_value=0.0, value=100.0, step=1.0)
        insulin = st.number_input("🧪 인슐린 수치", min_value=0.0, value=80.0, step=1.0)
        age = st.number_input("🎂 나이", min_value=0, value=30, step=1)
        
    with c3:
        bp = st.number_input("💓 혈압 (이완기)", min_value=0.0, value=70.0, step=1.0)
        bmi = st.number_input("⚖️ 체질량지수 (BMI)", min_value=0.0, value=25.0, step=0.1)
        st.write("") 
        
    st.markdown(" ")
    submit_button = st.form_submit_button(label="🔮 분석 시작하기", use_container_width=True)

# -------------------------------------------------------------
# 4. 분석 결과 출력 영역
# -------------------------------------------------------------
if submit_button:
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=['이완기', '공복혈당', '혈압', '피부두께', '인슐린', '체질량지수', '당뇨병계층함수', '나이']
    )

    input_data['대사위험점수'] = (
        (input_data['공복혈당'] >= 126).astype(int) + 
        (input_data['혈압'] >= 90).astype(int) + 
        (input_data['체질량지수'] >= 30).astype(int)
    )
    input_data['비만여부'] = (input_data['체질량지수'] >= 25).astype(int)
    input_data['고령위험'] = (input_data['나이'] >= 50).astype(int)
    input_data['혈당_비만상호작용'] = input_data['공복혈당'] * input_data['체질량지수']
    input_data['인슐린저항성'] = input_data['공복혈당'] * input_data['인슐린']

    try:
        input_scaled = scaler.transform(input_data)
        predicted = log_model_eng.predict(input_scaled)[0]
        prob = log_model_eng.predict_proba(input_scaled)[0][1] * 100

        st.markdown("---")
        st.subheader("📊 AI 분석 결과 대시보드")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if predicted == 1:
                st.error("### 🚨 당뇨 위험군 분류\n현재 입력된 지표 기준, 당뇨 위험 단계에 해당할 확률이 높습니다. 정밀 진단을 권장합니다.")
            else:
                st.success("### ✅ 정상 군 분류\n현재 입력된 지표 기준, 당뇨 질환으로부터 안전한 정상 수치 범위에 있습니다.")
                
        with res_col2:
            st.metric(label="당뇨 발병 확률", value=f"{prob:.1f} %")

        st.markdown("**당뇨 위험도 수치화**")
        st.progress(int(prob))
        
        with st.expander("🔍 AI 가공 데이터프레임 확인 (파생변수 포함)"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 시스템 연산 중 에러가 발생했습니다: {e}")