import streamlit as st
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import unicodedata

# --- 1. 경로 설정 (폴더 구조에 맞게 수정됨) ---
# 이미지 폴더와 데이터 폴더 경로를 상대 경로로 지정합니다.
추천이미지경로 = "images"
추천pool경로 = "data/추천_pool.csv"

# 모델 파일들도 'models' 폴더 안에 있는 것으로 경로를 수정했습니다.
# 주의: 실제 models 폴더 안의 파일명과 아래 이름이 정확히 일치해야 합니다!
tflite_model_paths = {
    "카테고리": "models/카테고리.tflite",
    "상의_반팔_스타일": "models/상의_반팔_스타일.tflite",
    "상의_반팔_색상": "models/상의_반팔_색상.tflite",
    "상의_긴팔_스타일": "models/상의_긴팔_스타일.tflite",
    "상의_긴팔_색상": "models/상의_긴팔_색상.tflite",
    "하의_반바지_스타일": "models/하의_반바지_스타일.tflite",
    "하의_반바지_색상": "models/하의_반바지_색상.tflite",
    "하의_긴바지_스타일": "models/하의_긴바지_스타일.tflite",
    "하의_긴바지_색상": "models/하의_긴바지_색상.tflite",
}

category_labels = ["상의_반팔", "상의_긴팔", "하의_반바지", "하의_긴바지"]
style_labels = ["댄디", "미니멀", "스트릿", "캐주얼"]
color_labels = ["무채색", "밝은색", "어두운색"]

# --- 2. 모델 로드 및 예측 함수 (캐싱 적용으로 속도 향상) ---

@st.cache_resource
def load_tflite_model(model_path):
    """TFLite 모델을 로드하고 텐서를 할당합니다. (캐싱 사용)"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"모델을 불러오는 중 오류가 발생했습니다: {model_path}\n오류 내용: {e}")
        return None

def run_tflite_model(interpreter, input_image):
    if interpreter is None:
        return np.zeros(10) # 더미 반환
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB").resize(target_size)
    img_array = (np.asarray(img).astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0), img

def predict_all_auto(image_file):
    img_array, preview_img = preprocess_image(image_file)
    
    # 1. 카테고리 예측
    category_interpreter = load_tflite_model(tflite_model_paths["카테고리"])
    category_probs = run_tflite_model(category_interpreter, img_array)
    category_pred = category_labels[np.argmax(category_probs)]

    # 2. 해당 카테고리의 스타일/색상 모델 로드
    style_model_path = tflite_model_paths.get(f"{category_pred}_스타일")
    color_model_path = tflite_model_paths.get(f"{category_pred}_색상")
    
    style_interpreter = load_tflite_model(style_model_path)
    color_interpreter = load_tflite_model(color_model_path)
    
    style_probs = run_tflite_model(style_interpreter, img_array)
    color_probs = run_tflite_model(color_interpreter, img_array)

    return {
        "카테고리": category_pred,
        "스타일": style_labels[np.argmax(style_probs)],
        "색상": color_labels[np.argmax(color_probs)],
        "preview": preview_img
    }

# --- 3. 추천 로직 함수들 ---

def get_opposite_categories(category_label):
    if "상의" in category_label:
        return ["하의_반바지", "하의_긴바지"]
    elif "하의" in category_label:
        return ["상의_반팔", "상의_긴팔"]
    return []

def 날씨에_어울리는_조건들(weather):
    if weather == "맑은 날":
        return ["밝은색", "무채색"], ["상의_반팔", "하의_반바지"]
    elif weather == "흐린 날":
        return ["어두운색", "무채색"], ["상의_긴팔", "하의_긴바지"]
    elif weather == "비 오는 날":
        return ["밝은색", "무채색"], ["상의_긴팔", "하의_긴바지"]
    return color_labels, category_labels

def recommend_best(pool_df, style, color, weather, k=3):
    if pool_df.empty:
        return pd.DataFrame()

    날씨색상, 날씨카테고리 = 날씨에_어울리는_조건들(weather)
    
    # 날씨에 맞는 카테고리 필터링
    가능한_카테고리 = set(pool_df["카테고리"].unique())
    추천가능_카테고리 = list(set(날씨카테고리).intersection(가능한_카테고리))
    pool_df = pool_df[pool_df["카테고리"].isin(추천가능_카테고리)]

    # 1순위: 스타일 + 색상(정확) + 날씨색상
    cond1 = (
        (pool_df["스타일"] == style) &
        (pool_df["색상"] == color) &
        (pool_df["색상"].isin(날씨색상))
    )
    res1 = pool_df[cond1]
    if len(res1) >= k:
        return res1.sample(k)

    # 2순위: 스타일 + 색상(정확) (날씨 조건 완화)
    cond2 = (
        (pool_df["스타일"] == style) &
        (pool_df["색상"] == color)
    )
    res2 = pool_df[cond2 & ~cond1]
    if len(res1) + len(res2) >= k:
        return pd.concat([res1, res2]).sample(k)

    # 3순위: 스타일만 일치 (색상 조건 완화)
    cond3 = (pool_df["스타일"] == style)
    res3 = pool_df[cond3 & ~(cond1 | cond2)]
    if len(res1) + len(res2) + len(res3) >= k:
        return pd.concat([res1, res2, res3]).sample(k)

    # Fallback: 위 조건들에서 찾은 거라도 긁어모아서 추천
    fallback = pd.concat([res1, res2, res3])
    if not fallback.empty:
        return fallback.sample(min(k, len(fallback)))

    # 최후의 수단: 전체에서 랜덤
    return pool_df.sample(min(k, len(pool_df)))

# --- 4. Streamlit UI 메인 화면 ---
st.set_page_config(page_title="오늘 뭐 입지?", layout="centered")
st.title("👕 오늘 뭐 입지?")
st.markdown("당신의 스타일에 날씨까지 고려해 추천해드려요!")

# 세션 상태 초기화
if "style_pref" not in st.session_state:
    st.session_state.style_pref = None
if "color_pref" not in st.session_state:
    st.session_state.color_pref = None

# 사용자 설정 입력
with st.expander("🛠️ 선호 스타일 및 색상 설정", expanded=True):
    style = st.selectbox("선호 스타일을 선택해주세요:", style_labels)
    color = st.selectbox("선호 색상을 선택해주세요:", color_labels)
    if st.button("✅ 설정 저장"):
        st.session_state.style_pref = style
        st.session_state.color_pref = color
        st.success(f"저장 완료! 🎉 선호 스타일: {style}, 색상: {color}")

st.subheader("🌤️ 오늘 날씨는 어떤가요?")
weather = st.radio("날씨 선택", ["맑은 날", "흐린 날", "비 오는 날"], horizontal=True)

st.subheader("📸 배경 제거된 옷 사진 업로드")
uploaded_file = st.file_uploader("상의 또는 하의 이미지를 업로드하세요 (배경 제거된 이미지)", type=["jpg", "jpeg", "png"])

# 이미지 업로드 시 예측 및 추천 실행
if uploaded_file and st.session_state.style_pref and st.session_state.color_pref:
    # 1. 예측 실행
    result = predict_all_auto(uploaded_file)
    st.image(result["preview"], caption="업로드된 이미지", use_column_width=True)

    st.success(f"예측 결과: {result['카테고리']} / {result['스타일']} / {result['색상']}")
    st.markdown(f"☁️ 선택한 날씨: **{weather}**")

    # 2. 추천 데이터 준비
    # images 폴더에 있는 실제 파일 확인
    try:
        실제파일들 = os.listdir(추천이미지경로)
        파일매핑 = {unicodedata.normalize("NFC", f).lower(): f for f in 실제파일들}
    except FileNotFoundError:
        st.error(f"오류: '{추천이미지경로}' 폴더를 찾을 수 없습니다. 폴더 구조를 확인해주세요.")
        st.stop()

    # 데이터 로드 및 파일명 매핑
    try:
        추천_pool = pd.read_csv(추천pool경로)
    except FileNotFoundError:
        st.error(f"오류: '{추천pool경로}' 파일을 찾을 수 없습니다. data 폴더를 확인해주세요.")
        st.stop()
        
    추천_pool["파일명"] = 추천_pool["파일명"].str.strip().apply(lambda x: unicodedata.normalize("NFC", x.lower()))
    추천_pool = 추천_pool[추천_pool["파일명"].isin(파일매핑)]
    추천_pool["파일명"] = 추천_pool["파일명"].map(파일매핑)

    # 3. 추천 알고리즘 실행
    opposite_categories = get_opposite_categories(result["카테고리"])
    추천_pool_filtered = 추천_pool[추천_pool["카테고리"].isin(opposite_categories)]

    추천결과 = recommend_best(
        추천_pool_filtered,
        st.session_state.style_pref,
        st.session_state.color_pref,
        weather
    )

    # 4. Fallback 로직 (결과 없을 시 대안 찾기)
    if 추천결과.empty:
        st.warning("❗조건에 맞는 반대 카테고리 추천이 없어 선호 스타일/색상 기반으로 재탐색합니다.")
        후보1 = 추천_pool[
            (추천_pool["스타일"] == st.session_state.style_pref) &
            (추천_pool["색상"] == st.session_state.color_pref)
        ]
        if not 후보1.empty:
            추천결과 = 후보1.sample(min(3, len(후보1)))
        else:
            후보2 = 추천_pool[
                (추천_pool["스타일"] == st.session_state.style_pref)
            ]
            if not 후보2.empty:
                추천결과 = 후보2.sample(min(3, len(후보2)))
            else:
                st.warning("🥲 최종 fallback: 전체 pool에서 임의 추천합니다.")
                추천결과 = 추천_pool.sample(min(3, len(추천_pool)))

    # 5. 결과 출력
    st.subheader("🔍 추천 결과")
    if 추천결과.empty:
        st.warning("추천 가능한 이미지가 아예 없습니다. data/추천_pool.csv와 images 폴더를 확인해주세요.")
    else:
        # 컬럼을 나누어 보기 좋게 출력
        cols = st.columns(len(추천결과))
        for idx, (_, row) in enumerate(추천결과.iterrows()):
            img_path = os.path.join(추천이미지경로, row["파일명"])
            with cols[idx]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{row['스타일']}, {row['색상']}")
                else:
                    st.warning("이미지 없음")
else:
    if not uploaded_file:
        st.info("👆 위에서 옷 이미지를 업로드해주세요.")
    elif not (st.session_state.style_pref and st.session_state.color_pref):
        st.info("👆 '선호 스타일 및 색상 설정'에서 설정을 저장해주세요.")
