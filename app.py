import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Akıllı Mahsul Tavsiye Sistemi", layout="centered")
st.title("🌱 Akıllı Mahsul Tavsiye Sistemi")

model = joblib.load('crop_model.pkl')
le = joblib.load('label_encoder.pkl')

col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Azot (N)", 0, 140, 50)
    p = st.number_input("Fosfor (P)", 0, 145, 50)
    k = st.number_input("Potasyum (K)", 0, 205, 50)
    temp = st.number_input("Sıcaklık (°C)", 8.0, 45.0, 25.0)
with col2:
    hum = st.number_input("Nem (%)", 14.0, 100.0, 70.0)
    ph = st.number_input("pH Değeri", 3.5, 10.0, 6.5)
    rain = st.number_input("Yağış (mm)", 20.0, 300.0, 100.0)

if st.button("En Uygun Mahsulü Bul"):
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    pred = model.predict(input_data)
    st.success(f"✅ Bu koşullar için en ideal mahsul: **{le.inverse_transform(pred)[0].upper()}**")
