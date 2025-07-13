import streamlit as st
import numpy as np
import joblib
import json

# === LOAD MODEL DAN PENDUKUNG ===
model = joblib.load("best_model_lgbm.pkl")  # ganti jika pakai GBM
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with open("kategori_tanaman.json", "r") as f:
    kategori_tanaman = json.load(f)

# === HEADER APLIKASI ===
st.title("🌱 Sistem Rekomendasi Tanaman Berdasarkan Kondisi Tanah")
st.markdown("Masukkan parameter tanah untuk mendapatkan rekomendasi tanaman yang cocok.")

# === FORM INPUT USER ===
n = st.number_input("🟩 Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
p = st.number_input("🟧 Phosphor (P)", min_value=0.0, max_value=200.0, value=42.0)
k = st.number_input("🟪 Kalium (K)", min_value=0.0, max_value=200.0, value=43.0)
temp = st.number_input("🌡️ Suhu (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("💧 Kelembaban (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input("🌍 pH Tanah", min_value=0.0, max_value=14.0, value=6.5)

# === PREDIKSI SAAT TOMBOL DIKLIK ===
if st.button("🔍 Prediksi Tanaman"):
    input_data = np.array([[n, p, k, temp, humidity, ph]])
    input_scaled = scaler.transform(input_data)

    y_pred = model.predict(input_scaled)
    nama_tanaman = label_encoder.inverse_transform(y_pred)[0]
    kategori = kategori_tanaman.get(nama_tanaman, "Tidak diketahui")

    st.success(f"🌾 Tanaman yang cocok: **{nama_tanaman.capitalize()}**")
    st.info(f"🧠 Kategori: **{kategori}**")

    # Tambahkan prediksi probabilitas (confidence score) jika tersedia
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        st.markdown("### 📊 Tiga Rekomendasi Teratas:")
        for idx in top_idx:
            label = label_encoder.inverse_transform([idx])[0]
            st.write(f"- {label.capitalize()} ({proba[idx]*100:.2f}%)")
