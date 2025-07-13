import streamlit as st
import numpy as np
import joblib
import json

# --- FUNGSI NAVIGASI BARU ---
# Fungsi ini akan mengubah state halaman, yang akan membuat radio button di sidebar ikut berubah
def go_to_predict():
    st.session_state.navigasi = "🔬 Prediksi Tanaman"

def go_to_home():
    st.session_state.navigasi = "🏠 Halaman Utama"

# --- FUNGSI VALIDASI INPUT ---
def validate_input(n, p, k, temp, humidity, ph):
    """Memvalidasi input pengguna agar berada dalam rentang yang wajar."""
    warnings = []
    if n < 10 and n != 0: warnings.append("Nilai Nitrogen (N) terlihat sangat rendah dan tidak realistis.")
    if p < 10 and p != 0: warnings.append("Nilai Fosfor (P) terlihat sangat rendah dan tidak realistis.")
    if k < 10 and k != 0: warnings.append("Nilai Kalium (K) terlihat sangat rendah dan tidak realistis.")
    if not (10 <= temp <= 45): warnings.append(f"Suhu ({temp}°C) berada di luar rentang wajar untuk pertanian (10-45°C).")
    if not (20 <= humidity <= 100): warnings.append(f"Kelembapan ({humidity}%) berada di luar rentang wajar (20-100%).")
    if not (3.5 <= ph <= 9): warnings.append(f"Nilai pH Tanah ({ph}) berada di luar rentang wajar (3.5-9).")
    return warnings

def main():
    # Set page config
    st.set_page_config(
        page_title="🌱 Crop Predictor App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # --- PEMUATAN MODEL DAN OBJEK PENDUKUNG ---
    try:
        model = joblib.load("best_model_lgbm.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        with open("kategori_tanaman.json", "r") as f:
            kategori_tanaman = json.load(f)
        model_ready = True
    except Exception as e:
        st.error(f"Gagal memuat file model atau pendukungnya: {e}")
        st.warning("Fungsi prediksi tidak akan berjalan. Pastikan file .pkl, .json, dan library yang dibutuhkan (seperti lightgbm) sudah benar.")
        model_ready = False

    # --- Load CSS ---
    # CSS Anda disatukan di sini agar lebih ringkas
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .main { background-color: #121212; color: white; }
        
        .subtitle { color:rgb(255, 255, 255); font-size: 1.5rem; text-align: center; margin-bottom: 1rem; font-weight: 400; }
        
        .info-section { background: #1a1a1a; border-radius: 12px; padding: 24px; margin: 16px 0; border-left: 4px solid #1ed760; }
        .info-section h3 { color: #1ed760; margin-bottom: 12px; font-weight: 600; }
        .info-section p { color: #b3b3b3; line-height: 1.6; margin-bottom: 8px; }
        
        /* Mengubah warna radio button di sidebar */
        div[data-baseweb="radio"] > label > div:first-child {
            background-color: white !important;
        }

        .stButton > button {
            width: 100%; /* Membuat tombol memenuhi kontainer */
            background: #1ed760; color: #000000; border: none; border-radius: 24px;
            padding: 12px 24px; font-weight: 600; font-size: 14px; transition: all 0.3s ease;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .stButton > button:hover { background: #1db954; transform: scale(1.02); }
        
        .stNumberInput > div > div > input { background: #2a2a2a; border: 1px solid #404040; color: white; border-radius: 8px; }
        .stNumberInput > div > div > input:focus { border-color: #1ed760; box-shadow: 0 0 0 2px rgba(30, 215, 96, 0.2); }
        
        .prediction-result {
            background: linear-gradient(135deg, #1ed760, #1db954); border-radius: 12px;
            padding: 24px; text-align: center; color: #000000; margin: 16px 0; font-weight: 600;
        }
        .prediction-result h2 { font-size: 2rem; margin-bottom: 8px; }
        .prediction-result p { font-size: 1.1rem; margin: 4px 0; }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Menu
    with st.sidebar:
        st.markdown("### 🎯 Menu")
        # Gunakan 'key' agar state-nya bisa diubah secara programatis
        page = st.radio(
            "Pilih halaman:",
            ["🏠 Halaman Utama", "🔬 Prediksi Tanaman"],
            key="navigasi"
        )
        st.markdown("---")
        st.info("Aplikasi ini dibuat oleh Rafif Huda untuk memprediksi tanaman yang cocok berdasarkan kondisi tanah.")

    # Main Content
    if page == "🏠 Halaman Utama":
        st.markdown("<h1 class='subtitle'>Sistem Prediksi Tanaman Berbasis Machine Learning</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-section">
            <h3>🔬 Tentang Sistem</h3>
            <p>Sistem cerdas yang menggunakan teknologi machine learning untuk memprediksi jenis tanaman yang paling cocok berdasarkan kondisi tanah Anda. Sistem ini dibuat dengan menggunakan model LightGBM yang telah dilatih dengan data yang akurat.</p>
            <p>Sistem ini menganalisis parameter seperti nitrogen (N), fosfor (P), kalium (K), suhu, kelembapan, dan pH tanah untuk memberikan rekomendasi yang akurat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-section">
            <h3>📋 Cara Penggunaan</h3>
            <p>1. Klik tombol "Mulai Prediksi Sekarang" di bawah ini atau pilih menu "Prediksi Tanaman" dari sidebar.</p>
            <p>2. Masukkan data kondisi tanah Anda pada formulir yang tersedia.</p>
            <p>3. Klik tombol "Prediksi" untuk mendapatkan rekomendasi tanaman.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- TOMBOL NAVIGASI BARU ---
        # Tombol ini memanggil fungsi go_to_predict saat diklik
        st.button("Mulai Prediksi Sekarang 🚀", on_click=go_to_predict, type="primary")
        
    elif page == "🔬 Prediksi Tanaman":
        # --- TOMBOL NAVIGASI BARU (DIBUAT LEBIH KECIL) ---
        # Menempatkan tombol di dalam kolom untuk mengontrol lebarnya
        button_cols = st.columns(5) # Buat 5 kolom agar tombol lebih kecil
        with button_cols[0]:
            st.button("⬅️ Kembali", on_click=go_to_home)

        st.markdown("<h1 class='subtitle' style='text-align:left;'>🔬 Prediksi Tanaman</h1>", unsafe_allow_html=True)
        st.markdown("<p>Masukkan parameter kondisi tanah untuk mendapatkan rekomendasi.</p>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.markdown("### 🧪 Parameter Kondisi Tanah")
            col1, col2, col3 = st.columns(3)
            with col1:
                n = st.number_input("Nitrogen (N)", min_value=0, max_value=500, help="Kandungan Nitrogen di tanah (kg/ha)")
                p = st.number_input("Fosfor (P)", min_value=0, max_value=200, help="Kandungan Fosfor di tanah (kg/ha)")
            with col2:
                k = st.number_input("Kalium (K)", min_value=0, max_value=400, help="Kandungan Kalium di tanah (kg/ha)")
                temp = st.number_input("Suhu (°C)", min_value=0.0, max_value=50.0, format="%.2f", help="Suhu rata-rata lingkungan dalam Celcius")
            with col3:
                humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, format="%.2f", help="Kelembapan relatif udara dalam persen")
                ph = st.number_input("pH Tanah", min_value=0.0, max_value=14.0, format="%.2f", help="Tingkat keasaman tanah")
            
            submitted = st.form_submit_button("🚀 Prediksi Tanaman")
        
        if submitted:
            validation_warnings = validate_input(n, p, k, temp, humidity, ph)
            if n == 0 or p == 0 or k == 0:
                st.warning("⚠️ Harap isi semua parameter kondisi tanah.")
            elif validation_warnings:
                st.error("Input Tidak Wajar Terdeteksi!")
                for warning in validation_warnings:
                    st.warning(f"-> {warning}")
                st.info("Harap masukkan nilai yang lebih realistis untuk mendapatkan prediksi yang akurat.")
            elif not model_ready:
                st.error("Prediksi tidak dapat dilakukan karena model gagal dimuat.")
            else:
                input_data = np.array([[n, p, k, temp, humidity, ph]])
                input_scaled = scaler.transform(input_data)
                y_pred = model.predict(input_scaled)
                nama_tanaman = label_encoder.inverse_transform(y_pred)[0]
                kategori = kategori_tanaman.get(nama_tanaman, "Tidak diketahui")

                st.markdown("---")
                st.subheader("✨ Hasil Rekomendasi ✨")
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🌾 Tanaman yang Cocok: {nama_tanaman.capitalize()}</h2>
                    <p>Kategori: {kategori}</p>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                    top_idx = np.argsort(proba)[::-1][:3]
                    
                    st.markdown("### 📊 Rekomendasi Tanaman Teratas:")
                    for idx in top_idx:
                        label = label_encoder.inverse_transform([idx])[0]
                        score = proba[idx] * 100
                        st.markdown(f"- **{label.capitalize()}** dengan tingkat keyakinan **{score:.2f}%**")

if __name__ == "__main__":
    main()
