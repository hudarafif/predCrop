import streamlit as st
import numpy as np
import joblib
import json

# --- FUNGSI VALIDASI INPUT (dari kode sebelumnya) ---
def validate_input(n, p, k, temp, humidity, ph):
    """Memvalidasi input pengguna agar berada dalam rentang yang wajar."""
    warnings = []
    # Cek nilai minimal untuk N, P, K
    if n < 10 and n != 0: warnings.append("Nilai Nitrogen (N) terlihat sangat rendah dan tidak realistis.")
    if p < 10 and p != 0: warnings.append("Nilai Fosfor (P) terlihat sangat rendah dan tidak realistis.")
    if k < 10 and k != 0: warnings.append("Nilai Kalium (K) terlihat sangat rendah dan tidak realistis.")

    # Cek rentang wajar untuk suhu, kelembapan, dan pH
    if not (10 <= temp <= 45): warnings.append(f"Suhu ({temp}°C) berada di luar rentang wajar untuk pertanian (10-45°C).")
    if not (20 <= humidity <= 100): warnings.append(f"Kelembapan ({humidity}%) berada di luar rentang wajar (20-100%).")
    if not (3.5 <= ph <= 9): warnings.append(f"Nilai pH Tanah ({ph}) berada di luar rentang wajar (3.5-9).")
    
    return warnings

def main():
    # Set page config
    st.set_page_config(
        page_title="🌱 App",
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
        st.warning("Fungsi prediksi tidak akan berjalan. Pastikan file .pkl dan .json ada di folder yang benar.")
        model_ready = False

    # Load CSS
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("File 'style.css' tidak ditemukan. Menggunakan gaya default Streamlit.")

    # Sidebar Menu
    with st.sidebar:
        st.markdown("### 🎯 Menu")
        page = st.radio(
            "Pilih halaman:",
            ["🏠 Halaman Utama", "🔬 Prediksi Tanaman"],
            index=0,
            key="navigasi"
        )
        
        st.markdown("---")
    
    # Main Content
    if page == "🏠 Halaman Utama":
        # Header
        st.markdown("""
        <p class="subtitle">Sistem Prediksi Tanaman Berbasis Machine Learning</p>
        """, unsafe_allow_html=True)
        
        # About Section
        st.markdown("""
        <div class="info-section">
            <h3>🔬 Tentang Sistem</h3>
            <p>Sistem cerdas yang menggunakan teknologi machine learning untuk memprediksi jenis tanaman yang paling cocok berdasarkan kondisi tanah Anda. Sistem ini dibuat dengan menggunakan model LightGBM yang telah dilatih dengan data yang akurat.</p>
            <p>Sistem ini menganalisis parameter seperti nitrogen (N), fosfor (P), kalium (K), suhu, kelembapan, dan pH tanah untuk memberikan rekomendasi yang akurat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # How to Use
        st.markdown("""
        <div class="info-section">
            <h3>📋 Cara Penggunaan</h3>
            <p>1. Pilih menu "Prediksi Tanaman" dari sidebar</p>
            <p>2. Masukkan data kondisi tanah Anda</p>
            <p>3. Klik tombol "Prediksi" untuk mendapatkan rekomendasi</p>
            <p>4. Lihat hasil prediksi </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <p>Dibuat Rafif Huda dengan ❤️</p>
            <p style="color: #666; font-size: 0.9rem;">© 2025.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "🔬 Prediksi Tanaman":
        st.markdown("""
        <h1 class="main-title">🔬 Prediksi Tanaman</h1>
        <p class="subtitle">Masukkan parameter kondisi tanah untuk mendapatkan rekomendasi</p>
        """, unsafe_allow_html=True)
        
        # Input Form
        with st.form("prediction_form"):
            st.markdown("### 🧪 Parameter Kondisi Tanah")
            col1, col2, col3 = st.columns(3)
            with col1:
                n = st.number_input("Nitrogen (N)", min_value=0, max_value=500)
                p = st.number_input("Fosfor (P)", min_value=0, max_value=200)
            with col2:
                k = st.number_input("Kalium (K)", min_value=0, max_value=400)
                temp = st.number_input("Suhu (°C)", min_value=0.0, max_value=50.0)
            with col3:
                humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0)
                ph = st.number_input("pH Tanah", min_value=0.0, max_value=14.0)
            
            submitted = st.form_submit_button("🚀 Prediksi Tanaman")
        
        # Results
        if submitted:
            # Panggil fungsi validasi dari kode sebelumnya
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
                # Jika semua validasi lolos, jalankan prediksi
                input_data = np.array([[n, p, k, temp, humidity, ph]])
                input_scaled = scaler.transform(input_data)
                y_pred = model.predict(input_scaled)
                nama_tanaman = label_encoder.inverse_transform(y_pred)[0]
                kategori = kategori_tanaman.get(nama_tanaman, "Tidak diketahui")

                st.markdown("---")
                st.subheader("✨ Hasil Rekomendasi ✨")
                
                # Menampilkan hasil dengan format dinamis
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
