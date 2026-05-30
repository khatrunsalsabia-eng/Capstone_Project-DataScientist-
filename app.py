# =========================================================
# TECH HIRE INTELLIGENCE - STREAMLIT DASHBOARD
# =========================================================

import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Memanggil fungsi modular dari tim
from feature_engineering import calculate_similarity, calculate_skill_match

# --- 1. SETUP HALAMAN STREAMLIT ---
st.set_page_config(page_title="Tech Hire Intelligence", page_icon="🤖", layout="wide")

# --- 2. SIDEBAR (Fitur dari Kode Lama) ---
st.sidebar.header("📌 Tentang Proyek")
st.sidebar.info("""
**Capstone Project - Coding Camp 2026**

Sistem Prediksi & Rekomendasi Kandidat IT Berbasis AI (Hybrid NLP).

**Fitur Utama:**
- Kategori Pekerjaan (Deep Learning)
- Analisis Kemiripan (Cosine Similarity)
- Pencocokan Keahlian (Skill Match)
- Filter Kepercayaan (Mencegah Non-IT)
""")

# --- 3. JUDUL UTAMA ---
st.title("🤖 Tech Hire Intelligence System")
st.markdown("""
Dashboard ini menganalisis tingkat kecocokan antara **Curriculum Vitae (CV)** kandidat 
dengan **Job Description (JD)** menggunakan pendekatan NLP dan Machine Learning.
""")

# --- 4. LOAD MODEL & VECTORIZER ---
@st.cache_resource
def load_ai_models():
    try:
        
        vectorizer = joblib.load('tfidf_vectorizer_final.pkl')
        encoder = joblib.load('label_encoder_final.pkl')
        
        model = load_model('tech_hire_model_final.keras') 
        
        return vectorizer, encoder, model
    except Exception as e:
        st.error(f"Gagal memuat model AI. Pastikan file .pkl dan .keras ada di folder yang benar. Error: {e}")
        return None, None, None
    
vectorizer, encoder, model = load_ai_models()
    
# --- 5. DAFTAR SKILL (Untuk Visualisasi Centang) ---
# Diambil dari kodemu yang lama agar visualisasi keahlian tetap muncul
skills_list = [
    'python', 'sql', 'machine learning', 'deep learning', 'tensorflow',
    'pandas', 'power bi', 'tableau', 'excel', 'communication', 'data analysis',
    'react', 'node js', 'php', 'github', 'agile', 'scrum'
]

# --- 6. UI INPUT TEKS ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Teks Curriculum Vitae (CV)")
    cv_input = st.text_area("Tempelkan seluruh teks CV pelamar di sini:", height=250)

with col2:
    st.markdown("### 🏢 Teks Job Description (JD)")
    job_input = st.text_area("Tempelkan deskripsi lowongan pekerjaan (IT) di sini:", height=250)

# --- 7. PROSES PREDIKSI & SCORING ---
if st.button("🚀 Analisis Kandidat (Run AI)", use_container_width=True):
    
    if cv_input.strip() == "" or job_input.strip() == "":
        st.warning("⚠️ Mohon isi kedua kolom teks (CV dan Job Desc) terlebih dahulu!")
    elif vectorizer and encoder and model:
        with st.spinner("AI sedang menganalisis dokumen teks dan hubungan semantik..."):
            
            # A. PREDIKSI MENGGUNAKAN DEEP LEARNING
            cv_vector = vectorizer.transform([cv_input]).toarray()
            prediction = model.predict(cv_vector)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_category = encoder.inverse_transform([predicted_class_index])[0]

            # B. PERHITUNGAN HYBRID SCORING
            sim_score = calculate_similarity(cv_input, job_input) * 100
            skill_score = calculate_skill_match(cv_input, job_input)
            final_score = (sim_score + skill_score) / 2

            # C. TAMPILAN HASIL & CONFIDENCE FILTER
            st.markdown("---")
            st.header("📊 Hasil Analisis AI")

            # Logika Cerdas: Filter Kepercayaan
            if confidence < 50.0:
                st.error("🚨 **PERINGATAN: CV DITOLAK (NON-IT / OUT OF DOMAIN)** 🚨")
                st.write(f"Tingkat keyakinan AI sangat rendah ({confidence:.2f}%). Teks kemungkinan besar bukan dari ranah IT.")
            else:
                st.success("✅ **KANDIDAT VALID (DOMAIN IT)**")
                
                # Menampilkan 4 Metrik Utama
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kategori Diprediksi", predicted_category)
                m2.metric("Keyakinan AI (Confidence)", f"{confidence:.1f}%")
                m3.metric("Skill Match Score", f"{skill_score:.1f}%")
                m4.metric("Hybrid Final Score", f"{final_score:.1f}%")

                # Progress Bar
                st.subheader("Tingkat Kecocokan (Compatibility Level)")
                st.progress(int(final_score))
                
                # Kesimpulan & Interpretasi
                st.subheader("Interpretasi & Rekomendasi")
                if final_score >= 75:
                    st.info("💡 **Rekomendasi:** Kandidat ini **SANGAT DIREKOMENDASIKAN**. Kandidat sangat cocok dengan kebutuhan pekerjaan.")
                elif final_score >= 50:
                    st.warning("💡 **Rekomendasi:** Kandidat ini **MEMENUHI STANDAR MINIMAL**. Cocok secara moderat.")
                else:
                    st.error("💡 **Rekomendasi:** Kandidat ini **TIDAK DISARANKAN**. Kecocokan rendah terhadap kebutuhan pekerjaan.")

                # Fitur Centang Skill dari Kode Lama Aqila
                st.subheader("🛠️ Keahlian yang Terdeteksi Cocok")
                matched_skills = []
                for skill in skills_list:
                    if skill in cv_input.lower() and skill in job_input.lower():
                        matched_skills.append(skill.title())

                if matched_skills:
                    for skill in matched_skills:
                        st.write(f"✅ {skill}")
                else:
                    st.write("Tidak ada keahlian spesifik yang terdeteksi cocok secara langsung.")

st.markdown("---")
st.markdown("Developed for Capstone Project | CV Matching using NLP & Machine Learning")
