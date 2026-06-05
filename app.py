# main app untuk dashboard streamlit

import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go # buat bikin visualisasi gauge chart

# import fungsi dari modul feature engineering
from feature_engineering import calculate_similarity, calculate_skill_match

# setup layout halaman
st.set_page_config(page_title="Tech Hire Intelligence", page_icon="💼", layout="wide")

# bikin sidebar buat panduan hrd
st.sidebar.header("📌 Panduan HRD")
st.sidebar.info("""
**Cara Menggunakan Dashboard:**
1. Siapkan teks CV pelamar (Bahasa Inggris).
2. Siapkan teks lowongan/Job Desc (Bahasa Inggris).
3. Tempel di kolom yang tersedia dan klik **Analisis Kandidat**.
""")

st.sidebar.header("📊 Legenda Penilaian")
st.sidebar.success("**> 75% : Sangat Direkomendasikan**\nKandidat sangat cocok untuk lanjut ke tahap Interview Teknisi.")
st.sidebar.warning("**50% - 74% : Memenuhi Standar**\nKandidat memiliki potensi, cek portofolio secara manual.")
st.sidebar.error("**< 50% : Tidak Disarankan**\nKualifikasi kandidat jauh di bawah kebutuhan lowongan.")

st.sidebar.markdown("---")
st.sidebar.caption("Sistem didukung oleh AI Hybrid Scoring (Konteks Teks 60% + Kata Kunci Keahlian 40%) untuk mencegah manipulasi kata kunci pada CV.")

# bagian header utama
st.title("💼 Tech Hire Intelligence System")
st.markdown("""
**Asisten Cerdas Rekrutmen IT Anda.** Gunakan sistem ini untuk melakukan *screening* awal secara objektif dan instan. Sistem akan mengalkulasi kecocokan antara **Curriculum Vitae (CV)** kandidat dengan **Job Description (JD)**.
""")

# load model dan pkl, pakai cache biar ga berat pas di-refresh
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
    
# list skill buat dicek match-nya nanti
skills_list = [
    'python', 'sql', 'machine learning', 'deep learning', 'tensorflow',
    'pandas', 'power bi', 'tableau', 'excel', 'communication', 'data analysis',
    'react', 'node js', 'php', 'github', 'agile', 'scrum'
]

# form input teks
st.markdown("---")
st.info("💡 **Tips Rekrutmen:** Sistem dioptimalkan untuk membaca dokumen **Bahasa Inggris** (standar industri IT). Pastikan teks CV dan JD berbahasa Inggris untuk penilaian paling akurat.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Teks Curriculum Vitae (CV)")
    cv_input = st.text_area("Tempelkan seluruh teks CV pelamar di sini:", height=250, placeholder="Contoh: Experienced Data Scientist with 3 years of experience in Python, SQL...")

with col2:
    st.markdown("### 🏢 Teks Job Description (JD)")
    job_input = st.text_area("Tempelkan deskripsi pekerjaan (Job Desc) di sini:", height=250, placeholder="Contoh: We are looking for a Data Scientist who is proficient in Machine Learning...")

# eksekusi pas tombol diklik
if st.button("🚀 Mulai Analisis Kandidat", use_container_width=True):
    
    if cv_input.strip() == "" or job_input.strip() == "":
        st.warning("⚠️ Mohon isi kedua kolom teks (CV dan Job Desc) terlebih dahulu sebelum memulai analisis!")
    elif vectorizer and encoder and model:
        with st.spinner("AI sedang membaca dan mengevaluasi profil kandidat..."):
            
            # prediksi role kandidat
            cv_vector = vectorizer.transform([cv_input]).toarray()
            prediction = model.predict(cv_vector)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_category = encoder.inverse_transform([predicted_class_index])[0]

            # hitung skor hybrid
            cv_vec = vectorizer.transform([cv_input])
            job_vec = vectorizer.transform([job_input])
            
            sim_result = calculate_similarity(cv_vec, job_vec)
            
            if isinstance(sim_result, list) or type(sim_result).__module__ == np.__name__:
                sim_score = sim_result[0] * 100
            else:
                sim_score = sim_result * 100
                
            try:
                skill_result = calculate_skill_match(cv_input, job_input, skills_list)
            except TypeError:
                try:
                    skill_result = calculate_skill_match([cv_input], [job_input])
                except TypeError:
                    skill_result = calculate_skill_match(cv_input, job_input)

            if isinstance(skill_result, list) or type(skill_result).__module__ == np.__name__:
                skill_score = skill_result[0]
            elif hasattr(skill_result, 'iloc'): 
                skill_score = skill_result.iloc[0]
            else:
                skill_score = skill_result

            if skill_score <= 1.0:
                skill_score = skill_score * 100

            final_score = (sim_score * 0.6) + (skill_score * 0.4)
            
            # render hasil ke ui
            st.markdown("---")
            st.header("📊 Laporan Hasil Seleksi Kandidat")

            # guardrail buat ngecek ada hard skill it atau engga
            hard_skills = ['python', 'sql', 'machine learning', 'deep learning', 'tensorflow', 'pandas', 'react', 'node', 'php', 'github', 'java', 'html', 'css', 'javascript']
            cv_has_hard_skill = any(skill in cv_input.lower() for skill in hard_skills)

            # validasi kelayakan
            if not cv_has_hard_skill:
                st.error("🚨 **PERINGATAN: KANDIDAT DITOLAK OTOMATIS (OUT OF SCOPE)** 🚨")
                st.write(f"Meskipun sistem sempat memprediksi profil ini adalah **{predicted_category.title()}**, namun AI mendeteksi **0% Hard-Skill IT teknis** di dalam teks CV pelamar. Dokumen ini teridentifikasi sebagai Non-IT (Marketing/Sales/Admin/dll) dan tidak lolos tahap verifikasi.")
            
            elif confidence < 50.0:
                st.error("🚨 **PERINGATAN: KANDIDAT DITOLAK OTOMATIS (LOW VALIDITY)** 🚨")
                st.write(f"Tingkat Validitas Profil hanya **{confidence:.2f}%**. Sistem mendeteksi profil pelamar ini berada di luar ranah spesifikasi IT yang dibutuhkan, atau deskripsi keahliannya terlalu umum (Admin/Hardware/Non-Teknis).")
            
            elif skill_score < 15.0:
                st.warning("⚠️ **PERINGATAN: KEAHLIAN TEKNIS TERLALU RENDAH** ⚠️")
                st.write(f"CV ini valid di bidang IT, namun persentase kecocokan keahlian (Skill Match) sangat rendah ({skill_score:.1f}%).")
            
            else:
                st.success("✅ **KANDIDAT VALID (DOMAIN IT TERVERIFIKASI)**")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Keahlian Dominan Pelamar", predicted_category.title(), 
                          help="Berdasarkan CV, AI menebak posisi ini yang paling sesuai untuk keseharian pelamar.")
                m2.metric("Validitas Profil IT", f"{confidence:.1f}%", 
                          help="Keyakinan AI bahwa pelamar ini benar-benar praktisi IT (mencegah CV non-IT masuk).")
                m3.metric("Kecocokan Keahlian", f"{skill_score:.1f}%", 
                          help="Persentase jumlah keahlian teknis (hard-skill) di lowongan yang benar-benar dimiliki pelamar.")
                m4.metric("Total Skor Kecocokan", f"{final_score:.1f}%", 
                          help="Skor final (60% Relevansi Pengalaman + 40% Kecocokan Keahlian teknis).")

                st.markdown("---")

                # tampilin visualisasi gauge chart
                st.subheader("🎯 Kesimpulan Kelayakan Kandidat")
                
                # setup bentuk chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = final_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Total Match Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},   # merah
                            {'range': [50, 75], 'color': "khaki"},       # kuning
                            {'range': [75, 100], 'color': "lightgreen"}  # hijau
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': final_score
                        }
                    }
                ))
                
                # atur layout chart
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                
                # render chart ke web
                st.plotly_chart(fig, use_container_width=True)
                
                # hasil rekomendasi
                if final_score >= 75:
                    st.info("📌 **KEPUTUSAN:** Kandidat ini **SANGAT DIREKOMENDASIKAN**. Kompetensi dan pengalaman sangat selaras dengan kriteria lowongan. Prioritaskan untuk dihubungi.")
                elif final_score >= 50:
                    st.warning("📌 **KEPUTUSAN:** Kandidat ini **MEMENUHI STANDAR MINIMAL**. Memiliki kualifikasi dasar moderat. Disarankan untuk meninjau portofolio secara manual sebelum wawancara.")
                else:
                    st.error("📌 **KEPUTUSAN:** Kandidat ini **TIDAK DISARANKAN**. Tingkat kecocokan terlalu rendah terhadap profil pekerjaan yang dibutuhkan.")
                
                with st.expander("🔍 Bagaimana AI Menghitung Angka Ini? (Transparansi Penilaian)"):
                    st.markdown(f"""
                    Skor akhir **{final_score:.1f}%** di atas tidak muncul secara acak, melainkan dievaluasi objektif berdasarkan dua metrik:
                    1. **Relevansi Pengalaman & Konteks (Bobot 60%):** Mendapatkan skor **{sim_score:.1f}%**. AI menilai apakah narasi, tanggung jawab pekerjaan lama, dan latar belakang kandidat relevan dengan kebutuhan, meskipun penulisannya tidak sama persis.
                    2. **Pencocokan Kata Kunci Teknis (Bobot 40%):** Mendapatkan skor **{skill_score:.1f}%**. AI memindai secara presisi jumlah *software*, *tools*, atau bahasa pemrograman spesifik yang wajib dikuasai kandidat sesuai permintaan lowongan.
                    """)

                st.subheader("🛠️ Keahlian Utama yang Terverifikasi")
                matched_skills = []
                for skill in skills_list:
                    if skill in cv_input.lower() and skill in job_input.lower():
                        matched_skills.append(skill.title())

                if matched_skills:
                    st.write("Sistem mendeteksi pelamar menguasai keahlian berikut yang juga dibutuhkan di lowongan:")
                    cols = st.columns(4)
                    for idx, skill in enumerate(matched_skills):
                        cols[idx % 4].write(f"✅ **{skill}**")
                else:
                    st.write("Tidak ada kata kunci keahlian utama yang secara langsung cocok secara ejaan.")

st.markdown("---")
st.caption("🚀 Developed for Capstone Project | Meringankan Beban Rekrutmen IT dengan Bantuan Artificial Intelligence")
