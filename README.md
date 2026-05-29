# Tech Hire Intelligence System Berbasis Artificial Intelligence (Data Scientist)

**ID Tim Capstone:** CC26-PSU379  
**Tema:** Future-ready Work & Economy  
**Peran Data Science:** Sistem Otomatisasi Rekomendasi & Prediksi Kesesuaian Kandidat  

---

## 📌 Latar Belakang & Permasalahan Bisnis
Proses rekrutmen tradisional sering kali memakan waktu lama karena pihak *Recruiter* atau HRD harus memeriksa ratusan berkas *Curriculum Vitae* (CV) pelamar kerja dan mencocokkannya dengan kriteria *Job Description* (JD) secara manual. Tantangan utama dalam proses ini meliputi:
1. **Volume Data yang Masif:** Tingginya kuantitas berkas pelamar membuat penyaringan awal menjadi sangat lambat.
2. **Subjektivitas Penilaian:** Penilaian manusia yang manual rentan terhadap bias ilmiah dan tidak konsisten.
3. **Miskalkulasi Kompetensi:** Sulitnya mengukur tingkat relevansi berkas teks secara objektif dan terukur.

Proyek ini hadir untuk membangun **Tech Hire Intelligence System**, sebuah sistem otomatis berbasis *Natural Language Processing* (NLP) yang mampu mengekstrak, menganalisis hubungan semantik teks, dan memberikan skor kecocokan (*Similarity Score*) kandidat secara instan dan objektif.

---

## Batasan Ruang Lingkup (Domain-Specific AI)
Sistem ini dirancang secara eksklusif untuk **Industri Teknologi / Ranah Pekerjaan IT**. Pembatasan ruang lingkup secara ketat ini diambil berdasarkan keputusan strategis Data Science untuk:
* **Meningkatkan Akurasi Ekstraksi:** Mencegah model melakukan "tebak buta" akibat tercampurnya istilah medis, finansial, atau *sales* yang tidak relevan dengan teknologi.
* **Optimalisasi Kamus Kompetensi:** Memberikan verifikasi *hard skills* yang tajam dan relevan dengan kebutuhan industri teknologi modern.

---

## 🛠️ Alur Kerja Teknis (Data Science Pipeline)

### 1. Data Preparation & Cleaning (`01_Data_Preparation_and_Cleaning.ipynb`)
Tahap awal berfokus pada pembersihan data mentah tekstual (`resumes_dataset.jsonl`) melalui langkah-langkah:
* **Penyelamatan Bug Fatal:** Memperbaiki struktur data di mana kolom isi teks CV dan *Job Description* sempat tertukar.
* **Text Preprocessing:** Menghapus *stopwords* bahasa Inggris secara manual, menangani *missing values*, dan menerapkan *lemmatization* untuk mendapatkan kata dasar teks murni.
* **Penyaringan Target:** Melakukan *filtering* data untuk mengisolasi kategori pekerjaan murni IT serta mendefinisikan fitur target `category`.

### 2. Feature Engineering & Modeling (`02_Feature_Engineering_and_Modeling.ipynb`)
Logika matematika pencocokan diimplementasikan secara modular ke dalam file `feature_engineering.py` dengan metode:
* **TF-IDF Vectorization:** Mengubah korpus teks CV dan JD bersih menjadi representasi matriks numerik.
* **Cosine Similarity:** Menghitung sudut kosinus antar-vektor teks untuk mengukur kedekatan semantik bahasa secara umum.
* **Rule-Based Skill Match Score:** Membuat fungsi kustom untuk menghitung persentase kecocokan kompetensi teknis (seperti *Python, SQL, Machine Learning*, dll) antara yang diminta perusahaan dengan yang dimiliki pelamar.
* **Experience Level Encoding:** Mendeteksi level pekerjaan (*Intern, Junior, Mid, Senior*) langsung dari teks lowongan.

### 3. Evaluasi Performa & A/B Testing (`03_Evaluation_and_Dashboard.ipynb`)
Validasi ilmiah dilakukan menggunakan metode *Offline A/B Testing* dengan membandingkan 30 sampel data uji *Gold Standard* (penilaian objektif manusia) terhadap dua jenis model:
* **Model A (Baseline):** Hanya mengandalkan perhitungan *Cosine Similarity* dasar.
* **Model B (Hybrid Scoring):** Menggabungkan *Cosine Similarity* dengan pendekatan kustom *Rule-Based Skill Match Score*.

**Hasil Pengujian Statistik (Independent T-Test):**
* **Rata-rata Error Model A:** 0.4406  
* **Rata-rata Error Model B:** 0.1152  
* **Nilai P-Value:** 0.000000 ($P < 0.05$)

**Kesimpulan Ilmiah:** Penurunan tingkat *error* pada Model B terbukti sangat signifikan secara statistik dan bukan faktor kebetulan. Pendekatan **Hybrid Scoring** jauh lebih akurat dan mendekati cara berpikir *recruiter* manusia.

---

## 🗂️ Struktur Repositori GitHub
```text
📁 Proyek_Capstone_CC26-PSU379/
│
├── 📁 data/                                  
│   └── resumes_dataset.jsonl                 <-- Dataset tekstual mentah (raw)
│   └── processed_data.csv                    <-- Dataset bersih siap latih
│
├── 📁 notebooks/                             
│   ├── 01_Data_Preparation_and_Cleaning.ipynb <-- Tahap penyelamatan & pembersihan data
│   ├── 02_Feature_Engineering_and_Modeling.ipynb <-- Tahap training model oleh tim AI
│   └── 03_Evaluation_and_Dashboard.ipynb     <-- Tahap analisis statistik A/B Testing
│
├── feature_engineering.py                    <-- Script modular fungsi matematika (Hybrid)
├── vectorizer_v10.pkl                        <-- Model TF-IDF pembaca teks
├── label_encoder_v10.pkl                     <-- Encoder label kategori
├── model_v10_finetuned.keras                 <-- Otak AI utama (Deep Learning Model)
├── requirements.txt                          <-- Library dependensi Python
└── README.md                                 <-- Dokumentasi utama proyek

(Catatan: File antarmuka aplikasi app.py berbasis Streamlit saat ini sedang dalam tahap pengembangan).

📈 Dampak Bisnis (Business Impact)
1. Efisiensi Waktu Screening: Memangkas durasi penyeleksian awal CV pelamar IT dari hitungan hari menjadi hitungan detik.

2. Rekomendasi Berakurasi Tinggi: Implementasi metode Hybrid memastikan kandidat yang direkomendasikan benar-benar menguasai stack tech spesifik yang dicari, menekan angka kesalahan rekrutmen (False Negatives).

3. Objektif & Standarisasi: Membantu perusahaan teknologi membangun standarisasi metrik penilaian kompetensi rekrutmen yang adil, transparan, dan berbasis data.
