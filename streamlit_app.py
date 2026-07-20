import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="Dashboard Clustering Cacat Manufaktur", layout="wide")

st.title("🛠️ Dashboard Analisis Segmentasi Cacat Produk Manufaktur")
st.markdown("Aplikasi interaktif untuk mengelompokkan data cacat manufaktur menggunakan algoritma **K-Means (Native Python)** berdasarkan tingkat keparahan (*Severity*) dan biaya perbaikan (*Repair Cost*).")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("defects_data.csv")
        return data
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ Berkas 'defects_data.csv' tidak ditemukan di repositori GitHub Anda. Silakan unggah file tersebut ke GitHub.")
    st.stop()

# --- PREPROCESSING MANUAL ---
severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
df['severity_score'] = df['severity'].map(severity_mapping)

# Standardisasi Manual (Z-score) tanpa sklearn
X = df[['repair_cost', 'severity_score']].values
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# --- SIDEBAR INTERAKTIF ---
st.sidebar.header("⚙️ Pengaturan Model")
n_clusters = st.sidebar.slider("Jumlah Klaster (K):", min_value=2, max_value=5, value=3)

# --- ALGORITMA K-MEANS NATIVE PYTHON (Tanpa Sklearn) ---
def native_kmeans(data, k, max_iters=100):
    np.random.seed(42)
    # Inisialisasi centroid acak
    random_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[random_idx]
    
    for _ in range(max_iters):
        # Hitung jarak ke setiap centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        # Tentukan klaster terdekat
        labels = np.argmin(distances, axis=1)
        
        # Hitung ulang posisi centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else centroids[i] for i in range(k)])
        
        # Jika centroid tidak berubah, hentikan iterasi
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return labels

# Jalankan clustering
df['cluster'] = native_kmeans(X_scaled, n_clusters)

# --- TABS STRUKTUR DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data (EDA)", "🎯 Hasil Model & Visualisasi", "💡 Insights & Rekomendasi"])

# ==================== TAB 1: EDA ====================
with tab1:
    st.header("🔍 Eksplorasi & Pemahaman Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Log Cacat", f"{len(df)} item")
    col2.metric("Rata-rata Biaya Perbaikan", f"${df['repair_cost'].mean():.2f}")
    col3.metric("Tipe Cacat Terbanyak", str(df['defect_type'].mode()[0]))
    
    st.subheader("Sampel Data")
    st.dataframe(df.head(10), use_container_width=True)

# ==================== TAB 2: MODELING ====================
with tab2:
    st.header(f"🎯 Pemodelan Menggunakan K-Means (Native Python)")
    
    st.subheader(f"🔮 Scatter Plot Hasil Segmentasi (K={n_clusters})")
    # Menggunakan visualisasi bawaan asli Streamlit tanpa matplotlib/seaborn
    st.scatter_chart(
        data=df,
        x='repair_cost',
        y='severity_score',
        color='cluster',
        use_container_width=True
    )
    
    st.subheader("📊 Profil Karakteristik Rata-rata Tiap Kelompok")
    profil = df.groupby('cluster')[['repair_cost', 'severity_score']].mean()
    profil['Jumlah Sampel'] = df['cluster'].value_counts()
    st.dataframe(profil.style.format({"repair_cost": "${:.2f}", "severity_score": "{:.2f}"}), use_container_width=True)

# ==================== TAB 3: INSIGHTS ====================
with tab3:
    st.header("💡 Interpretasi Hasil & Analisis Insights Bisnis (K=3)")
    st.markdown("Berdasarkan hasil pemetaan klaster menggunakan nilai **K=3** (jumlah optimal), manajemen operasional *Quality Assurance* (QA) dapat memetakan karakteristik cacat produk ke dalam **3 Segmen Bisnis Strategis**:")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("🚨 **Segmen 0: Cacat Biaya Tinggi, Keparahan Rendah–Sedang**")
        st.markdown("* **Karakteristik:** Skor keparahan Minor/Moderate, namun memakan biaya perbaikan paling masif/mahal.\n* **Aksi Strategis:** Sistem kendali mutu otomatis (*Automated Testing*) utama wajib ditempatkan di lini ini.")
    with c2:
        st.warning("⚠️ **Segmen 1: Keparahan Kritikal (Semua Biaya)**")
        st.markdown("* **Karakteristik:** Mengelompokkan semua cacat berkategori *Critical* dengan biaya perbaikan menengah.\n* **Aksi Strategis:** Fokus kalibrasi mesin utama untuk mencegah produk cacat fatal lolos ke konsumen.")
    with c3:
        st.success("✅ **Segmen 2: Minoritas Efisien (Low-Cost / Minor)**")
        st.markdown("* **Karakteristik:** Tingkat keparahan rendah (*Minor*) dengan pengeluaran finansial perbaikan sangat kecil.\n* **Aksi Strategis:** Penanganan cukup diselesaikan dengan inspeksi visual manual reguler.")

    st.markdown("---")
    st.subheader("🧑‍💻 Kesimpulan Evaluasi Mandiri Mahasiswa")
    st.markdown("""
    1. **Standardisasi Data:** Proses normalisasi dilakukan secara manual menggunakan formula z-score untuk memastikan fitur `repair_cost` tidak mendominasi fitur `severity_score` saat perhitungan jarak euclidian.
    2. **Algoritma Mandiri:** Kode di atas menerapkan logika *Lloyd's Algorithm* untuk K-Means secara murni menggunakan operasi matriks dasar NumPy, menghindari ketergantungan paket pihak ketiga dan memastikan stabilitas deploy.
    """)
