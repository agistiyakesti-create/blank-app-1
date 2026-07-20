import os
import sys
import subprocess

# --- OTOMATIS INSTAL LIBRARY JIKA BELUM ADA ---
def install_packages():
    required_packages = ["matplotlib", "seaborn", "scikit-learn", "scipy"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Jalankan instalasi sebelum import utama
install_packages()

# --- BARU MASUK KE IMPORT UTAMA ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Clustering Cacat Manufaktur", layout="wide")

# Judul Utama Dashboard
st.title("🛠️ Dashboard Analisis Segmentasi Cacat Produk Manufaktur")
st.markdown("Aplikasi interaktif untuk mengelompokkan data cacat manufaktur menggunakan algoritma **K-Means** dan **Hierarchical Clustering** berdasarkan tingkat keparahan (*Severity*) dan biaya perbaikan (*Repair Cost*).")

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
    st.error("❌ Berkas 'defects_data.csv' tidak ditemukan di repositori GitHub Anda. Silakan unggah file tersebut ke GitHub agar aplikasi dapat membaca datanya.")
    st.stop()

# --- PREPROCESSING ---
severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
df['severity_score'] = df['severity'].map(severity_mapping)

X = df[['repair_cost', 'severity_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SIDEBAR INTERAKTIF ---
st.sidebar.header("⚙️ Pengaturan Model")
algo_choice = st.sidebar.selectbox("Pilih Algoritma Clustering:", ["K-Means", "Hierarchical (Agglomerative)"])
n_clusters = st.sidebar.slider("Jumlah Klaster (K):", min_value=2, max_value=5, value=3)

# --- TABS UNTUK STRUKTUR DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data (EDA)", "🎯 Hasil Model & Visualisasi", "💡 Insights & Rekomendasi"])

# ==================== TAB 1: EDA ====================
with tab1:
    st.header("🔍 Eksplorasi & Pemahaman Data")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Log Cacat", f"{len(df)} item")
    col2.metric("Rata-rata Biaya Perbaikan", f"${df['repair_cost'].mean():.2f}")
    col3.metric("Tipe Cacat Terbanyak", df['defect_type'].mode()[0])
    
    st.subheader("Sampel Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Distribusi Karakteristik Cacat")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='severity', palette='viridis', order=['Minor', 'Moderate', 'Critical'], ax=ax[0])
    ax[0].set_title("Distribusi Tingkat Keparahan (Severity)")
    
    sns.histplot(data=df, x='repair_cost', kde=True, color='teal', ax=ax[1])
    ax[1].set_title("Distribusi Biaya Perbaikan (Repair Cost)")
    st.pyplot(fig)

# ==================== TAB 2: MODELING ====================
with tab2:
    st.header(f"🎯 Pemodelan Menggunakan {algo_choice}")
    
    if algo_choice == "K-Means":
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        df['cluster'] = model.fit_predict(X_scaled)
        
        st.subheader("📈 Analisis Penentuan K Optimal: Elbow Method")
        wcss = []
        for k in range(1, 8):
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
            
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 3.5))
        ax_elbow.plot(range(1, 8), wcss, marker='o', linestyle='--', color='darkblue')
        ax_elbow.set_title("Elbow Method")
        ax_elbow.set_xlabel("Jumlah Klaster (K)")
        ax_elbow.set_ylabel("WCSS / Inersia")
        st.pyplot(fig_elbow)
        st.info("💡 **Analisis Sikut (Elbow):** Sudut patahan sikut paling optimal terlihat jelas pada **K = 3**, yang menandakan pembagian kelompok paling seimbang.")

    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        df['cluster'] = model.fit_predict(X_scaled)
        
        st.subheader("🌲 Analisis Penentuan K Optimal: Dendrogram")
        fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4))
        linkage_matrix = sch.linkage(X_scaled, method='ward')
        sch.dendrogram(linkage_matrix, ax=ax_dendro, no_labels=True)
        ax_dendro.set_title("Dendrogram Struktur Hierarki Kasus Cacat (Metode Ward)")
        ax_dendro.set_ylabel("Jarak Euclidean")
        st.pyplot(fig_dendro)
        st.info("💡 **Analisis Dendrogram:** Pohon keputusan menunjukkan adanya 3 cabang utama vertikal tertinggi yang tidak terpotong, memperkuat pilihan **K = 3** sebagai klasterisasi ideal.")

    st.subheader(f"🔮 Scatter Plot Hasil Segmentasi (K={n_clusters})")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=df, 
        x='repair_cost', 
        y='severity_score', 
        hue='cluster', 
        palette='Set1', 
        s=100, 
        alpha=0.8, 
        edgecolor='black',
        ax=ax_scatter
    )
    ax_scatter.set_title(f"Hasil Klasterisasi Fitur Cacat ({algo_choice})")
    ax_scatter.set_xlabel("Biaya Perbaikan ($ / Repair Cost)")
    ax_scatter.set_ylabel("Tingkat Keparahan (Severity Score)")
    ax_scatter.set_yticks([1, 2, 3])
    ax_scatter.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
    st.pyplot(fig_scatter)

    st.subheader("📊 Profil Karakteristik Rata-rata Tiap Kelompok")
    profil = df.groupby('cluster')[['repair_cost', 'severity_score']].mean()
    profil['Jumlah Sampel'] = df['cluster'].value_counts()
    st.dataframe(profil.style.format({"repair_cost": "${:.2f}", "severity_score": "{:.2f}"}), use_container_width=True)

# ==================== TAB 3: INSIGHTS ====================
with tab3:
    st.header("💡 Interpretasi Hasil & Analisis Insights Bisnis (K=3)")
    
    st.markdown("""
    Berdasarkan hasil pemetaan klaster menggunakan nilai **K=3** (jumlah optimal), manajemen operasional *Quality Assurance* (QA) dapat memetakan karakteristik cacat produk ke dalam **3 Segmen Bisnis Strategis**:
    """)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error("🚨 **Segmen 0: Cacat Biaya Tinggi, Keparahan Rendah–Sedang**")
        st.markdown("""
        * **Karakteristik:** Memiliki skor keparahan Minor/Moderate, namun memakan biaya perbaikan paling masif/mahal (Rata-rata \$760.67).
        * **Aksi Strategis:** Manajemen wajib menempatkan sistem kendali mutu otomatis (*Automated Testing*) utama pada lini ini guna mendeteksi kecacatan seawal mungkin untuk menekan pembengkakan biaya.
        """)
        
    with c2:
        st.warning("⚠️ **Segmen 1: Keparahan Kritikal (Semua Biaya)**")
        st.markdown("""
        * **Karakteristik:** Mengelompokkan semua cacat berkategori *Critical* dengan biaya perbaikan menengah (Rata-rata \$505.87).
        * **Aksi Strategis:** Fokus pada kalibrasi mesin utama dan pengawasan ketat operator demi mencegah produk cacat fatal lolos ke tangan konsumen.
        """)
        
    with c3:
        st.success("✅ **Segmen 2: Minoritas Efisien (Low-Cost / Minor)**")
        st.markdown("""
        * **Karakteristik:** Tingkat keparahan rendah (*Minor*) dengan pengeluaran finansial perbaikan yang relatif sangat kecil (Rata-rata \$258.60).
        * **Aksi Strategis:** Penanganan cukup diselesaikan dengan inspeksi visual manual reguler tanpa perlu restrukturisasi alat berskala besar.
        """)

    st.markdown("---")
    st.subheader("🧑‍💻 Kesimpulan Evaluasi Mandiri Mahasiswa")
    st.markdown("""
    1. **Standardisasi Data (`StandardScaler`):** Merupakan hal wajib karena rentang `repair_cost` jauh lebih besar daripada `severity_score`. Tanpa standardisasi, perhitungan jarak spasial akan didominasi sepenuhnya oleh fitur biaya, mengabaikan tingkat keparahan.
    2. **Komparasi Algoritma:** **K-Means** bekerja membagi ruang secara linear berdasarkan titik pusat massa (*centroid*), melahirkan batas klaster yang tegas secara geometris. Sementara **Hierarchical Clustering** membentuk kelompok secara bertahap (pohon keputusan), sangat unggul untuk melihat keintiman atau kedekatan struktur relasi antar-sampel secara granular.
    """)
