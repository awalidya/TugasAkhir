import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import davies_bouldin_score, silhouette_score
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Styling sidebar dan header
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #D5E5D5 !important;
        color: black !important;
    }
    .custom-title {
        font-family: 'Great Vibes', cursive;
        font-size: 18px;
        font-weight: bold;
        display: block;
        color: black !important;
    }
    .move-down {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar dengan logo dan navigasi
with st.sidebar:
    col1, col2 = st.columns([4, 3])
    with col1:
        st.image("https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png", width=200)
    with col2:
        st.markdown("<span class='custom-title move-down'>Trash Achievement</span>", unsafe_allow_html=True)

    tabs = ["Halaman Utama", "Upload Data", "Pemodelan", "Visualisasi"]
    for tab in tabs:
        if st.button(tab):
            st.session_state.selected_tab = tab

# State default
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Halaman Utama'

# Daftar kolom numerik
numeric_columns = [
    'sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan',
    'penanganan', 'perc_penanganan', 'sampah_terkelola', 'perc_sampah_terkelola', 'daur_ulang'
]
feature_outlier = ['sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan', 'penanganan', 'sampah_terkelola']
scaling_columns = ['sampah_tahunan', 'pengurangan', 'penanganan']

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

# === Halaman Utama ===
if st.session_state.selected_tab == "Halaman Utama":
    st.write("""
    Selamat datang di platform analisis wilayah berbasis pengelolaan sampah.
    \nMelalui pendekatan data dan metode klastering, kami menyajikan gambaran menyeluruh tentang bagaimana berbagai daerah di Indonesia menangani permasalahan sampah. 
    Dengan memetakan wilayah berdasarkan pola pengurangan, penanganan, dan daur ulang sampah, platform ini diharapkan dapat menjadi acuan bagi pengambil kebijakan, 
    peneliti, maupun masyarakat umum dalam mendorong pengelolaan sampah yang lebih efektif dan berkelanjutan.
    """)

# === Upload Data ===
elif st.session_state.selected_tab == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df)

        # Missing value
        st.subheader("🧱 Missing Value Sebelum Penanganan")
        missing_before = df.isnull().sum()
        for col, count in missing_before.items():
            if count > 0:
                st.markdown(f"- **{col}**: {count} missing value")
        if missing_before.sum() == 0:
            st.success("Tidak ada missing value yang terdeteksi.")

        handle_missing_values(df)
        st.session_state.df = df

        st.subheader("🧹 Missing Value Setelah Penanganan")
        missing_after = df.isnull().sum()
        if missing_after.sum() == 0:
            st.success("Semua missing value telah berhasil ditangani!")

        # Outlier sebelum
        st.subheader("Plot Outlier Sebelum Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for i, col in enumerate(numeric_columns[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        st.pyplot(fig)

        # Outlier sesudah
        for col in feature_outlier:
            handle_outliers_iqr(df, col)

        st.subheader("Plot Outlier Setelah Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for i, col in enumerate(feature_outlier[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        st.pyplot(fig)

        # Scaling
        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
        X = df[scaling_columns].values
        st.session_state.df = df

        st.subheader("Data Setelah Scaling")
        st.dataframe(df[scaling_columns].head())

        st.subheader("EDA")
        st.dataframe(df[scaling_columns].describe().T)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, column in enumerate(df[scaling_columns]):
            sns.histplot(df[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Histogram of {column}')
        st.pyplot(fig)

        correlation_matrix_selected = df[scaling_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix_selected, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap for Selected Features")
        st.pyplot(fig)

# === Pemodelan ===
elif st.session_state.selected_tab == "Pemodelan":
    if 'df' in st.session_state:
        df = st.session_state.df.copy()
        X = df[scaling_columns].values

        st.subheader("Pemodelan Clustering dengan Mean Shift")
        custom_bw = st.number_input("🎛️ Sesuaikan nilai Bandwidth", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

        if st.button("🚀 Jalankan Clustering"):
            ms = MeanShift(bandwidth=custom_bw, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            n_clusters = len(np.unique(labels))

            df['cluster_labels'] = labels
            st.session_state.df = df
            st.session_state.ms_final = ms

            st.success(f"Jumlah klaster terbentuk: {n_clusters}")

            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='plasma', marker='o', label='Data Points')
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                       s=150, c='blue', marker='X', label='Cluster Centers')
            ax.set_xlabel('Sampah Tahunan')
            ax.set_ylabel('Pengurangan')
            ax.set_zlabel('Penanganan')
            ax.set_title('3D Mean Shift Clustering')
            ax.legend()
            st.pyplot(fig)

            if len(set(labels)) > 1:
                dbi = davies_bouldin_score(X, labels)
                sil = silhouette_score(X, labels)
                st.markdown(f"📈 **Davies-Bouldin Index**: `{dbi:.3f}`")
                st.markdown(f"📉 **Silhouette Score**: `{sil:.3f}`")
            else:
                st.warning("Hanya 1 klaster terbentuk, tidak bisa mengevaluasi.")

        if 'cluster_labels' in df.columns:
            for cluster_id in sorted(df['cluster_labels'].unique()):
                st.write(f"🟢 **Data Cluster {cluster_id}**")
                st.dataframe(df[df['cluster_labels'] == cluster_id])
    else:
        st.warning("Silakan unggah data terlebih dahulu.")

# === Visualisasi ===
elif st.session_state.selected_tab == "Visualisasi":
    def visualisasi_page(df, n_clusters):
        st.title("Visualisasi Klaster")
        st.markdown(f"### Jumlah Klaster: {n_clusters}")

        cluster_dfs = {
            label: df[df['cluster_labels'] == label]
            for label in sorted(df['cluster_labels'].unique())
        }

        for label, cluster_df in cluster_dfs.items():
            st.markdown(f"### Klaster {label}")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                    <div style='background-color:#FDAB9E; padding:15px; border-radius:10px; text-align:center;'>
                        <h4>Sampah Tahunan</h4>
                        <p style='font-size:24px; font-weight:bold;'>{cluster_df['sampah_tahunan'].sum():,.0f}</p>
                        <p>ton/tahun</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div style='background-color:#FBF3B9; padding:15px; border-radius:10px; text-align:center;'>
                        <h4>Pengurangan</h4>
                        <p style='font-size:24px; font-weight:bold;'>{cluster_df['pengurangan'].sum():,.0f}</p>
                        <p>ton/tahun</p>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div style='background-color:#FFB433; padding:15px; border-radius:10px; text-align:center;'>
                        <h4>Penanganan</h4>
                        <p style='font-size:24px; font-weight:bold;'>{cluster_df['penanganan'].sum():,.0f}</p>
                        <p>ton/tahun</p>
                    </div>
                """, unsafe_allow_html=True)

            # Membuat dua kolom
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Perbandingan Rata-Rata Persentase Pengurangan dan Penanganan per Klaster")
                avg_df = pd.DataFrame({
                    f"Klaster {label}": cluster_df[['perc_pengurangan', 'perc_penanganan']].mean()
                    for label, cluster_df in cluster_dfs.items()
                })
            
                fig, ax = plt.subplots()
                avg_df.T.plot(kind='bar', ax=ax)
                plt.title("Rata-rata Persentase per Klaster")
                plt.ylabel("Persentase (%)")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Perbandingan Rata-Rata Sampah Harian dan Sampah Tahunan per Klaster")
                avg_df = pd.DataFrame({
                    f"Klaster {label}": cluster_df[['sampah_harian', 'sampah_tahunan']].mean()
                    for label, cluster_df in cluster_dfs.items()
                })
                
                fig, ax = plt.subplots()
                avg_df.T.plot(kind='bar', ax=ax)
                plt.title("Rata-rata Sampah Harian dan Sampah Tahunan per Klaster")
                plt.ylabel("Ton/Tahun")
                plt.xticks(rotation=0)
                st.pyplot(fig)
                
    # Tampilkan DataFrame untuk setiap klaster yang dapat diurutkan
            st.markdown(f"### 📋 Tabel Klaster {label}")
            tabel_klaster = cluster_df[['Kabupaten/Kota', 'sampah_harian', 'sampah_tahunan', 'pengurangan', 'penanganan']]
            st.dataframe(tabel_klaster, use_container_width=True)

    if 'df' in st.session_state and 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df
        n_clusters = len(df['cluster_labels'].unique())
        visualisasi_page(df, n_clusters)
    else:
        st.warning("Silakan jalankan pemodelan terlebih dahulu agar klaster tersedia.")
