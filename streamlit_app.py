import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import joblib
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")


# Menyisipkan font kustom dari Google Fonts
st.markdown("""
    <style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #D5E5D5 !important; /* Soft orange */
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



# Sidebar dengan logo dan judul
with st.sidebar:
    col1, col2 = st.columns([4, 3])
    with col1:
        st.image(
            "https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png",
            width=200
        )
    with col2:
        st.markdown("<span class='custom-title move-down'>Trash Achievement</span>", unsafe_allow_html=True)
        
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

# tab = st.sidebar.selectbox("Pilih Menu", ["Halaman Utama", "Upload Data", "Pemodelan", "Visualisasi"])

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Halaman Utama'

# Fungsi untuk mengatur tab yang dipilih
def set_tab(tab_name):
    st.session_state.selected_tab = tab_name

# Sidebar dengan tombol-tombol navigasi
with st.sidebar:
    tabs = ["Halaman Utama", "Upload Data", "Pemodelan", "Visualisasi"]
    for tab in tabs:
        if st.button(tab):
            st.session_state.selected_tab = tab
            
if st.session_state.selected_tab == "Halaman Utama":
# if tab == "Halaman Utama":
    st.write("""
    Selamat datang di platform analisis wilayah berbasis pengelolaan sampah 
    \nMelalui pendekatan data dan metode klastering, kami menyajikan gambaran menyeluruh tentang bagaimana berbagai daerah di Indonesia menangani permasalahan sampah. 
    Dengan memetakan wilayah berdasarkan pola pengurangan, penanganan, dan daur ulang sampah, platform ini diharapkan dapat menjadi acuan bagi pengambil kebijakan, 
    peneliti, maupun masyarakat umum dalam mendorong pengelolaan sampah yang lebih efektif dan berkelanjutan. 
    Mari bersama menciptakan lingkungan yang lebih bersih dan sehat melalui keputusan berbasis data
    """)

    # Jika menu Upload Data dipilih
elif st.session_state.selected_tab == "Upload Data":
# elif tab == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df)

        # Proses pemodelan
        df = st.session_state.df

        # Proses lainnya hanya akan dijalankan jika df ada
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
        for col, count in missing_after.items():
            if count > 0:
                st.markdown(f"- **{col}**: {count} missing value")
        if missing_after.sum() == 0:
            st.success("Semua missing value telah berhasil ditangani!")

        st.subheader("Plot Outlier Sebelum Penanganan")
        # Membuat 2 baris dan 3 kolom untuk 6 boxplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # Ukuran figure yang lebih besar untuk 6 boxplot
        axes = axes.flatten()  # Mempermudah akses ke setiap subplot

        # Iterasi untuk menampilkan boxplot untuk setiap kolom
        for i, col in enumerate(numeric_columns[:6]):  # Mengambil 6 kolom pertama
            sns.boxplot(x=df[col], ax=axes[i])  # Plot boxplot pada subplot yang sesuai
            axes[i].set_title(f"Boxplot {col}")

        # Menampilkan figure
        st.pyplot(fig)

        st.subheader("Plot Outlier Setelah Penanganan")
        # Mengatasi outlier dan plot boxplot setelah penanganan
        for col in feature_outlier[:6]:  # Menyesuaikan agar hanya 6 kolom pertama yang diproses
            handle_outliers_iqr(df, col)

        # Membuat 2 baris dan 3 kolom untuk 6 boxplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # Ukuran figure yang lebih besar untuk 6 boxplot
        axes = axes.flatten()  # Mempermudah akses ke setiap subplot

        # Iterasi untuk menampilkan boxplot untuk setiap kolom
        for i, col in enumerate(feature_outlier[:6]):  # Mengambil 6 kolom pertama
            sns.boxplot(x=df[col], ax=axes[i])  # Plot boxplot pada subplot yang sesuai
            axes[i].set_title(f"Boxplot {col}")

        # Menampilkan figure
        st.pyplot(fig)

        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
        X = df[scaling_columns].values

        st.subheader("Data Setelah Scaling")
        st.dataframe(df[scaling_columns].head())

        st.subheader("EDA")
        st.dataframe(df[scaling_columns].describe().T)

        # Membuat satu figure dengan 3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 baris, 3 kolom untuk subplot
        axes = axes.flatten()  # Flatten untuk memudahkan akses subplot

        # Iterasi untuk menampilkan histogram untuk setiap kolom
        for i, column in enumerate(df[scaling_columns][:3]):  # Mengambil 3 kolom pertama
            sns.histplot(df[column], kde=True, ax=axes[i])  # Plot histogram pada subplot yang sesuai
            axes[i].set_title(f'Histogram of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')

        # Menampilkan figure
        st.pyplot(fig)

        correlation_matrix_selected = df[scaling_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix_selected, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap for Selected Features")
        st.pyplot(fig)

elif st.session_state.selected_tab == "Pemodelan":
# elif tab == "Pemodelan":
    if 'df' in st.session_state:
        df = st.session_state.df.copy()
        X = df[scaling_columns].values

        from sklearn.cluster import MeanShift
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        st.subheader("Pemodelan Clustering dengan Mean Shift")

        # Input bandwidth interaktif
        custom_bw = st.number_input(
            label="🎛️ Sesuaikan nilai Bandwidth",
            min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f"
        )

        if st.button("🚀 Jalankan Clustering"):
            ms = MeanShift(bandwidth=custom_bw, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            n_clusters = len(np.unique(labels))

            # Simpan hasil ke session_state
            df['cluster_labels'] = labels
            st.session_state.df = df
            st.session_state.ms_final = ms

            st.success(f"Jumlah klaster terbentuk: {n_clusters}")

            # Plot 3D
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

            # Evaluasi
            if len(set(labels)) > 1:
                dbi = davies_bouldin_score(X, labels)
                sil = silhouette_score(X, labels)
                st.markdown(f"📈 **Davies-Bouldin Index**: `{dbi:.3f}`")
                st.markdown(f"📉 **Silhouette Score**: `{sil:.3f}`")
            else:
                st.warning("Tidak bisa menghitung DBI/Silhouette karena hanya ada 1 klaster.")
    else:
        st.warning("Silakan unggah data terlebih dahulu.")
        
    if 'df' in st.session_state:
        df = st.session_state.df.copy()
       
        if 'cluster_labels' not in df.columns:
            st.warning("Proses clustering belum dilakukan atau kolom 'cluster_labels' tidak ada.")
        else: 
            # Proses clustering sudah berhasil, bisa lanjut ke analisis klaster
            cluster_0_df = df[df['cluster_labels'] == 0]
            cluster_1_df = df[df['cluster_labels'] == 1]
            st.write("🔵 **Data Cluster 0**")
            st.dataframe(cluster_0_df)       
            st.write("🟠 **Data Cluster 1**")
            st.dataframe(cluster_1_df)
               
    else:
        st.warning("Silakan unggah data terlebih dahulu.")
 
elif st.session_state.selected_tab == "Visualisasi":
def visualisasi_page(df, n_clusters):
    st.title("Visualisasi Klaster")

    if df is not None and 'cluster_labels' in df.columns:
        # Tampilkan jumlah klaster
        st.markdown(f"### Jumlah Klaster: {n_clusters}")

        # Buat DataFrame per klaster secara dinamis
        cluster_dfs = {
            label: df[df['cluster_labels'] == label]
            for label in sorted(df['cluster_labels'].unique())
        }

        # Visualisasi statistik utama per klaster
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

        # Bar chart perbandingan rata-rata persentase
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

        # Scatter plot berdasarkan klaster
        st.markdown("### Scatter Plot Pengurangan vs Penanganan")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x='perc_pengurangan',
            y='perc_penanganan',
            hue='cluster_labels',
            palette='Set2',
            s=100
        )
        plt.xlabel("Persentase Pengurangan (%)")
        plt.ylabel("Persentase Penanganan (%)")
        plt.title("Klastering Wilayah")
        st.pyplot(fig)

    else:
        st.warning("Data belum tersedia atau belum dilakukan klastering.")

