import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Menambahkan logo di atas menu sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png",  # Memperbaiki tanda kutip yang hilang
    width=150  # ubah angkanya sesuai kebutuhan (misal: 100, 200, dll)
)


# Kolom lainnya tetap seperti semula
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

def jumlah_outlier(df, kolom):
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[kolom] < lower) | (df[kolom] > upper)].shape[0]

def persen_outlier(df, kolom):
    jumlah = jumlah_outlier(df, kolom)
    return (jumlah / df.shape[0]) * 100

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

# ✅ Perubahan: Hapus tombol upload, langsung tampilkan file uploader
halaman_utama = st.sidebar.button("Halaman Utama")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
visualisasi = st.sidebar.button("Visualisasi")

if halaman_utama:
    st.write(
        """
        Selamat datang di platform analisis wilayah berbasis pengelolaan sampah
        Melalui pendekatan data dan metode klastering, kami menyajikan gambaran menyeluruh tentang bagaimana berbagai daerah di Indonesia menangani permasalahan sampah. 
        Dengan memetakan wilayah berdasarkan pola pengurangan, penanganan, dan daur ulang sampah, platform ini diharapkan dapat menjadi acuan bagi pengambil kebijakan, 
        peneliti, maupun masyarakat umum dalam mendorong pengelolaan sampah yang lebih efektif dan berkelanjutan. 
        Mari bersama menciptakan lingkungan yang lebih bersih dan sehat melalui keputusan berbasis data.
        """
    )    

# Menampilkan teks drag file jika belum ada file yang diupload
if not uploaded_file:
    with st.sidebar.empty():
        st.write("👈 Drag file here to upload")
        
# ✅ Perubahan: Jalankan proses jika file sudah diupload
if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data berhasil diunggah!")
    st.dataframe(df.head())

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

    model_filename = "mean_shift_model_bandwidth_1.5.joblib"
    ms_final = joblib.load(model_filename)
    st.success("Model Mean Shift berhasil dimuat!")

    st.session_state.df['cluster_labels'] = ms_final.predict(X)
    st.success("Prediksi cluster selesai!")

if visualisasi:
    if 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df.copy()

        st.subheader("Data Cluster 0 dan Cluster 1")
        cluster_0_df = df[df['cluster_labels'] == 0]
        cluster_1_df = df[df['cluster_labels'] == 1]

        st.write("🔵 **Data Cluster 0**")
        st.dataframe(cluster_0_df)

        st.write("🟠 **Data Cluster 1**")
        st.dataframe(cluster_1_df)

        st.subheader("Statistik Deskriptif Cluster 0 dan Cluster 1")
        st.write("**Statistik Deskriptif Cluster 0**")
        st.dataframe(cluster_0_df.describe())

        st.write("**Statistik Deskriptif Cluster 1**")
        st.dataframe(cluster_1_df.describe())

        st.subheader("Rata-rata Persentase Pengurangan & Penanganan per Cluster")
        cluster_0_avg = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
        cluster_1_avg = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()
        avg_df = pd.DataFrame({"Klaster 0": cluster_0_avg, "Klaster 1": cluster_1_avg})

        fig, ax = plt.subplots(figsize=(8, 5))
        avg_df.T.plot(kind='bar', ax=ax, color=['blue', 'orange'])
        for i, cluster in enumerate(avg_df.columns):
            for j, val in enumerate(avg_df[cluster]):
                ax.text(i + j*0.25 - 0.15, val + 0.5, f"{val:.2f}", ha='center', fontsize=10)
        ax.set_title("Rata-rata Persentase Pengurangan dan Penanganan")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Rata-rata Persentase")
        st.pyplot(fig)

        st.subheader("Visualisasi Klaster 3D")
        labels = df['cluster_labels']
        cluster_centers = ms_final.cluster_centers_

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot data points
        ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                c=labels, cmap='plasma', marker='o', label='Data Points')

        # Plot cluster centers
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                s=250, c='blue', marker='X', label='Cluster Centers')

        # Set axis labels
        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan Sampah')
        ax.set_zlabel('Penanganan Sampah')

        # Menambahkan legenda
        ax.legend()

        # Menampilkan grafik 3D
        st.pyplot(fig)
