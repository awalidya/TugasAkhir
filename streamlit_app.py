import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Kolom numerik yang akan digunakan
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

# Menangani missing value secara umum
def handle_missing_values(df):
    # Mengganti missing values dengan median (untuk kolom numerik)
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data berhasil diunggah!")
    st.dataframe(df.head())

    # 1. Menampilkan Missing Value sebelum penanganan
    st.subheader("Missing Value Sebelum Penanganan")
    missing_values_before = df.isnull().sum()
    st.write(missing_values_before)

    # 2. Menangani missing values secara keseluruhan
    handle_missing_values(df)
    st.session_state.df = df

    # 2. Menampilkan Missing Value setelah penanganan
    st.subheader("Missing Value Setelah Penanganan")
    missing_values_after = df.isnull().sum()
    st.write(missing_values_after)

    # 3. Plot Outlier sebelum penanganan
    st.subheader("Plot Outlier Sebelum Penanganan")
    for col in numeric_columns:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot {col}")
        st.pyplot(fig)

    # 4. Plot Outlier setelah penanganan
    st.subheader("Plot Outlier Setelah Penanganan")
    for col in feature_outlier:
        handle_outliers_iqr(df, col)

    for col in feature_outlier:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot {col}")
        st.pyplot(fig)

    # 5. Scaling Columns
    scaler = RobustScaler()
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    X = df[scaling_columns].values  # Menggunakan values agar menjadi array NumPy

    st.subheader("Data Setelah Scaling")
    st.dataframe(df[scaling_columns].head())

    # 6. Bandwidth Tuning dan Evaluasi Mean Shift
    bandwidths = [1.0, 1.5, 2.0]
    for bw in bandwidths:
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        centers = ms.cluster_centers_

        st.write(f"Bandwidth = {bw}, Jumlah cluster = {len(np.unique(labels))}")
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', marker='p')  # Memperbaiki scatter plot
        ax.scatter(centers[:, 0], centers[:, 1], s=250, c='blue', marker='X')
        ax.set_title(f'Mean Shift Clustering (Bandwidth = {bw})')
        st.pyplot(fig)

        if len(set(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            st.write(f"Silhouette Score: {sil_score:.3f}")
        else:
            st.write("Silhouette Score tidak dapat dihitung karena hanya 1 cluster.")

    # Final clustering dengan bandwidth terbaik
    ms_final = MeanShift(bandwidth=1.5, bin_seeding=True)
    ms_final.fit(X)
    st.session_state.df['cluster_labels'] = ms_final.labels_
    st.success("Clustering selesai menggunakan bandwidth 1.5")

    # Hasil Klastering
    if 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df.copy()

        st.subheader("📋 Data Cluster 0 dan Cluster 1")

        cluster_0_df = df[df['cluster_labels'] == 0]
        cluster_1_df = df[df['cluster_labels'] == 1]

        st.write("🔵 **Data Cluster 0**")
        st.dataframe(cluster_0_df)

        st.write("🟠 **Data Cluster 1**")
        st.dataframe(cluster_1_df)

        st.subheader("📊 Statistik Deskriptif Cluster 0 dan Cluster 1")

        st.write("🔵 **Statistik Deskriptif Cluster 0**")
        st.dataframe(cluster_0_df.describe())

        st.write("🟠 **Statistik Deskriptif Cluster 1**")
        st.dataframe(cluster_1_df.describe())

        st.subheader("📈 Rata-rata Persentase Pengurangan & Penanganan per Cluster")

        cluster_0_avg = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
        cluster_1_avg = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()

        avg_df = pd.DataFrame({
            "Klaster 0": cluster_0_avg,
            "Klaster 1": cluster_1_avg
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        avg_df.T.plot(kind='bar', ax=ax, color=['blue', 'orange'])

        for i, cluster in enumerate(avg_df.columns):
            for j, val in enumerate(avg_df[cluster]):
                ax.text(i + j*0.25 - 0.15, val + 0.5, f"{val:.2f}", ha='center', fontsize=10)

        ax.set_title("Rata-rata Persentase Pengurangan dan Penanganan")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Rata-rata Persentase")
        st.pyplot(fig)
