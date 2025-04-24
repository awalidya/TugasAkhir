import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Menampilkan sidebar untuk file upload
st.sidebar.title('Upload Data')
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Menu input untuk memilih fitur
st.sidebar.subheader("Pilih Fitur untuk Analisis")
sampah_tahunan = st.sidebar.number_input("Masukkan nilai Sampah Tahunan", min_value=0, value=1000)
pengurangan = st.sidebar.number_input("Masukkan nilai Pengurangan", min_value=0, value=200)
penanganan = st.sidebar.number_input("Masukkan nilai Penanganan", min_value=0, value=150)

# Fungsi untuk preprocessing data (mengganti missing values, outlier handling, dan scaling)
def preprocess_data(df):
    # Menangani missing values
    kolom_ubah = ['daur_ulang', 'pengurangan', 'perc_pengurangan', 'penanganan', 'perc_penanganan', 'sampah_terkelola', 'perc_sampah_terkelola']
    for kolom in kolom_ubah:
        median_value = df[kolom].replace(0.0, np.nan).median()
        df[kolom] = df[kolom].replace(0.0, np.nan).fillna(median_value)
    
    # Handling Outliers menggunakan metode IQR
    def handle_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    feature_outlier = ['sampah_tahunan', 'pengurangan', 'penanganan']
    for column in feature_outlier:
        handle_outliers_iqr(df, column)

    # Scaling data dengan RobustScaler
    scaler = RobustScaler()
    scaling_columns = ['sampah_tahunan', 'pengurangan', 'penanganan']
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    
    return df

if uploaded_file is not None:
    # Membaca data
    df = pd.read_csv(uploaded_file)
    
    # Menampilkan data preview
    st.write("Data Preview")
    st.write(df.head())
    
    # Preprocessing data yang diunggah
    df = preprocess_data(df)

    # Menyaring data untuk fitur yang ingin digunakan dalam clustering
    X = df[['sampah_tahunan', 'pengurangan', 'penanganan']]

    # Load model Mean Shift
    ms = joblib.load('mean_shift_model_bandwidth_1.5.joblib')

    # Prediksi cluster menggunakan model
    cluster_labels = ms.predict(X)

    # Menambahkan hasil clustering ke dataframe
    df['cluster_labels'] = cluster_labels

    # Menampilkan hasil clustering
    st.write("Hasil Clustering")
    st.write(df[['sampah_tahunan', 'pengurangan', 'penanganan', 'cluster_labels']].head())

    # Visualisasi 3D
    st.subheader("Visualisasi 3D Mean Shift Clustering")

    # Membuat figure dan subplot 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot titik data berdasarkan hasil clustering
    ax.scatter(X['sampah_tahunan'], X['pengurangan'], X['penanganan'],
               c=cluster_labels, cmap='plasma', marker='o', label='Data Points')

    # Plot pusat klaster
    cluster_centers = ms.cluster_centers_
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
               s=250, c='blue', marker='X', label='Cluster Centers')

    # Menambahkan label sumbu
    ax.set_xlabel('Sampah Tahunan')
    ax.set_ylabel('Pengurangan')
    ax.set_zlabel('Penanganan')

    # Menambahkan judul dan legenda
    plt.title('3D Mean Shift Clustering')
    ax.legend()

    # Menampilkan plot
    st.pyplot(fig)

    # Filter berdasarkan klaster
    cluster_0_df = df[df['cluster_labels'] == 0]
    cluster_1_df = df[df['cluster_labels'] == 1]

    # Menampilkan statistik deskriptif untuk masing-masing cluster
    st.write("Descriptive Statistics for Cluster 0")
    st.write(cluster_0_df.describe())

    st.write("Descriptive Statistics for Cluster 1")
    st.write(cluster_1_df.describe())

    # Rata-rata fitur untuk setiap cluster
    avg_cluster_0 = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
    avg_cluster_1 = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()

    # Menampilkan rata-rata untuk masing-masing cluster
    st.write("Rata-rata untuk Cluster 0:")
    st.write(avg_cluster_0)

    st.write("Rata-rata untuk Cluster 1:")
    st.write(avg_cluster_1)

    # Membuat dataframe dengan rata-rata setiap cluster
    avg_df = pd.DataFrame({
        "Klaster 1": avg_cluster_0,
        "Klaster 2": avg_cluster_1
    })

    # Menampilkan rata-rata dalam tabel
    st.write("Rata-rata Fitur untuk Masing-masing Cluster")
    st.write(avg_df)

# Menambahkan fungsi inputan untuk user
st.sidebar.subheader("Input Data untuk Prediksi")
input_data = {
    'sampah_tahunan': sampah_tahunan,
    'pengurangan': pengurangan,
    'penanganan': penanganan
}

# Menampilkan inputan yang dimasukkan oleh user
st.write("Input Data yang Dimasukkan:")
st.write(input_data)

# Jika tombol prediksi diklik
if st.sidebar.button('Prediksi'):
    # Membuat dataframe untuk inputan user
    input_df = pd.DataFrame([input_data])

    # Preprocessing inputan user (tanpa menampilkan detail preprocessing)
    input_df = preprocess_data(input_df)

    # Prediksi cluster menggunakan model
    prediksi_cluster = ms.predict(input_df[['sampah_tahunan', 'pengurangan', 'penanganan']])
    st.write(f"Cluster Prediksi: {prediksi_cluster[0]}")


