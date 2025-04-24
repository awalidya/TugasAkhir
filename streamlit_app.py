import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

# Set page config for wide layout
st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Menampilkan sidebar untuk file upload
st.sidebar.title('Upload Data')
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Jika file diupload
if uploaded_file is not None:
    # Membaca file CSV ke dataframe
    df = pd.read_csv(uploaded_file)

    # 1. MISSING VALUE
    df.isnull().sum()

    # Daftar kolom yang akan diperbaiki
    kolom_ubah = ['daur_ulang', 'pengurangan', 'perc_pengurangan', 'penanganan', 'perc_penanganan', 'sampah_terkelola', 'perc_sampah_terkelola']

    # Loop untuk mengganti 0.0 dan NaN dengan median masing-masing kolom
    for kolom in kolom_ubah:
        median_value = df[kolom].replace(0.0, np.nan).median()  # Hitung median tanpa 0.0
        df[kolom] = df[kolom].replace(0.0, np.nan).fillna(median_value)

    df.isnull().sum()

    # 2. OUTLIER
    # Menghitung jumlah outlier menggunakan metode IQR
    def jumlah_outlier(df, kolom):
        quartile_1 = df[kolom].quantile(0.25)
        quartile_3 = df[kolom].quantile(0.75)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - 1.5 * iqr
        upper_bound = quartile_3 + 1.5 * iqr

        outliers = df[(df[kolom] < lower_bound) | (df[kolom] > upper_bound)]
        return outliers.shape[0]

    # Menghitung persentase jumlah outlier menggunakan metode IQR
    def persen_outlier(df, kolom):
        quartile_1 = df[kolom].quantile(0.25)
        quartile_3 = df[kolom].quantile(0.75)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - 1.5 * iqr
        upper_bound = quartile_3 + 1.5 * iqr

        outliers = df[(df[kolom] < lower_bound) | (df[kolom] > upper_bound)]
        persentase_outlier = (outliers.shape[0] / df.shape[0]) * 100
        return persentase_outlier

    # Membuat gambar terpisah untuk setiap boxplot
    fig, axs = plt.subplots(nrows=len(kolom_ubah), figsize=(10, 15))

    # Membuat boxplot untuk setiap kolom dan menambahkan ke subplot yang sesuai
    for i, column in enumerate(kolom_ubah):
        sns.boxplot(x=df[column], ax=axs[i])
        axs[i].set_title(f'Boxplot of {column}')
        axs[i].set_xlabel(column)

    plt.tight_layout()
    st.pyplot(fig)

    # Kolom yang memiliki outlier berdasarkan hasil visualisasi boxplot dari kolom tersebut
    feature_outlier = ['sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan', 'penanganan', 'sampah_terkelola']

    # Menampilkan jumlah serta persentase outlier pada tabel user
    st.write("Jumlah outlier tabel")
    for feature in feature_outlier:
        persentase_outlier = persen_outlier(df, feature)  # Menghitung persentase jumlah outlier
        jumlah_outliers = jumlah_outlier(df, feature)  # Menghitung jumlah outlier
        formatted_persentase_outlier = "{:.2f}".format(persentase_outlier)
        st.write(f'{feature} : {jumlah_outliers} : {formatted_persentase_outlier}%')

    # 3. HANDLING OUTLIER
    # Handling outliers menggunakan metode IQR
    def handle_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Menentukan batas bawah dan batas atas
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Mengganti nilai outlier dengan nilai batas atas atau batas bawah terdekat
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    # Handling outlier pada kolom yang memiliki nilai outlier
    for column in feature_outlier:
        handle_outliers_iqr(df, column)

    # Menampilkan jumlah serta persentase outlier pada tabel user setelah penanganan
    st.write("Jumlah outlier setelah penanganan")
    for feature in feature_outlier:
        persentase_outlier = persen_outlier(df, feature)  # Menghitung persentase jumlah outlier
        jumlah_outliers = jumlah_outlier(df, feature)  # Menghitung jumlah outlier
        formatted_persentase_outlier = "{:.2f}".format(persentase_outlier)
        st.write(f'{feature} : {jumlah_outliers} : {formatted_persentase_outlier}%')

    # Membuat gambar terpisah untuk setiap boxplot feature yang sebelumnya memiliki outlier
    fig, axs = plt.subplots(nrows=len(feature_outlier), figsize=(10, 15))

    # Membuat boxplot untuk setiap kolom dan menambahkan ke subplot yang sesuai
    for i, column in enumerate(feature_outlier):
        sns.boxplot(x=df[column], ax=axs[i])
        axs[i].set_title(f'Boxplot of {column}')
        axs[i].set_xlabel(column)

    plt.tight_layout()
    st.pyplot(fig)

    # 4. FEATURE SCALING
    scaling_columns = ['sampah_tahunan', 'pengurangan', 'penanganan']

    # Inisialisasi RobustScaler
    scaler = RobustScaler()

    # Transformasi data menggunakan RobustScaler
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    # 5. EDA
    st.write(df[scaling_columns].describe().T)

    # Visualisasi distribusi fitur
    for column in df[scaling_columns]:
        st.subheader(f'Histogram of {column}')
        fig = plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        st.pyplot(fig)

    # Hitung matriks korelasi untuk kolom yang ada dalam scaling_columns
    correlation_matrix_selected = df[scaling_columns].corr()

    # Plot heatmap untuk korelasi fitur yang dipilih
    st.subheader("Correlation Heatmap for Selected Features")
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_selected, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

    # 6. PEMODELAN
    X = df[['sampah_tahunan', 'pengurangan', 'penanganan']]

    # Load model Mean Shift yang telah disimpan
    ms = joblib.load('mean_shift_model_bandwidth_1.5.joblib')

    # Prediksi cluster menggunakan model
    cluster_labels = ms.predict(X)

    # Menambahkan hasil clustering ke dataframe
    df['cluster_labels'] = cluster_labels

    # 7. VISUALISASI
    # 3D Visualisasi hasil clustering
    from mpl_toolkits.mplot3d import Axes3D

    # Membuat figure dan subplot 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot titik data berdasarkan hasil clustering
    ax.scatter(X['sampah_tahunan'], X['pengurangan'], X['penanganan'],
               c=cluster_labels, cmap='plasma', marker='o', label='Data Points')

    # Plot pusat klaster
    ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], ms.cluster_centers_[:, 2],
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

    # 8. DESKRIPSI PER KLASTER
    # Filter data berdasarkan cluster labels
    cluster_0_df = df[df['cluster_labels'] == 0]
    cluster_1_df = df[df['cluster_labels'] == 1]

    # Menampilkan deskripsi untuk masing-masing cluster
    st.subheader("Deskripsi Cluster 0")
    st.write(cluster_0_df.describe())

    st.subheader("Deskripsi Cluster 1")
    st.write(cluster_1_df.describe())

    # Rata-rata persentase pengurangan dan penanganan per klaster
    avg_cluster_0 = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
    avg_cluster_1 = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()

    # Menampilkan hasil rata-rata
    st.subheader("Rata-rata Cluster 0")
    st.write(avg_cluster_0)

    st.subheader("Rata-rata Cluster 1")
    st.write(avg_cluster_1)

    # Membuat dataframe baru dengan rata-rata dari masing-masing cluster
    avg_df = pd.DataFrame({
        "Klaster 1": avg_cluster_0,
        "Klaster 2": avg_cluster_1
    })

    # Plot bar chart
    st.subheader("Rata-rata Persentase Pengurangan dan Penanganan per Klaster")
    fig = avg_df.T.plot(kind='bar', figsize=(8, 5), color=['blue', 'orange'])
    plt.xlabel("Klaster")
    plt.ylabel("Rata-rata Persentase")
    plt.title("Rata-rata Persentase Pengurangan dan Penanganan per Klaster")
    plt.legend(["Persentase Pengurangan", "Persentase Penanganan"])

    # Menambahkan nilai di atas batang
    for i, cluster in enumerate(avg_df.columns):
        for j, val in enumerate(avg_df[cluster]):
            plt.text(i + j*0.2 - 0.1, val + 0.5, round(val, 2), ha='center', fontsize=10)

    st.pyplot(fig)
