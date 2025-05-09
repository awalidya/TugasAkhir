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

# Menambahkan logo di atas menu sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png", 
    width=150
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

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

# Membuat dua tab
tab1, tab2 = st.tabs(["Pengolahan Data", "Visualisasi"])

with tab1:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df.head())

with tab2:
    st.subheader("🧾 Input Data Manual")
    
    with st.expander("Tambahkan Data Baru Secara Manual"):
        kabupaten_kota = st.text_input("Kabupaten/Kota")
        provinsi = st.text_input("Provinsi")
        
        # Input data numerik
        manual_data = {}
        for col in numeric_columns:
            manual_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=0.0, step=1.0)
        
        if st.button("Tambahkan Data"):
            # Membuat dataframe dari inputan
            new_row = {"kabupaten_kota": kabupaten_kota, "provinsi": provinsi, **manual_data}
            new_df = pd.DataFrame([new_row])

            # Pastikan semua kolom scaling ada di dataframe
            for col in scaling_columns:
                if col not in new_df.columns:
                    st.error(f"Kolom '{col}' tidak ada dalam inputan.")
                    break
            else:
                # Menggunakan transform untuk mengubah data baru sesuai dengan skala yang ada
                scaled_input = scaler.transform(new_df[scaling_columns])

                # Prediksi cluster untuk data yang baru
                cluster_label = ms_final.predict(scaled_input)

                # Tampilkan hasil cluster
                st.write(f"Data yang dimasukkan berada pada cluster: **Cluster {cluster_label[0]}**")
                
                # Menampilkan data yang dimasukkan
                st.write("Data yang dimasukkan:")
                st.dataframe(new_df)

# Menambahkan pemeriksaan untuk memastikan `st.session_state.df` ada sebelum menjalankan operasi lainnya
if 'df' in st.session_state:
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

with tab2:
    st.subheader("Visualisasi")

    if 'cluster_labels' not in st.session_state.df.columns:
        st.warning("Data belum diproses untuk clustering. Silakan jalankan proses klasterisasi terlebih dahulu.")
    else:
        df = st.session_state.df.copy()

        st.subheader("Data per Cluster")
        cluster_labels = df['cluster_labels'].unique()
        for label in sorted(cluster_labels):
            st.write(f"🔹 **Data Cluster {label}**")
            st.dataframe(df[df['cluster_labels'] == label])

        st.subheader("Rata-rata Persentase Pengurangan & Penanganan per Cluster")
        avg_df = df.groupby('cluster_labels')[['perc_pengurangan', 'perc_penanganan']].mean().T

        fig, ax = plt.subplots(figsize=(8, 5))
        avg_df.plot(kind='bar', ax=ax, color=['blue', 'orange'])
        for i, cluster in enumerate(avg_df.columns):
            for j, val in enumerate(avg_df[cluster]):
                ax.text(i + j*0.25 - 0.15, val + 0.5, f"{val:.2f}", ha='center', fontsize=10)
        ax.set_title("Rata-rata Persentase Pengurangan dan Penanganan per Klaster")
        ax.set_xlabel("Variabel")
        ax.set_ylabel("Rata-rata Persentase")
        st.pyplot(fig)

        st.subheader("Visualisasi 3D Hasil Clustering")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                   c=df['cluster_labels'], cmap='plasma', marker='o')

        if hasattr(st.session_state, 'model') and hasattr(st.session_state.model, 'cluster_centers_'):
            cluster_centers = st.session_state.model.cluster_centers_
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                       s=250, c='blue', marker='X', label='Cluster Centers')

        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan')
        ax.set_zlabel('Penanganan')
        ax.set_title('3D Mean Shift Clustering')
        ax.legend()
        st.pyplot(fig)




