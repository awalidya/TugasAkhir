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

# Membuat navigasi sidebar
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Upload Data", "Input Data Manual"]
)

if menu == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df.head())

        # Menangani missing values
        handle_missing_values(df)
        st.session_state.df = df
        
        st.subheader("🧹 Missing Value Setelah Penanganan")
        missing_after = df.isnull().sum()
        for col, count in missing_after.items():
            if count > 0:
                st.markdown(f"- **{col}**: {count} missing value")
        if missing_after.sum() == 0:
            st.success("Semua missing value telah berhasil ditangani!")
        
        # Menangani outliers
        st.subheader("Plot Outlier Sebelum Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        
        st.pyplot(fig)
        
        for col in feature_outlier[:6]:
            handle_outliers_iqr(df, col)
        
        st.subheader("Plot Outlier Setelah Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_outlier[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        
        st.pyplot(fig)
        
        # Scaling data
        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
        X = df[scaling_columns].values
        
        st.subheader("Data Setelah Scaling")
        st.dataframe(df[scaling_columns].head())
        
        # Menampilkan EDA
        st.subheader("EDA")
        st.dataframe(df[scaling_columns].describe().T)

        # Heatmap correlation
        correlation_matrix_selected = df[scaling_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix_selected, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap for Selected Features")
        st.pyplot(fig)

        # Memuat model Mean Shift
        model_filename = "mean_shift_model_bandwidth_1.5.joblib"
        ms_final = joblib.load(model_filename)
        st.success("Model Mean Shift berhasil dimuat!")

        # Prediksi cluster
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

elif menu == "Input Data Manual":
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
                scaled_input = scaler.transform(new_df[scaling
