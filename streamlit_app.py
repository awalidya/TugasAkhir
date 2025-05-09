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

# Logo di sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png",
    width=150
)

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

tab = st.sidebar.selectbox("Pilih Menu", ["Halaman Utama", "Upload Data", "Visualisasi"])

if tab == "Halaman Utama":
    st.write("""
    Selamat datang di platform analisis wilayah berbasis pengelolaan sampah...
    """)

elif tab == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df)

        handle_missing_values(df)
        for col in feature_outlier:
            handle_outliers_iqr(df, col)

        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
        X = df[scaling_columns].values
        st.session_state.df = df

        # Load model dan simpan ke session_state
        model_filename = "mean_shift_model_bandwidth_1.5.joblib"
        try:
            ms_final = joblib.load(model_filename)
            st.session_state.ms_final = ms_final
            st.success("Model Mean Shift berhasil dimuat!")
        except Exception as e:
            st.session_state.ms_final = None
            st.error(f"Terjadi kesalahan saat memuat model: {e}")

        # Prediksi klaster
        if st.session_state.ms_final is not None and hasattr(st.session_state.ms_final, 'cluster_centers_'):
            st.session_state.df['cluster_labels'] = st.session_state.ms_final.predict(X)
            st.success("Prediksi cluster selesai!")

elif tab == "Visualisasi":
    if 'df' in st.session_state and 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df.copy()
        ms_final = st.session_state.get('ms_final', None)

        st.subheader("Data Cluster 0 dan Cluster 1")
        cluster_0_df = df[df['cluster_labels'] == 0]
        cluster_1_df = df[df['cluster_labels'] == 1]
        st.write("🔵 **Data Cluster 0**")
        st.dataframe(cluster_0_df)
        st.write("🟠 **Data Cluster 1**")
        st.dataframe(cluster_1_df)

        st.subheader("Statistik Deskriptif Cluster 0 dan Cluster 1")
        st.write("**Cluster 0**")
        st.dataframe(cluster_0_df.describe())
        st.write("**Cluster 1**")
        st.dataframe(cluster_1_df.describe())

        st.subheader("Rata-rata Persentase Pengurangan & Penanganan per Cluster")
        avg_df = pd.DataFrame({
            "Klaster 0": cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean(),
            "Klaster 1": cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()
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

        # Visualisasi 3D
        st.subheader("Visualisasi Klaster 3D")
        if ms_final is not None:
            labels = df['cluster_labels']
            cluster_centers = ms_final.cluster_centers_

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                       c=labels, cmap='plasma', marker='o', label='Data Points')

            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                       s=250, c='blue', marker='X', label='Cluster Centers')

            ax.set_xlabel('Sampah Tahunan')
            ax.set_ylabel('Pengurangan Sampah')
            ax.set_zlabel('Penanganan Sampah')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Model belum dimuat atau tidak ditemukan. Silakan unggah data terlebih dahulu.")
