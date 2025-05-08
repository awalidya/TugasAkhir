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

# ✅ Perubahan: Hapus tombol upload, langsung tampilkan file uploader
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

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

    # Lanjutkan proses lainnya seperti analisis, visualisasi, dan clustering sesuai kebutuhan

    # Proses Model
    model_filename = "mean_shift_model_bandwidth_1.5.joblib"
    ms_final = joblib.load(model_filename)
    st.success("Model Mean Shift berhasil dimuat!")

    # Melakukan scaling pada data
    X = df[scaling_columns].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Simpan scaler di session_state
    st.session_state.scaler = scaler
    st.session_state.df['cluster_labels'] = ms_final.predict(X_scaled)
    st.success("Prediksi cluster selesai!")

    # Tampilkan hasil clustering di bawah
    if 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df.copy()

        st.subheader("Data Cluster 0 dan Cluster 1")
        cluster_0_df = df[df['cluster_labels'] == 0]
        cluster_1_df = df[df['cluster_labels'] == 1]

        st.write("🔵 **Data Cluster 0**")
        st.dataframe(cluster_0_df)

        st.write("🟠 **Data Cluster 1**")
        st.dataframe(cluster_1_df)

        # Rata-rata untuk visualisasi
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

        # Visualisasi 3D clustering
        st.subheader("Visualisasi 3D Hasil Clustering")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot titik data berdasarkan hasil clustering
        ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                   c=df['cluster_labels'], cmap='plasma', marker='o', label='Data Points')

        # Plot pusat klaster (asumsi Anda memiliki data cluster_centers)
        cluster_centers = ms_final.cluster_centers_  # Asumsi center didapat dari model Mean Shift
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


# Inisialisasi session state
if 'input_manual' not in st.session_state:
    st.session_state.input_manual = False

# Tombol toggle
if not st.session_state.input_manual:
    if st.sidebar.button("Input Data Manual"):
        st.session_state.input_manual = True
else:
    if st.sidebar.button("Tutup Input Manual"):
        st.session_state.input_manual = False

# Form input manual
if st.session_state.input_manual:
    st.subheader("📝 Input Data Manual (tanpa Sampah Harian)")
    sampah_tahunan = st.number_input("Sampah Tahunan", min_value=0.0)
    pengurangan = st.number_input("Pengurangan", min_value=0.0)
    penanganan = st.number_input("Penanganan", min_value=0.0)

    if st.button("Proses Data Manual"):
        input_data = np.array([[sampah_tahunan, pengurangan, penanganan]])
        st.write("📊 Data yang dimasukkan:")
        st.write(pd.DataFrame(input_data, columns=["Sampah Tahunan", "Pengurangan", "Penanganan"]))

        try:
            scaler = st.session_state.scaler  # Ambil dari session state
            input_scaled = scaler.transform(input_data)

            cluster_label = ms_final.predict(input_scaled)
            st.success(f"✅ Data dimasukkan ke dalam Klaster: {cluster_label[0]}")
        except KeyError:
            st.error("❌ Scaler belum tersedia. Silakan unggah data terlebih dahulu.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
