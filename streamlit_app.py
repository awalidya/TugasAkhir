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

# Kolom-kolom numerik dan lainnya
target_columns = [
    'sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan',
    'penanganan', 'perc_penanganan', 'sampah_terkelola', 'perc_sampah_terkelola', 'daur_ulang'
]
outlier_columns = ['sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan', 'penanganan', 'sampah_terkelola']
scaling_columns = ['sampah_tahunan', 'pengurangan', 'penanganan']

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
input_data = st.sidebar.button("Input Data Manual")

# Input manual
if input_data:
    st.subheader("📥 Input Data Manual")
    with st.form("manual_input_form"):
        kab_kota = st.text_input("Kabupaten/Kota")
        provinsi = st.text_input("Provinsi")
        sampah_harian = st.number_input("Sampah Harian (ton)", min_value=0.0, format="%.2f")
        sampah_tahunan = st.number_input("Sampah Tahunan (ton)", min_value=0.0, format="%.2f")
        pengurangan = st.number_input("Pengurangan (ton)", min_value=0.0, format="%.2f")
        perc_pengurangan = st.number_input("Persentase Pengurangan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        penanganan = st.number_input("Penanganan (ton)", min_value=0.0, format="%.2f")
        perc_penanganan = st.number_input("Persentase Penanganan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        sampah_terkelola = st.number_input("Sampah Terkelola (ton)", min_value=0.0, format="%.2f")
        perc_sampah_terkelola = st.number_input("Persentase Sampah Terkelola (%)", min_value=0.0, max_value=100.0, format="%.2f")
        daur_ulang = st.number_input("Daur Ulang (ton)", min_value=0.0, format="%.2f")
        submit = st.form_submit_button("Simpan dan Proses Data Manual")

        if submit:
            manual_df = pd.DataFrame([{ 
                'Kabupaten/Kota': kab_kota,
                'Provinsi': provinsi,
                'sampah_harian': sampah_harian,
                'sampah_tahunan': sampah_tahunan,
                'pengurangan': pengurangan,
                'perc_pengurangan': perc_pengurangan,
                'penanganan': penanganan,
                'perc_penanganan': perc_penanganan,
                'sampah_terkelola': sampah_terkelola,
                'perc_sampah_terkelola': perc_sampah_terkelola,
                'daur_ulang': daur_ulang
            }])
            st.session_state.df = manual_df
            st.success("Data manual berhasil disimpan!")
            st.dataframe(manual_df)

# Jika file diupload atau input manual tersimpan
if 'df' in st.session_state:
    df = st.session_state.df.copy()

    st.subheader("🧱 Cek dan Tangani Missing Values")
    st.write("Sebelum Penanganan:")
    st.dataframe(df.isnull().sum())
    handle_missing_values(df)
    st.write("Setelah Penanganan:")
    st.dataframe(df.isnull().sum())

    st.subheader("🧹 Plot Outlier Sebelum Penanganan")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, col in enumerate(outlier_columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

    for col in outlier_columns:
        handle_outliers_iqr(df, col)

    st.subheader("📦 Plot Outlier Setelah Penanganan")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, col in enumerate(outlier_columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

    scaler = RobustScaler()
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    st.subheader("📊 Data Setelah Scaling")
    st.dataframe(df[scaling_columns].head())

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("🔥 Korelasi")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[scaling_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🚀 Clustering dengan Mean Shift")
    try:
        model = joblib.load("mean_shift_model_bandwidth_1.5.joblib")
        X = df[scaling_columns].values
        labels = model.predict(X)
        df['cluster'] = labels
        st.success("Prediksi klaster berhasil!")
        st.dataframe(df)

        st.subheader("📊 Visualisasi 3D")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[scaling_columns[0]], df[scaling_columns[1]], df[scaling_columns[2]], c=labels, cmap='plasma')
        ax.set_xlabel(scaling_columns[0])
        ax.set_ylabel(scaling_columns[1])
        ax.set_zlabel(scaling_columns[2])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal melakukan clustering: {e}")
