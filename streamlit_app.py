import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Kolom input manual
numeric_columns = [
    'sampah_harian', 'sampah_tahunan', 'pengurangan', 'perc_pengurangan',
    'penanganan', 'perc_penanganan', 'sampah_terkelola', 'perc_sampah_terkelola', 'daur_ulang'
]
scaling_columns = ['sampah_tahunan', 'pengurangan', 'penanganan']

@st.cache_data
def load_model():
    return joblib.load("mean_shift_model_bandwidth_1.5.joblib")

# Menu Sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Upload Data CSV", "Input Data Manual"])

if menu == "Upload Data CSV":
    st.subheader("Upload Data CSV")
    uploaded_file = st.file_uploader("Pilih File CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diupload:", data.head())
        
        # Lakukan preprocessing jika diperlukan di sini
        st.subheader("Proses Klastering")
        scaler = RobustScaler()
        data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
        
        # Load model
        model = load_model()
        
        # Prediksi klaster
        labels = model.predict(data[scaling_columns])
        
        # Tampilkan hasil klastering
        st.subheader("Hasil Klastering")
        data['Cluster'] = labels
        st.write(data)
        
        # Visualisasi Klastering
        st.subheader("Visualisasi Klaster 3D")
        cluster_centers = model.cluster_centers_
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data['sampah_tahunan'], data['pengurangan'], data['penanganan'],
                   c=labels, cmap='plasma', marker='o', label='Data Points')

        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   s=250, c='blue', marker='X', label='Cluster Centers')

        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan Sampah')
        ax.set_zlabel('Penanganan Sampah')

        ax.legend()
        st.pyplot(fig)

elif menu == "Input Data Manual":
    st.subheader("Masukkan Data Secara Manual")
    
    # Input manual
    sampah_harian = st.number_input("Sampah Harian", min_value=0.0)
    sampah_tahunan = st.number_input("Sampah Tahunan", min_value=0.0)
    pengurangan = st.number_input("Pengurangan Sampah", min_value=0.0)
    perc_pengurangan = st.number_input("Persentase Pengurangan", min_value=0.0, max_value=100.0)
    penanganan = st.number_input("Penanganan Sampah", min_value=0.0)
    perc_penanganan = st.number_input("Persentase Penanganan", min_value=0.0, max_value=100.0)
    sampah_terkelola = st.number_input("Sampah Terkelola", min_value=0.0)
    perc_sampah_terkelola = st.number_input("Persentase Sampah Terkelola", min_value=0.0, max_value=100.0)
    daur_ulang = st.number_input("Daur Ulang", min_value=0.0)
    
    # Simpan inputan manual ke dalam DataFrame
    input_data = {
        'sampah_harian': [sampah_harian],
        'sampah_tahunan': [sampah_tahunan],
        'pengurangan': [pengurangan],
        'perc_pengurangan': [perc_pengurangan],
        'penanganan': [penanganan],
        'perc_penanganan': [perc_penanganan],
        'sampah_terkelola': [sampah_terkelola],
        'perc_sampah_terkelola': [perc_sampah_terkelola],
        'daur_ulang': [daur_ulang]
    }
    
    input_df = pd.DataFrame(input_data)

    if st.button("Proses Data"):
        # Lakukan scaling pada data input
        scaler = RobustScaler()
        input_df[scaling_columns] = scaler.fit_transform(input_df[scaling_columns])
        
        # Load model
        model = load_model()
        
        # Melakukan prediksi klaster
        labels = model.predict(input_df[scaling_columns])
        
        # Tampilkan hasil klastering
        st.subheader("Hasil Klastering Kota yang Dimasukkan")
        if labels[0] == 0:
            st.success("Kota ini masuk ke dalam Klaster 0.")
        else:
            st.success("Kota ini masuk ke dalam Klaster 1.")
        
        # Tampilkan data input dengan klaster
        input_df['Cluster'] = labels
        st.write(input_df)

        # Visualisasi Klastering
        st.subheader("Visualisasi Klaster 3D")
        cluster_centers = model.cluster_centers_
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(input_df['sampah_tahunan'], input_df['pengurangan'], input_df['penanganan'],
                   c=labels, cmap='plasma', marker='o', label='Data Points')

        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   s=250, c='blue', marker='X', label='Cluster Centers')

        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan Sampah')
        ax.set_zlabel('Penanganan Sampah')

        ax.legend()
        st.pyplot(fig)
