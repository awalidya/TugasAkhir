import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from mpl_toolkits.mplot3d import Axes3D

# Menambahkan CSS untuk mengubah warna latar belakang
st.markdown(
    """
    <style>
    .main {
        background-color: #d4edda;  /* Warna hijau muda */
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah")

# Menyusun menu di sidebar
menu = ["Upload Data", "Input Data Manual"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

# Fungsi untuk menangani file dan data manual
def handle_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("Data berhasil diunggah!")
    st.dataframe(df.head())

def handle_manual_data():
    st.subheader("🧾 Input Data Manual")
    
    with st.expander("Tambahkan Data Baru Secara Manual"):
        kabupaten_kota = st.text_input("Kabupaten/Kota")
        provinsi = st.text_input("Provinsi")
        
        manual_data = {}
        for col in numeric_columns:
            manual_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=0.0, step=1.0)
        
        if st.button("Tambahkan Data"):
            new_row = {"kabupaten_kota": kabupaten_kota, "provinsi": provinsi, **manual_data}
            new_df = pd.DataFrame([new_row])

            for col in scaling_columns:
                if col not in new_df.columns:
                    st.error(f"Kolom '{col}' tidak ada dalam inputan.")
                    break
            else:
                scaled_input = scaler.transform(new_df[scaling_columns])
                cluster_label = ms_final.predict(scaled_input)
                st.write(f"Data yang dimasukkan berada pada cluster: **Cluster {cluster_label[0]}**")
                st.write("Data yang dimasukkan:")
                st.dataframe(new_df)

# Memilih menu berdasarkan pilihan pengguna
if choice == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        handle_uploaded_data(uploaded_file)

elif choice == "Input Data Manual":
    handle_manual_data()

# Pastikan st.session_state.df ada sebelum menjalankan operasi lainnya
if 'df' in st.session_state:
    df = st.session_state.df
    # Proses lainnya jika ada data
    st.subheader("🧱 Missing Value Sebelum Penanganan")
    missing_before = df.isnull().sum()
    for col, count in missing_before.items():
        if count > 0:
            st.markdown(f"- **{col}**: {count} missing value")
    if missing_before.sum() == 0:
        st.success("Tidak ada missing value yang terdeteksi.")
