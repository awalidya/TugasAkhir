import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Aplikasi Pengelompokan Wilayah", layout="wide")
st.title("📊 Aplikasi Pengelompokan Wilayah Berdasarkan Pengelolaan Sampah")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigasi", [
    "Upload Data", "Input Data Manual", "Pengolahan Data", 
    "EDA & Scaling", "Clustering & Visualisasi"])

# Fungsi Load Data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Upload Data
if menu == "Upload Data":
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")
        st.dataframe(df.head())

# Input Data Manual
elif menu == "Input Data Manual":
    st.header("🧾 Input Data Manual")
    if 'df' not in st.session_state:
        st.warning("Silakan upload atau input data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()
        new_data = {}
        for col in df.columns:
            new_data[col] = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
        if st.button("Tambah Data"):
            df = df.append(new_data, ignore_index=True)
            st.session_state.df = df
            st.success("Data berhasil ditambahkan!")
            st.dataframe(df.tail())

# Pengolahan Data: Missing Values dan Outlier
elif menu == "Pengolahan Data":
    st.header("🧼 Pengolahan Data")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()

        # Missing Values
        st.subheader("🔍 Missing Values")
        st.write(df.isnull().sum())
        if st.button("Isi Missing Values dengan Median"):
            df = df.fillna(df.median(numeric_only=True))
            st.session_state.df = df
            st.success("Missing values telah diisi dengan median.")

        # Outlier Detection
        st.subheader("📦 Deteksi Outlier (Boxplot)")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Pilih kolom untuk melihat outlier:", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=selected_col, ax=ax)
        st.pyplot(fig)

        # Perhitungan Outlier
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        st.write(f"Jumlah outlier: {outliers.shape[0]}")
        st.write(f"Persentase outlier: {100 * outliers.shape[0] / df.shape[0]:.2f}%")

# EDA dan Scaling
elif menu == "EDA & Scaling":
    st.header("📊 EDA dan Standardisasi")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()

        # Histogram
        st.subheader("📈 Histogram")
        selected_col = st.selectbox("Pilih kolom untuk histogram:", df.select_dtypes(include=['float64', 'int64']).columns)
        fig, ax = plt.subplots()
        df[selected_col].hist(ax=ax, bins=20)
        st.pyplot(fig)

        # Korelasi
        st.subheader("📊 Korelasi (Heatmap)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Scaling
        st.subheader("⚙️ Standardisasi Data")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
        st.session_state.df_scaled = df_scaled
        st.dataframe(df_scaled.head())

# Clustering dan Visualisasi
elif menu == "Clustering & Visualisasi":
    st.header("🔍 Clustering dan Visualisasi")
    if 'df_scaled' not in st.session_state:
        st.warning("Silakan lakukan scaling terlebih dahulu.")
    else:
        df_scaled = st.session_state.df_scaled.copy()

        # Mean Shift Clustering
        ms = MeanShift()
        cluster_labels = ms.fit_predict(df_scaled)
        df_clustered = st.session_state.df.copy()
        df_clustered['Cluster'] = cluster_labels

        st.subheader("📑 Ringkasan Cluster")
        cluster_summary = df_clustered.groupby('Cluster').mean()
        st.dataframe(cluster_summary)

        # Visualisasi Bar Chart
        st.subheader("📊 Visualisasi Jumlah Data per Cluster")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots()
        cluster_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # Visualisasi 3D
        st.subheader("🌐 Visualisasi 3D dengan PCA")
        pca = PCA(n_components=3)
        components = pca.fit_transform(df_scaled)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=cluster_labels, cmap='viridis')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        st.pyplot(fig)

        # Simpan hasil clustering
        if st.button("💾 Simpan Hasil Clustering"):
            df_clustered.to_csv("hasil_clustering.csv", index=False)
            st.success("Hasil clustering disimpan sebagai 'hasil_clustering.csv'")
