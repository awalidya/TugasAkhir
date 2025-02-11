import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

# Konfigurasi halaman
st.set_page_config(page_title='trash-achievement', layout='wide')

# Membuat tab menu di bagian atas
menu = ["Upload Data", "Pengolahan Data", "Algoritma Clustering", "Visualisasi"]
tabs = st.tabs(menu)

# TAB 1: Upload Data
with tabs[0]:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['df'] = df  # Simpan dataframe di session_state
        st.write("Data yang diunggah:")
        st.dataframe(df.head())

# TAB 2: Pengolahan Data
with tabs[1]:
    st.header("Exploratory Data Analysis & Preprocessing")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Informasi Data")
        st.write(df.info())
        st.subheader("Statistik Data")
        st.write(df.describe())
        
        # Visualisasi distribusi data
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            st.subheader(f"Distribusi {column}")
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
        
        # Handling Missing Values
        df.fillna(df.median(), inplace=True)
        st.session_state['df'] = df
        st.success("Data preprocessing selesai!")
    else:
        st.warning("Harap unggah data terlebih dahulu.")

# TAB 3: Algoritma Clustering
with tabs[2]:
    st.header("Penerapan Mean Shift Clustering")
    if 'df' in st.session_state:
        df = st.session_state['df']
        X = df[['sampah_tahunan', 'perc_pengurangan', 'perc_penanganan', 'perc_sampah_terkelola']]
        bandwidth = st.slider("Pilih Bandwidth", 0.5, 2.0, 1.5, 0.1)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        df['cluster_labels'] = ms.labels_
        st.session_state['df'] = df
        
        n_clusters = len(np.unique(ms.labels_))
        st.write(f"Jumlah Cluster: {n_clusters}")
        if n_clusters > 1:
            sil_score = silhouette_score(X, ms.labels_)
            st.write(f"Silhouette Score: {sil_score:.3f}")
        
        st.write("Hasil Clustering:")
        st.dataframe(df[['sampah_tahunan', 'perc_pengurangan', 'perc_penanganan', 'perc_sampah_terkelola', 'cluster_labels']])
    else:
        st.warning("Harap unggah data terlebih dahulu.")

# TAB 4: Visualisasi
with tabs[3]:
    st.header("Visualisasi Hasil Clustering")
    if 'df' in st.session_state:
        df = st.session_state['df']
        cluster_counts = df['cluster_labels'].value_counts()
        cluster_percentages = (cluster_counts / cluster_counts.sum()) * 100
        
        fig, ax = plt.subplots()
        ax.pie(cluster_percentages, labels=cluster_percentages.index, autopct='%1.1f%%', startangle=140)
        ax.set_title("Persentase Tiap Cluster")
        st.pyplot(fig)
    else:
        st.warning("Harap unggah data terlebih dahulu.")
