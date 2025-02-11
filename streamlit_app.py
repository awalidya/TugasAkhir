import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

# Konfigurasi halaman
st.set_page_config(page_title='trash-achievement', layout='wide')

# Sidebar Menu
menu = st.sidebar.selectbox("Pilih Menu:", ["Data", "Pengolahan Data", "Algoritma", "Visualisasi"])

def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    return None

# Menu Data
if menu == "Data":
    st.title("Upload Data")
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df.head())

# Menu Pengolahan Data
elif menu == "Pengolahan Data":
    st.title("Exploratory Data Analysis (EDA) & Preprocessing")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df.rename(columns={
            'Timbulan Sampah Harian(ton)': 'sampah_harian',
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            '%Pengurangan Sampah(B/A)': 'perc_pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan',
            '%Penanganan Sampah(C/A)': 'perc_penanganan',
            'Sampah Terkelola Tahunan (ton/tahun)(B+C)': 'sampah_terkelola',
            '%Sampah Terkelola(B+C)/A': 'perc_sampah_terkelola',
            'Daur ulang Sampah Tahunan (ton/tahun)(D)': 'daur_ulang',
            'Recycling Rate(D+E)/A': 'recycling_rate'
        }, inplace=True)
        st.write("Informasi Data:")
        st.write(df.info())

        st.write("Ringkasan Statistik:")
        st.write(df.describe())

        # Visualisasi Distribusi Data
        st.subheader("Distribusi Data")
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        for column in numeric_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        # Preprocessing (Mengisi Missing Value dengan Median)
        for col in ['daur_ulang', 'pengurangan', 'perc_pengurangan', 'penanganan', 'perc_penanganan']:
            df[col] = df[col].fillna(df[col].median())
        st.write("Missing values setelah preprocessing:")
        st.write(df.isnull().sum())

# Menu Algoritma
elif menu == "Algoritma":
    st.title("Mean Shift Clustering")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        X = df[['sampah_tahunan', 'perc_pengurangan', 'perc_penanganan', 'perc_sampah_terkelola']]

        bandwidth_values = np.arange(0.5, 2.1, 0.5)
        best_score = -1
        best_bandwidth = None

        for bandwidth in bandwidth_values:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            n_clusters = len(np.unique(labels))
            if n_clusters > 1:
                sil_score = silhouette_score(X, labels)
                if sil_score > best_score:
                    best_score = sil_score
                    best_bandwidth = bandwidth

        st.write(f"Bandwidth terbaik: {best_bandwidth}, Silhouette Score: {best_score:.3f}")
        ms = MeanShift(bandwidth=best_bandwidth, bin_seeding=True)
        ms.fit(X)
        df['cluster_labels'] = ms.labels_
        st.write("Hasil Clustering:")
        st.write(df[['sampah_tahunan', 'perc_pengurangan', 'perc_penanganan', 'perc_sampah_terkelola', 'cluster_labels']])

# Menu Visualisasi
elif menu == "Visualisasi":
    st.title("Visualisasi Hasil Clustering")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if 'cluster_labels' in df.columns:
            cluster_counts = df['cluster_labels'].value_counts()
            cluster_percentages = (cluster_counts / cluster_counts.sum()) * 100

            fig, ax = plt.subplots()
            ax.pie(cluster_percentages, labels=cluster_percentages.index, autopct='%1.1f%%', startangle=140)
            ax.set_title('Persentase Tiap Cluster')
            st.pyplot(fig)
        else:
            st.warning("Lakukan clustering terlebih dahulu di menu Algoritma.")
