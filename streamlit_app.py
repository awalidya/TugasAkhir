import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

st.title("Aplikasi Clustering Capaian Pengelolaan Sampah")

# Sidebar
selected_tab = st.sidebar.selectbox("Pilih Menu", ["Upload Data", "Pengolahan Data", "Algoritma Clustering", "Visualisasi"])

# Tab 1 - Upload Data
if selected_tab == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df  # simpan data
        st.success("Data berhasil diunggah!")
        st.dataframe(df)

# Tab 2 - Pengolahan Data
elif selected_tab == "Pengolahan Data":
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())

        st.subheader("Boxplot Sebelum Penanganan Outlier")
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            ax.boxplot(df[col])
            ax.set_title(f'Boxplot {col}')
            st.pyplot(fig)

        st.subheader("Penanganan Outlier dengan Metode IQR")
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_iqr = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.session_state['data_clean'] = df_iqr
        st.success("Outlier berhasil ditangani dengan metode IQR")

        st.subheader("Boxplot Setelah Penanganan Outlier")
        for col in num_cols:
            fig, ax = plt.subplots()
            ax.boxplot(df_iqr[col])
            ax.set_title(f'Boxplot Setelah IQR - {col}')
            st.pyplot(fig)
    else:
        st.warning("Silakan upload data terlebih dahulu di tab Upload Data.")

# Tab 3 - Algoritma Clustering
elif selected_tab == "Algoritma Clustering":
    if 'data_clean' in st.session_state:
        df = st.session_state['data_clean']
        X = df[['sampah_tahunan', 'pengurangan', 'penanganan']]
        bandwidth_values = [1.0, 1.5, 2.0]

        for bw in bandwidth_values:
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            st.write(f"### Bandwidth Value: {bw}")
            fig, ax = plt.subplots()
            ax.scatter(X['sampah_tahunan'], X['penanganan'], c=labels, cmap='plasma', marker='p')
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=250, c='blue', marker='X', label='Cluster Centers')
            ax.set_title(f'Mean Shift Clustering (Bandwidth = {bw})')
            ax.set_xlabel('Sampah Tahunan')
            ax.set_ylabel('Penanganan')
            ax.legend()
            st.pyplot(fig)

            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                st.write(f"Silhouette Score: {sil_score:.3f}")
            else:
                st.write("Silhouette Score tidak dapat dihitung karena hanya ada 1 cluster.")

        # Simpan hasil klaster terakhir
        df['cluster'] = labels
        st.session_state['clustered'] = df
        st.session_state['cluster_centers'] = cluster_centers
    else:
        st.warning("Data belum diproses. Silakan selesaikan tab Pengolahan Data terlebih dahulu.")

# Tab 4 - Visualisasi
elif selected_tab == "Visualisasi":
    if 'clustered' in st.session_state:
        df = st.session_state['clustered']

        cluster_0_df = df[df['cluster'] == 0]
        cluster_1_df = df[df['cluster'] == 1]

        st.subheader("Cluster 0")
        st.dataframe(cluster_0_df)
        st.write(cluster_0_df.describe())

        st.subheader("Cluster 1")
        st.dataframe(cluster_1_df)
        st.write(cluster_1_df.describe())

        # Hitung rata-rata
        avg_cluster_0 = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
        avg_cluster_1 = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()

        st.write("Rata-rata Cluster 0:")
        st.write(avg_cluster_0)

        st.write("Rata-rata Cluster 1:")
        st.write(avg_cluster_1)

        avg_df = pd.DataFrame({
            "Klaster 1": avg_cluster_0,
            "Klaster 2": avg_cluster_1
        })

        fig, ax = plt.subplots()
        avg_df.T.plot(kind='bar', ax=ax, figsize=(8, 5), color=['blue', 'orange'])
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Rata-rata Persentase")
        ax.set_title("Rata-rata Persentase Pengurangan dan Penanganan per Klaster")
        ax.legend(["Persentase Pengurangan", "Persentase Penanganan"])

        for i, cluster in enumerate(avg_df.columns):
            for j, val in enumerate(avg_df[cluster]):
                ax.text(i + j*0.2 - 0.1, val + 0.5, round(val, 2), ha='center', fontsize=10)

        st.pyplot(fig)

        # Visualisasi 3D
        st.subheader("Visualisasi 3D Clustering")
        cluster_centers = st.session_state['cluster_centers']
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                   c=df['cluster'], cmap='plasma', marker='o', label='Data Points')
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   s=250, c='blue', marker='X', label='Cluster Centers')
        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan')
        ax.set_zlabel('Penanganan')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Belum ada hasil clustering. Silakan jalankan tab Algoritma Clustering terlebih dahulu.")
