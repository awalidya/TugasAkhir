import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Konfigurasi tampilan
st.set_page_config(page_title="Pengelompokan Wilayah Sampah", layout="wide")
st.title("📊 Aplikasi Pengelompokan Wilayah Berdasarkan Capaian Pengelolaan Sampah di Indonesia")

# Membuat tab
tabs = st.tabs(["Upload Data", "Pengolahan Data", "Algoritma Clustering", "Visualisasi"])

# Tab 1: Upload Data
with tabs[0]:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.dataframe(df)

# Tab 2: Pengolahan Data
elif selected_tab == "Pengolahan Data":
    st.header("Pengolahan Data")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Asli")
        st.write(df)

        # Histogram
        st.subheader("Histogram")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20)
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)

        # Boxplot Sebelum Penanganan Outlier
        st.subheader("Boxplot Sebelum Penanganan Outlier")
        for col in numeric_columns:
            fig, ax = plt.subplots()
            df.boxplot(column=col, ax=ax)
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)

        # Penanganan Outlier dengan Metode IQR
        st.subheader("Penanganan Outlier dengan Metode IQR")

        # Ambil hanya kolom numerik
        numerical_df = df[numeric_columns]

        # Hitung Q1, Q3, dan IQR
        Q1 = numerical_df.quantile(0.25)
        Q3 = numerical_df.quantile(0.75)
        IQR = Q3 - Q1

        # Deteksi outlier
        outliers = ((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR)))
        jumlah_outlier = outliers.sum().sum()
        persentase_outlier = (jumlah_outlier / numerical_df.size) * 100

        st.write(f"Jumlah total outlier yang terdeteksi: {jumlah_outlier}")
        st.write(f"Persentase outlier: {persentase_outlier:.2f}%")

        # Hapus outlier
        df_clean = df[~((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.subheader("Data Setelah Penanganan Outlier")
        st.write(df_clean)

        # Boxplot Setelah Penanganan Outlier
        st.subheader("Boxplot Setelah Penanganan Outlier")
        for col in numeric_columns:
            fig, ax = plt.subplots()
            df_clean.boxplot(column=col, ax=ax)
            ax.set_title(f"Boxplot Setelah Penanganan Outlier: {col}")
            st.pyplot(fig)


# Tab 3: Algoritma Clustering
with tabs[2]:
    st.header("🧠 Algoritma Clustering - Mean Shift")
    
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Pilih fitur yang digunakan untuk clustering
        features = ['sampah_tahunan', 'pengurangan', 'penanganan']
        X = df[features]

        from sklearn.cluster import MeanShift
        from sklearn.metrics import silhouette_score
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        bandwidth_values = [1.0, 1.5, 2.0]

        for bw in bandwidth_values:
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            st.write(f"### Bandwidth Value: {bw}")

            # Visualisasi hasil clustering 2D
            fig2d, ax2d = plt.subplots()
            scatter = ax2d.scatter(X['sampah_tahunan'], X['penanganan'], c=labels, cmap='plasma', marker='p')
            ax2d.scatter(cluster_centers[:, 0], cluster_centers[:, 2], s=250, c='blue', marker='X', label='Cluster Centers')
            ax2d.set_title(f'Mean Shift Clustering (Bandwidth = {bw})')
            ax2d.set_xlabel('Sampah Tahunan')
            ax2d.set_ylabel('Penanganan')
            ax2d.legend()
            st.pyplot(fig2d)

            # Silhouette Score
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                st.write(f"Silhouette Score: {sil_score:.3f}")
            else:
                st.write("Silhouette Score tidak dapat dihitung karena hanya ada 1 cluster.")

        # Jumlah cluster
        n_clusters = len(np.unique(labels))
        st.write(f"Jumlah cluster yang terbentuk (terakhir): {n_clusters}")

        # Visualisasi 3D
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        ax3d.scatter(X['sampah_tahunan'], X['pengurangan'], X['penanganan'],
                     c=labels, cmap='plasma', marker='o', label='Data Points')

        ax3d.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                     s=250, c='blue', marker='X', label='Cluster Centers')

        ax3d.set_xlabel('Sampah Tahunan')
        ax3d.set_ylabel('Pengurangan')
        ax3d.set_zlabel('Penanganan')
        ax3d.set_title('3D Mean Shift Clustering')
        ax3d.legend()
        st.pyplot(fig3d)
    else:
        st.warning("Harap unggah dan olah data terlebih dahulu.")

# Tab 4: Visualisasi
with tabs[3]:
    st.header("📊 Visualisasi Hasil Clustering")
    
    if 'df' in st.session_state and 'labels' in locals():
        df = st.session_state['df']
        df['cluster'] = labels

        # Membagi data ke dalam dua cluster
        cluster_0_df = df[df['cluster'] == 0]
        cluster_1_df = df[df['cluster'] == 1]

        # Tampilkan DataFrame masing-masing cluster
        st.subheader("Data Cluster 0")
        st.dataframe(cluster_0_df)

        st.subheader("Statistik Cluster 0")
        st.dataframe(cluster_0_df.describe())

        st.subheader("Data Cluster 1")
        st.dataframe(cluster_1_df)

        st.subheader("Statistik Cluster 1")
        st.dataframe(cluster_1_df.describe())

        # Menghitung rata-rata persentase
        avg_cluster_0 = cluster_0_df[['perc_pengurangan', 'perc_penanganan']].mean()
        avg_cluster_1 = cluster_1_df[['perc_pengurangan', 'perc_penanganan']].mean()

        # Gabungkan hasil rata-rata ke dalam satu dataframe
        avg_df = pd.DataFrame({
            "Klaster 0": avg_cluster_0,
            "Klaster 1": avg_cluster_1
        })

        # Bar Chart
        fig_avg, ax = plt.subplots(figsize=(8, 5))
        avg_df.T.plot(kind='bar', ax=ax, color=['blue', 'orange'])

        ax.set_xlabel("Klaster")
        ax.set_ylabel("Rata-rata Persentase")
        ax.set_title("Rata-rata Persentase Pengurangan dan Penanganan per Klaster")
        ax.legend(["Persentase Pengurangan", "Persentase Penanganan"])

        # Tambahkan label nilai di atas batang
        for i, cluster in enumerate(avg_df.columns):
            for j, val in enumerate(avg_df[cluster]):
                ax.text(i + j * 0.25 - 0.15, val + 0.5, round(val, 2), ha='center', fontsize=10)

        st.pyplot(fig_avg)
    else:
        st.warning("Hasil clustering belum tersedia. Silakan lakukan proses clustering terlebih dahulu.")
