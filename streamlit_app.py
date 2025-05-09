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

# Logo
st.sidebar.image(
    "https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png", 
    width=150
)

# Simulasi tab menu di sidebar
menu = st.sidebar.radio("📂 Navigasi", ["📤 Upload & Preprocessing", "📊 Visualisasi"])

# Kolom yang digunakan
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
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower, upper)

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

# =======================
# 📤 Upload & Preprocessing
# =======================
if menu == "📤 Upload & Preprocessing":
    st.header("📤 Upload & Preprocessing")
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil diunggah!")
        st.dataframe(df)

        # Penanganan Missing Value
        st.subheader("🧱 Missing Value Sebelum Penanganan")
        missing_before = df.isnull().sum()
        for col, count in missing_before.items():
            if count > 0:
                st.markdown(f"- **{col}**: {count} missing value")
        if missing_before.sum() == 0:
            st.success("Tidak ada missing value!")

        handle_missing_values(df)
        st.session_state.df = df

        st.subheader("🧹 Missing Value Setelah Penanganan")
        missing_after = df.isnull().sum()
        for col, count in missing_after.items():
            if count > 0:
                st.markdown(f"- **{col}**: {count} missing value")
        if missing_after.sum() == 0:
            st.success("Semua missing value telah berhasil ditangani!")

        # Boxplot sebelum penanganan
        st.subheader("📦 Plot Outlier Sebelum Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for i, col in enumerate(numeric_columns[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        st.pyplot(fig)

        # Penanganan outlier
        for col in feature_outlier[:6]:
            handle_outliers_iqr(df, col)

        # Boxplot sesudah penanganan
        st.subheader("📦 Plot Outlier Setelah Penanganan")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for i, col in enumerate(feature_outlier[:6]):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot {col}")
        st.pyplot(fig)

        # Scaling
        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
        st.session_state.scaler = scaler

        st.subheader("📏 Data Setelah Scaling")
        st.dataframe(df[scaling_columns].head())

        st.subheader("📊 EDA")
        st.dataframe(df[scaling_columns].describe().T)

        # Histogram
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(scaling_columns):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f"Histogram {col}")
        st.pyplot(fig)

        # Korelasi
        corr = df[scaling_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap Korelasi")
        st.pyplot(fig)

        # Clustering
        model = joblib.load("mean_shift_model_bandwidth_1.5.joblib")
        X_scaled = df[scaling_columns].values
        df['cluster_labels'] = model.predict(X_scaled)
        st.session_state.df = df
        st.success("✅ Clustering selesai!")

# =======================
# 📊 Visualisasi
# =======================
if menu == "📊 Visualisasi":
    if 'df' in st.session_state and 'cluster_labels' in st.session_state.df.columns:
        df = st.session_state.df.copy()

        st.subheader("📁 Data per Cluster")
        for i in sorted(df['cluster_labels'].unique()):
            st.write(f"🔸 **Cluster {i}**")
            st.dataframe(df[df['cluster_labels'] == i])

        # Rata-rata
        st.subheader("📈 Rata-rata Persentase per Cluster")
        means = df.groupby('cluster_labels')[['perc_pengurangan', 'perc_penanganan']].mean().T
        fig, ax = plt.subplots()
        means.plot(kind='bar', ax=ax)
        ax.set_title("Rata-rata Persentase Pengurangan & Penanganan")
        st.pyplot(fig)

        # 3D plot
        st.subheader("📍 Visualisasi 3D Clustering")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['sampah_tahunan'], df['pengurangan'], df['penanganan'],
                   c=df['cluster_labels'], cmap='plasma')
        ax.set_xlabel('Sampah Tahunan')
        ax.set_ylabel('Pengurangan')
        ax.set_zlabel('Penanganan')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Data belum tersedia. Silakan lakukan preprocessing terlebih dahulu.")
