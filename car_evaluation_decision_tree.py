import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="Car Evaluation Analytics | Kelompok 7",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

KOLOM = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
FITUR = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
URUTAN_KELAS = ["unacc", "acc", "good", "vgood"]
LABEL_KELAS = {
    "unacc": "Tidak Layak",
    "acc": "Cukup Layak",
    "good": "Layak",
    "vgood": "Sangat Layak",
}
WARNA_KELAS = {
    "unacc": "#ef6b6b",
    "acc": "#f5b942",
    "good": "#55b889",
    "vgood": "#5799e8",
}


# =============================================================================
# TEMA
# =============================================================================

def terapkan_tema(dark_mode: bool):
    if dark_mode:
        bg = "#101827"
        panel = "#172235"
        text = "#edf2f7"
        muted = "#a8b3c7"
        border = "#2b3a52"
        accent = "#8b9cff"
    else:
        bg = "#f4f6fb"
        panel = "#ffffff"
        text = "#20283a"
        muted = "#687386"
        border = "#e3e8f0"
        accent = "#596be8"

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        [data-testid="stSidebar"] {{ background: {panel}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        h1, h2, h3, h4, p, label {{ color: {text}; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; }}
        div[data-testid="stMetric"] {{
            background: {panel}; border: 1px solid {border};
            padding: 16px 18px; border-radius: 15px;
        }}
        div[data-testid="stMetric"] label {{ color: {muted}; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {text}; font-weight: 700;
        }}
        .hero {{
            background: linear-gradient(120deg, {panel}, {bg});
            border: 1px solid {border}; border-radius: 20px;
            padding: 24px 26px; margin-bottom: 18px;
        }}
        .hero small {{ color: {muted}; letter-spacing: 1.4px; }}
        .hero h1 {{ margin: 8px 0 5px 0; font-size: 2rem; }}
        .hero p {{ color: {muted}; margin-bottom: 0; }}
        .info-card {{
            background: {panel}; border: 1px solid {border};
            border-radius: 15px; padding: 17px; height: 100%;
        }}
        .info-card p {{ color: {muted}; margin-bottom: 0; }}
        .stButton button, .stDownloadButton button {{
            border-radius: 10px; font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return {
        "bg": bg, "panel": panel, "text": text,
        "muted": muted, "border": border, "accent": accent
    }


def gaya_grafik(fig, ax, tema):
    fig.patch.set_facecolor(tema["panel"])
    ax.set_facecolor(tema["panel"])
    ax.tick_params(colors=tema["muted"])
    ax.xaxis.label.set_color(tema["muted"])
    ax.yaxis.label.set_color(tema["muted"])
    ax.title.set_color(tema["text"])
    for spine in ax.spines.values():
        spine.set_color(tema["border"])
    ax.grid(axis="y", alpha=0.18, color=tema["muted"])
    return fig


def tampilkan_grafik(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# PENGOLAHAN DATA
# =============================================================================

class AnalisisCar:
    def __init__(self, data: pd.DataFrame, nama_dataset="Dataset"):
        self.nama_dataset = nama_dataset
        self.data = self.bersihkan_data(data)

    @staticmethod
    def bersihkan_data(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Menangani file UCI yang tidak memiliki nama kolom.
        if len(df.columns) == 7 and not set(KOLOM).issubset(set(df.columns)):
            df.columns = KOLOM

        # Menangani variasi nama kolom yang umum.
        alias = {
            "maintenance": "maint",
            "maintain": "maint",
            "luggage_boot": "lug_boot",
            "luggage": "lug_boot",
            "class_label": "class",
            "evaluation": "class",
            "target": "class",
        }
        df = df.rename(columns=alias)

        if not set(KOLOM).issubset(set(df.columns)):
            raise ValueError(
                "Kolom dataset tidak sesuai. Dataset harus memiliki kolom: "
                + ", ".join(KOLOM)
            )

        df = df[KOLOM].copy()
        for kolom in KOLOM:
            df[kolom] = df[kolom].astype(str).str.strip().str.lower()

        df = df.replace({"nan": np.nan, "none": np.nan, "": np.nan})
        df = df.dropna(subset=KOLOM).drop_duplicates().reset_index(drop=True)

        valid = {
            "buying": {"vhigh", "high", "med", "low"},
            "maint": {"vhigh", "high", "med", "low"},
            "doors": {"2", "3", "4", "5more"},
            "persons": {"2", "4", "more"},
            "lug_boot": {"small", "med", "big"},
            "safety": {"low", "med", "high"},
            "class": set(URUTAN_KELAS),
        }
        for kolom, nilai_valid in valid.items():
            df = df[df[kolom].isin(nilai_valid)]

        return df.reset_index(drop=True)

    def ringkasan(self):
        return {
            "jumlah_data": len(self.data),
            "jumlah_fitur": len(FITUR),
            "jumlah_kelas": self.data["class"].nunique(),
            "data_kosong": int(self.data.isna().sum().sum()),
        }

    def distribusi_kelas(self):
        return (
            self.data["class"]
            .value_counts()
            .reindex(URUTAN_KELAS, fill_value=0)
        )

    def rekomendasi(self, buying="low", safety="high", persons="4"):
        hasil = self.data[
            (self.data["buying"] == buying)
            & (self.data["safety"] == safety)
            & (self.data["persons"] == persons)
        ].copy()
        if hasil.empty:
            hasil = self.data[
                (self.data["buying"] == buying)
                & (self.data["safety"] == safety)
            ].copy()
        return hasil


def baca_file(file):
    nama = file.name.lower()
    if nama.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            if len(df.columns) == 1:
                file.seek(0)
                df = pd.read_csv(file, header=None)
            return df
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
    return pd.read_excel(file)


def muat_dataset(uploaded_file, nama_default):
    if uploaded_file is not None:
        return baca_file(uploaded_file)
    if os.path.exists(nama_default):
        if nama_default.lower().endswith(".csv"):
            return pd.read_csv(nama_default, header=None)
        return pd.read_excel(nama_default)
    return None


# =============================================================================
# MACHINE LEARNING
# =============================================================================

class ModelDecisionTree:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.report = None
        self.cm = None
        self.X_train = None
        self.y_train = None
        self.preprocessor = None
        self.feature_names = None

    def latih_model(self, test_size=0.2, max_depth=8, min_samples_leaf=2):
        X = self.data[FITUR]
        y = self.data["class"]

        if y.nunique() < 2:
            raise ValueError("Data harus memiliki minimal dua kelas untuk melatih model.")

        stratify = y if y.value_counts().min() >= 2 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        kategori = FITUR
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("kategori", OneHotEncoder(handle_unknown="ignore"), kategori)
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.cm = confusion_matrix(self.y_test, self.y_pred, labels=URUTAN_KELAS)
        self.report = classification_report(
            self.y_test,
            self.y_pred,
            labels=URUTAN_KELAS,
            target_names=[LABEL_KELAS[k] for k in URUTAN_KELAS],
            output_dict=True,
            zero_division=0,
        )
        return self

    def nama_fitur_encoded(self):
        encoder = self.model.named_steps["preprocessor"].named_transformers_["kategori"]
        return encoder.get_feature_names_out(FITUR)

    def feature_importance(self):
        clf = self.model.named_steps["classifier"]
        return pd.DataFrame({
            "Fitur": self.nama_fitur_encoded(),
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=False)

    def prediksi(self, input_data):
        return self.model.predict(pd.DataFrame([input_data]))[0]

    def probabilitas(self, input_data):
        probabilities = self.model.predict_proba(pd.DataFrame([input_data]))[0]
        classes = self.model.named_steps["classifier"].classes_
        return pd.DataFrame({"Kelas": classes, "Probabilitas": probabilities})


# =============================================================================
# GRAFIK MATPLOTLIB
# =============================================================================

def grafik_distribusi(data, tema):
    counts = data["buying"].value_counts().reindex(
        ["low", "med", "high", "vhigh"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    gaya_grafik(fig, ax, tema)
    ax.bar(counts.index, counts.values, color="#7186e8", width=0.58)
    ax.set_title("Distribusi Harga Pembelian", pad=14, fontweight="bold")
    ax.set_xlabel("Kategori buying")
    ax.set_ylabel("Jumlah data")
    fig.tight_layout()
    return fig


def grafik_safety(data, tema):
    urutan = ["low", "med", "high"]
    tabel = pd.crosstab(data["safety"], data["class"]).reindex(
        index=urutan, columns=URUTAN_KELAS, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    gaya_grafik(fig, ax, tema)
    bawah = np.zeros(len(tabel))
    for kelas in URUTAN_KELAS:
        ax.bar(
            tabel.index, tabel[kelas].values, bottom=bawah,
            label=LABEL_KELAS[kelas], color=WARNA_KELAS[kelas]
        )
        bawah += tabel[kelas].values
    ax.set_title("Hubungan Safety dengan Kelas Mobil", pad=14, fontweight="bold")
    ax.set_xlabel("Kategori safety")
    ax.set_ylabel("Jumlah data")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def grafik_donut(data, tema):
    counts = data["class"].value_counts().reindex(URUTAN_KELAS, fill_value=0)
    values = counts[counts > 0]
    labels = [LABEL_KELAS[k] for k in values.index]
    colors = [WARNA_KELAS[k] for k in values.index]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor(tema["panel"])
    wedges, _ = ax.pie(
        values.values, startangle=90, colors=colors,
        wedgeprops={"width": 0.38, "edgecolor": tema["panel"]}
    )
    ax.text(
        0, 0, f"{len(data):,}\nData",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=tema["text"]
    )
    ax.set_title("Proporsi Kelas Mobil", color=tema["text"], pad=12, fontweight="bold")
    ax.legend(
        wedges, labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
        frameon=False, labelcolor=tema["muted"], fontsize=8
    )
    fig.tight_layout()
    return fig


def grafik_buying_maint(data, tema):
    tabel = pd.crosstab(data["buying"], data["maint"]).reindex(
        index=["low", "med", "high", "vhigh"],
        columns=["low", "med", "high", "vhigh"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    gaya_grafik(fig, ax, tema)
    x = np.arange(len(tabel.index))
    lebar = 0.18
    for i, kolom in enumerate(tabel.columns):
        ax.bar(
            x + (i - 1.5) * lebar, tabel[kolom].values,
            width=lebar, label=kolom
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tabel.index)
    ax.set_title("Perbandingan Buying dan Maintenance", pad=14, fontweight="bold")
    ax.set_xlabel("Buying")
    ax.set_ylabel("Jumlah data")
    ax.legend(title="Maintenance", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def grafik_feature_importance(model, tema):
    imp = model.feature_importance().head(12).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    gaya_grafik(fig, ax, tema)
    ax.barh(imp["Fitur"], imp["Importance"], color="#7186e8")
    ax.set_title("Feature Importance Decision Tree", pad=14, fontweight="bold")
    ax.set_xlabel("Nilai importance")
    fig.tight_layout()
    return fig


def grafik_confusion_matrix(model, tema):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(tema["panel"])
    ax.set_facecolor(tema["panel"])
    im = ax.imshow(model.cm, cmap="Blues")
    ax.set_xticks(range(len(URUTAN_KELAS)))
    ax.set_yticks(range(len(URUTAN_KELAS)))
    ax.set_xticklabels([LABEL_KELAS[k] for k in URUTAN_KELAS], rotation=25, ha="right")
    ax.set_yticklabels([LABEL_KELAS[k] for k in URUTAN_KELAS])
    ax.set_xlabel("Prediksi", color=tema["muted"])
    ax.set_ylabel("Aktual", color=tema["muted"])
    ax.set_title("Confusion Matrix", color=tema["text"], fontweight="bold", pad=12)
    threshold = model.cm.max() / 2 if model.cm.size else 0
    for i in range(model.cm.shape[0]):
        for j in range(model.cm.shape[1]):
            ax.text(
                j, i, str(model.cm[i, j]), ha="center", va="center",
                color="white" if model.cm[i, j] > threshold else "#1f2937",
                fontweight="bold"
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def grafik_probabilitas(prob_df, tema):
    prob_df = prob_df.sort_values("Probabilitas")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    gaya_grafik(fig, ax, tema)
    warna = [WARNA_KELAS.get(k, "#7186e8") for k in prob_df["Kelas"]]
    ax.barh(
        [LABEL_KELAS.get(k, k) for k in prob_df["Kelas"]],
        prob_df["Probabilitas"], color=warna
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilitas")
    ax.set_title("Probabilitas Prediksi", pad=14, fontweight="bold")
    fig.tight_layout()
    return fig


def grafik_perbandingan(data_2025, data_2026, tema):
    d1 = data_2025["class"].value_counts().reindex(URUTAN_KELAS, fill_value=0)
    d2 = data_2026["class"].value_counts().reindex(URUTAN_KELAS, fill_value=0)
    x = np.arange(len(URUTAN_KELAS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4))
    gaya_grafik(fig, ax, tema)
    ax.bar(x - width / 2, d1.values, width, label="2025", color="#7186e8")
    ax.bar(x + width / 2, d2.values, width, label="2026", color="#55b889")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_KELAS[k] for k in URUTAN_KELAS])
    ax.set_ylabel("Jumlah data")
    ax.set_title("Perbandingan Kelas Mobil 2025 dan 2026", pad=14, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# =============================================================================
# SESSION STATE DAN SIDEBAR
# =============================================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    st.markdown("## 🚗 Car Analytics")
    st.caption("Kelompok 7 | Car Evaluation")
    st.session_state.dark_mode = st.toggle(
        "Dark Mode", value=st.session_state.dark_mode
    )
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["Dashboard", "Machine Learning", "Explore Data", "Katalog Mobil", "Compare Dataset"],
    )
    st.divider()
    st.caption("Car Evaluation Analytics")
    st.caption("Decision Tree Classification")

TEMA = terapkan_tema(st.session_state.dark_mode)


# =============================================================================
# LOAD DATASET
# =============================================================================

st.sidebar.markdown("### Dataset")
file_2025 = st.sidebar.file_uploader(
    "Upload dataset 2025", type=["csv", "xlsx", "xls"], key="data2025"
)
file_2026 = st.sidebar.file_uploader(
    "Upload dataset 2026", type=["csv", "xlsx", "xls"], key="data2026"
)

try:
    raw_2025 = muat_dataset(file_2025, "car_evaluation_2025.xlsx")
    raw_2026 = muat_dataset(file_2026, "car_evaluation_2026.xlsx")

    if raw_2025 is None:
        st.markdown(
            """
            <div class="hero">
                <small>CAR EVALUATION ANALYTICS • KELOMPOK 7</small>
                <h1>Dashboard Analisis Kelayakan Mobil</h1>
                <p>Silakan upload dataset 2025 dan/atau 2026 melalui sidebar untuk mulai menggunakan dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Dataset perlu memiliki 7 kolom: buying, maint, doors, persons, "
            "lug_boot, safety, dan class. File CSV tanpa header juga didukung."
        )
        st.stop()

    analisis_2025 = AnalisisCar(raw_2025, "2025")
    analisis_2026 = AnalisisCar(raw_2026, "2026") if raw_2026 is not None else None
    data = analisis_2025.data

except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <small>CAR EVALUATION ANALYTICS • KELOMPOK 7</small>
        <h1>Car Evaluation Dashboard</h1>
        <p>Analisis data, visualisasi, dan prediksi kelayakan mobil menggunakan Decision Tree.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HALAMAN 1 - DASHBOARD
# =============================================================================

if halaman == "Dashboard":
    st.subheader("Overview Dataset 2025")
    ringkasan = analisis_2025.ringkasan()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{ringkasan['jumlah_data']:,}")
    c2.metric("Jumlah Fitur", ringkasan["jumlah_fitur"])
    c3.metric("Jumlah Kelas", ringkasan["jumlah_kelas"])
    c4.metric("Data Kosong", ringkasan["data_kosong"])

    st.divider()
    kiri, kanan = st.columns(2)
    with kiri:
        tampilkan_grafik(grafik_distribusi(data, TEMA))
    with kanan:
        tampilkan_grafik(grafik_safety(data, TEMA))

    kiri, kanan = st.columns(2)
    with kiri:
        tampilkan_grafik(grafik_donut(data, TEMA))
    with kanan:
        tampilkan_grafik(grafik_buying_maint(data, TEMA))

    st.subheader("Ringkasan Kelas Mobil")
    distribusi = analisis_2025.distribusi_kelas()
    tabel_distribusi = pd.DataFrame({
        "Kelas": [LABEL_KELAS[k] for k in URUTAN_KELAS],
        "Jumlah": [int(distribusi[k]) for k in URUTAN_KELAS],
        "Persentase": [
            f"{(distribusi[k] / len(data) * 100):.2f}%" if len(data) else "0%"
            for k in URUTAN_KELAS
        ],
    })
    st.dataframe(tabel_distribusi, use_container_width=True, hide_index=True)

    st.subheader("Preview Dataset")
    st.dataframe(data.head(15), use_container_width=True, hide_index=True)


# =============================================================================
# HALAMAN 2 - MACHINE LEARNING
# =============================================================================

elif halaman == "Machine Learning":
    st.subheader("Decision Tree Classification")
    st.write(
        "Atur parameter model di bawah ini, lalu latih model untuk melihat "
        "hasil klasifikasi kelayakan mobil."
    )

    with st.expander("Pengaturan Parameter Model", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
        with col2:
            max_depth = st.slider("Max Depth", 1, 20, 8)
        with col3:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2)

    if st.button("🚀 Train Decision Tree", type="primary", use_container_width=True):
        try:
            with st.spinner("Sedang melatih model..."):
                model = ModelDecisionTree(data)
                model.latih_model(
                    test_size=test_size,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                )
                st.session_state.model_car = model
            st.success("Model berhasil dilatih!")
        except Exception as e:
            st.error(f"Model gagal dilatih: {e}")

    if "model_car" in st.session_state:
        model = st.session_state.model_car
        st.divider()
        st.markdown("### Hasil Evaluasi Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{model.accuracy * 100:.2f}%")
        col2.metric("Data Training", len(model.X_train))
        col3.metric("Data Testing", len(model.X_test))

        kiri, kanan = st.columns(2)
        with kiri:
            tampilkan_grafik(grafik_feature_importance(model, TEMA))
        with kanan:
            tampilkan_grafik(grafik_confusion_matrix(model, TEMA))

        st.markdown("### Classification Report")
        report_df = pd.DataFrame(model.report).T
        st.dataframe(report_df.round(3), use_container_width=True)

        st.markdown("### Struktur Decision Tree")
        if st.checkbox("Tampilkan visualisasi pohon keputusan"):
            clf = model.model.named_steps["classifier"]
            feature_names = model.nama_fitur_encoded()
            fig, ax = plt.subplots(figsize=(20, 9))
            fig.patch.set_facecolor(TEMA["panel"])
            plot_tree(
                clf,
                feature_names=feature_names,
                class_names=[LABEL_KELAS.get(k, k) for k in clf.classes_],
                filled=True,
                rounded=True,
                fontsize=7,
                ax=ax,
            )
            ax.set_title("Decision Tree", color=TEMA["text"], pad=15)
            fig.tight_layout()
            tampilkan_grafik(fig)

        st.divider()
        st.markdown("### Simulasi Prediksi Mobil")
        p1, p2, p3 = st.columns(3)
        with p1:
            buying = st.selectbox("Buying", ["low", "med", "high", "vhigh"])
            maint = st.selectbox("Maintenance", ["low", "med", "high", "vhigh"])
        with p2:
            doors = st.selectbox("Doors", ["2", "3", "4", "5more"])
            persons = st.selectbox("Persons", ["2", "4", "more"])
        with p3:
            lug_boot = st.selectbox("Luggage Boot", ["small", "med", "big"])
            safety = st.selectbox("Safety", ["low", "med", "high"])

        if st.button("Prediksi Kelayakan Mobil", use_container_width=True):
            input_data = {
                "buying": buying,
                "maint": maint,
                "doors": doors,
                "persons": persons,
                "lug_boot": lug_boot,
                "safety": safety,
            }
            hasil = model.prediksi(input_data)
            prob = model.probabilitas(input_data)
            st.success(f"Hasil prediksi: **{LABEL_KELAS.get(hasil, hasil)} ({hasil})**")
            kiri, kanan = st.columns(2)
            with kiri:
                tampilkan_grafik(grafik_probabilitas(prob, TEMA))
            with kanan:
                st.dataframe(
                    prob.assign(
                        Kelas=prob["Kelas"].map(lambda x: LABEL_KELAS.get(x, x)),
                        Probabilitas=prob["Probabilitas"].map(lambda x: f"{x * 100:.2f}%"),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Klik tombol Train Decision Tree untuk menampilkan hasil model.")


# =============================================================================
# HALAMAN 3 - EXPLORE DATA
# =============================================================================

elif halaman == "Explore Data":
    st.subheader("Eksplorasi Dataset")
    st.write("Gunakan filter untuk melihat data berdasarkan kategori tertentu.")

    f1, f2, f3 = st.columns(3)
    with f1:
        pilih_buying = st.multiselect(
            "Buying", sorted(data["buying"].unique()),
            default=sorted(data["buying"].unique())
        )
        pilih_maint = st.multiselect(
            "Maintenance", sorted(data["maint"].unique()),
            default=sorted(data["maint"].unique())
        )
    with f2:
        pilih_doors = st.multiselect(
            "Doors", sorted(data["doors"].unique()),
            default=sorted(data["doors"].unique())
        )
        pilih_persons = st.multiselect(
            "Persons", sorted(data["persons"].unique()),
            default=sorted(data["persons"].unique())
        )
    with f3:
        pilih_lug = st.multiselect(
            "Luggage Boot", sorted(data["lug_boot"].unique()),
            default=sorted(data["lug_boot"].unique())
        )
        pilih_safety = st.multiselect(
            "Safety", sorted(data["safety"].unique()),
            default=sorted(data["safety"].unique())
        )
        pilih_class = st.multiselect(
            "Class", URUTAN_KELAS, default=URUTAN_KELAS
        )

    hasil_filter = data[
        data["buying"].isin(pilih_buying)
        & data["maint"].isin(pilih_maint)
        & data["doors"].isin(pilih_doors)
        & data["persons"].isin(pilih_persons)
        & data["lug_boot"].isin(pilih_lug)
        & data["safety"].isin(pilih_safety)
        & data["class"].isin(pilih_class)
    ]

    st.metric("Jumlah Data Setelah Filter", f"{len(hasil_filter):,}")
    st.dataframe(hasil_filter, use_container_width=True, hide_index=True)

    csv = hasil_filter.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Filter (CSV)",
        data=csv,
        file_name="hasil_filter_car_evaluation.csv",
        mime="text/csv",
    )

    st.subheader("Statistik Deskriptif")
    st.dataframe(
        hasil_filter.describe(include="all").fillna("-"),
        use_container_width=True,
    )


# =============================================================================
# HALAMAN 4 - KATALOG MOBIL
# =============================================================================

elif halaman == "Katalog Mobil":
    st.subheader("Katalog dan Rekomendasi Mobil")
    st.write("Pilih kriteria mobil untuk melihat data yang sesuai.")

    c1, c2, c3 = st.columns(3)
    with c1:
        filter_buying = st.selectbox("Kategori Buying", ["Semua", "low", "med", "high", "vhigh"])
    with c2:
        filter_safety = st.selectbox("Kategori Safety", ["Semua", "low", "med", "high"])
    with c3:
        filter_persons = st.selectbox("Kapasitas Penumpang", ["Semua", "2", "4", "more"])

    katalog = data.copy()
    if filter_buying != "Semua":
        katalog = katalog[katalog["buying"] == filter_buying]
    if filter_safety != "Semua":
        katalog = katalog[katalog["safety"] == filter_safety]
    if filter_persons != "Semua":
        katalog = katalog[katalog["persons"] == filter_persons]

    st.metric("Mobil yang Sesuai Kriteria", f"{len(katalog):,} data")
    if not katalog.empty:
        st.dataframe(katalog, use_container_width=True, hide_index=True)

        ringkas = (
            katalog.groupby("class")
            .size()
            .reindex(URUTAN_KELAS, fill_value=0)
            .reset_index(name="Jumlah")
        )
        ringkas["Keterangan"] = ringkas["class"].map(LABEL_KELAS)
        st.markdown("### Ringkasan Rekomendasi")
        st.dataframe(
            ringkas[["class", "Keterangan", "Jumlah"]],
            use_container_width=True,
            hide_index=True,
        )
        terbaik = katalog[katalog["class"].isin(["good", "vgood"])]
        if not terbaik.empty:
            st.success(
                f"Ditemukan {len(terbaik)} data dengan kategori Layak atau Sangat Layak."
            )
        else:
            st.warning("Belum ditemukan data dengan kategori Layak atau Sangat Layak.")
    else:
        st.warning("Tidak ada data yang sesuai dengan kombinasi filter tersebut.")


# =============================================================================
# HALAMAN 5 - COMPARE DATASET
# =============================================================================

elif halaman == "Compare Dataset":
    st.subheader("Perbandingan Dataset 2025 dan 2026")

    if analisis_2026 is None:
        st.info("Upload dataset 2026 melalui sidebar untuk menggunakan fitur perbandingan.")
    else:
        data_2026 = analisis_2026.data
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data 2025", f"{len(data):,}")
        col2.metric("Jumlah Data 2026", f"{len(data_2026):,}")
        perubahan = len(data_2026) - len(data)
        col3.metric("Selisih Jumlah Data", f"{perubahan:+,}")

        tampilkan_grafik(grafik_perbandingan(data, data_2026, TEMA))

        tabel = pd.DataFrame({
            "Kelas": [LABEL_KELAS[k] for k in URUTAN_KELAS],
            "2025": [
                int((data["class"] == k).sum()) for k in URUTAN_KELAS
            ],
            "2026": [
                int((data_2026["class"] == k).sum()) for k in URUTAN_KELAS
            ],
        })
        tabel["Perubahan"] = tabel["2026"] - tabel["2025"]
        st.dataframe(tabel, use_container_width=True, hide_index=True)

        total_2025 = len(data)
        total_2026 = len(data_2026)
        layak_2025 = data["class"].isin(["good", "vgood"]).sum()
        layak_2026 = data_2026["class"].isin(["good", "vgood"]).sum()
        persen_2025 = layak_2025 / total_2025 * 100 if total_2025 else 0
        persen_2026 = layak_2026 / total_2026 * 100 if total_2026 else 0

        st.markdown("### Perbandingan Kategori Layak")
        a, b = st.columns(2)
        a.metric("Layak 2025", f"{persen_2025:.2f}%")
        b.metric("Layak 2026", f"{persen_2026:.2f}%")

        if persen_2026 > persen_2025:
            st.success("Persentase mobil kategori Layak/Sangat Layak meningkat pada dataset 2026.")
        elif persen_2026 < persen_2025:
            st.warning("Persentase mobil kategori Layak/Sangat Layak menurun pada dataset 2026.")
        else:
            st.info("Persentase mobil kategori Layak/Sangat Layak sama pada kedua dataset.")


# =============================================================================
# KESIMPULAN DAN DOWNLOAD LAPORAN
# =============================================================================

st.divider()
with st.expander("Kesimpulan Analisis"):
    distribusi = analisis_2025.distribusi_kelas()
    kelas_terbanyak = distribusi.idxmax() if len(distribusi) else "-"
    st.write(
        f"Berdasarkan dataset 2025, terdapat **{len(data):,} data** yang "
        f"dianalisis dengan **{len(FITUR)} fitur**. Kelas yang paling banyak "
        f"ditemukan adalah **{LABEL_KELAS.get(kelas_terbanyak, kelas_terbanyak)}**."
    )
    st.write(
        "Fitur buying, maintenance, doors, persons, luggage boot, dan safety "
        "digunakan sebagai variabel untuk menentukan kelas kelayakan mobil."
    )
    if "model_car" in st.session_state:
        st.write(
            f"Model Decision Tree yang terakhir dilatih memperoleh akurasi "
            f"**{st.session_state.model_car.accuracy * 100:.2f}%** pada data testing."
        )

laporan = [
    "LAPORAN CAR EVALUATION ANALYTICS - KELOMPOK 7",
    f"Waktu dibuat: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
    "",
    "RINGKASAN DATASET 2025",
    f"Jumlah data: {len(data)}",
    f"Jumlah fitur: {len(FITUR)}",
    "",
    "DISTRIBUSI KELAS",
]
for kelas in URUTAN_KELAS:
    jumlah = int((data["class"] == kelas).sum())
    laporan.append(f"{LABEL_KELAS[kelas]} ({kelas}): {jumlah}")

if analisis_2026 is not None:
    laporan += [
        "",
        "RINGKASAN DATASET 2026",
        f"Jumlah data: {len(analisis_2026.data)}",
    ]

if "model_car" in st.session_state:
    laporan += [
        "",
        "HASIL MACHINE LEARNING",
        f"Accuracy: {st.session_state.model_car.accuracy * 100:.2f}%",
    ]

st.download_button(
    "⬇️ Download Kesimpulan (TXT)",
    data="\n".join(laporan).encode("utf-8"),
    file_name="laporan_car_evaluation.txt",
    mime="text/plain",
)
