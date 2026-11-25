import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==============================================================================
# KONFIGURASI HALAMAN & SESSION STATE
# ==============================================================================
st.set_page_config(
    page_title='Car Evaluation Analytics & ML • Kelompok 7',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='collapsed'
)

NAMA_KELOMPOK = 'Kelompok 7'
TAHUN_1, TAHUN_2 = '2025', '2026'
FILE_LOKAL = {TAHUN_1: 'Car_Evaluation_2025.xlsx', TAHUN_2: 'Car_Evaluation_2026.xlsx'}
FILE_ASLI = '14_Car_Evaluation.xlsx'

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'dashboard'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True


# ==============================================================================
# KELAS LOGIKA ANALISIS DATA
# ==============================================================================
class AnalisisCar:
    NAMA_KOLOM = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    FITUR = NAMA_KOLOM[:-1]
    URUTAN_KELAS = ['unacc', 'acc', 'good', 'vgood']
    LABEL_KELAS = {
        'unacc': 'Unacceptable',
        'acc': 'Acceptable',
        'good': 'Good',
        'vgood': 'Very Good'
    }
    URUTAN_LEVEL = ['low', 'med', 'high', 'vhigh']
    URUTAN_SAFETY = ['low', 'med', 'high']
    MAP_WARNA_KELAS = {
        'unacc': '#EF4444',
        'acc': '#F59E0B',
        'good': '#3B82F6',
        'vgood': '#10B981'
    }
    KRITERIA_A = 'buying=low & safety=high'
    KRITERIA_B = 'buying=low & safety=high & class=acc'
    JUDUL_PERBANDINGAN = 'Perbandingan Jumlah Mobil Memenuhi Kriteria (2025 vs 2026)'
    CATATAN_SIMULASI = (
        'Catatan: Pembagian 2025/2026 merupakan simulasi berdasarkan pemisahan baris dataset asli. '
        'Dataset Car Evaluation tidak memuat variabel tahun maupun volume penjualan aktual.'
    )

    def __init__(self, df):
        self.df = df
        self.daftar_tahun = sorted(df['tahun'].unique())

    @staticmethod
    def baca_excel(sumber, tahun):
        if isinstance(sumber, bytes):
            sumber = io.BytesIO(sumber)
        df = pd.read_excel(
            sumber, header=None, names=AnalisisCar.NAMA_KOLOM, engine='openpyxl'
        ).dropna(how='all')
        for kolom in AnalisisCar.NAMA_KOLOM:
            df[kolom] = (
                df[kolom].astype('string').str.strip().str.lower()
                .str.replace(r'\.1$', '', regex=True)
            )
        df = df[df['buying'] != 'buying'].dropna(subset=AnalisisCar.NAMA_KOLOM)
        df = df.reset_index(drop=True)
        df['tahun'] = str(tahun)
        return df

    @staticmethod
    def gabung(daftar_df):
        return pd.concat(daftar_df, ignore_index=True)

    def validasi(self):
        masalah = []
        if self.df.empty:
            return ['Dataset tidak memiliki data valid.']
        nilai_valid = {
            'buying': ['low', 'med', 'high', 'vhigh'],
            'maint': ['low', 'med', 'high', 'vhigh'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high'],
            'class': self.URUTAN_KELAS,
        }
        for kolom, pilihan in nilai_valid.items():
            if not self.df[kolom].isin(pilihan).all():
                contoh = self.df.loc[~self.df[kolom].isin(pilihan), kolom].unique().tolist()
                masalah.append(f'Kolom {kolom} memiliki nilai tidak dikenal: {contoh[:5]}')
        return masalah

    def jumlah_per_tahun(self):
        return self.df['tahun'].value_counts().sort_index()

    def jumlah_kelas(self):
        return self.df['class'].value_counts().reindex(self.URUTAN_KELAS, fill_value=0)

    def persen_unacc(self):
        n = int((self.df['class'] == 'unacc').sum())
        return n, (n / len(self.df) * 100) if len(self.df) else 0.0

    def _urutan(self, kolom):
        return self.URUTAN_SAFETY if kolom == 'safety' else self.URUTAN_LEVEL

    def tabel_silang(self, kolom, persen=False):
        tabel = pd.crosstab(
            self.df[kolom], self.df['class'],
            normalize='index' if persen else False
        )
        if persen:
            tabel *= 100
        return tabel.reindex(
            index=self._urutan(kolom),
            columns=self.URUTAN_KELAS,
            fill_value=0
        )

    def safety_high_unacc(self):
        tinggi = self.df[self.df['safety'] == 'high']
        return int((tinggi['class'] == 'unacc').sum()), len(tinggi)

    def mobil_bagus(self):
        return self.df[self.df['class'].isin(['good', 'vgood'])]

    def safety_low(self):
        rendah = self.df[self.df['safety'] == 'low']
        return len(rendah), int((rendah['class'] == 'unacc').sum())

    def buying_low(self):
        murah = self.df[self.df['buying'] == 'low']
        n_unacc = int((murah['class'] == 'unacc').sum())
        return len(murah), n_unacc, (n_unacc / len(murah) * 100) if len(murah) else 0.0

    def rekomendasi(self, harga='Semua', safety='Semua', persons='Semua', tahun='Semua'):
        pilih = self.df[self.df['class'].isin(['good', 'vgood'])].copy()
        if harga != 'Semua':
            pilih = pilih[pilih['buying'] == harga]
        if safety != 'Semua':
            pilih = pilih[pilih['safety'] == safety]
        if persons != 'Semua':
            pilih = pilih[pilih['persons'] == persons]
        if tahun != 'Semua':
            pilih = pilih[pilih['tahun'] == tahun]
        if pilih.empty:
            return pilih
        pilih['_kelas'] = pilih['class'].map({'vgood': 0, 'good': 1})
        pilih['_maint'] = pilih['maint'].map({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
        return pilih.sort_values(['_kelas', '_maint'])[self.NAMA_KOLOM + ['tahun']]

    def kriteria_per_tahun(self):
        a = self.df[(self.df['buying'] == 'low') & (self.df['safety'] == 'high')]
        b = a[a['class'] == 'acc']
        res = pd.DataFrame({
            self.KRITERIA_A: a['tahun'].value_counts(),
            self.KRITERIA_B: b['tahun'].value_counts()
        }).reindex(self.daftar_tahun).fillna(0).astype(int)
        res.index.name = 'Tahun'
        return res

    def detail_kelas_per_tahun(self):
        a = self.df[(self.df['buying'] == 'low') & (self.df['safety'] == 'high')]
        return pd.crosstab(a['tahun'], a['class']).reindex(
            index=self.daftar_tahun, columns=self.URUTAN_KELAS, fill_value=0
        )

    def ringkas_perbandingan(self):
        return [
            f'Tahun {tahun}: {nilai[self.KRITERIA_A]} mobil memenuhi {self.KRITERIA_A}, '
            f'{nilai[self.KRITERIA_B]} di antaranya berkelas acc.'
            for tahun, nilai in self.kriteria_per_tahun().iterrows()
        ]

    def kesimpulan(self):
        total = len(self.df)
        baik = len(self.mobil_bagus())
        n_unacc, p_unacc = self.persen_unacc()
        n_low, n_low_unacc = self.safety_low()
        _, _, p_murah = self.buying_low()
        return [
            f'Dari {total:,} kombinasi spesifikasi mobil, {baik:,} ({baik / total * 100:.2f}%) '
            f'berkategori Good / Very Good dan {n_unacc:,} ({p_unacc:.2f}%) Unacceptable.',
            f'Dari {n_low:,} mobil dengan safety rendah (low), {n_low_unacc:,} '
            f'({n_low_unacc / n_low * 100 if n_low else 0:.2f}%) berstatus Unacceptable.',
            f'Sebanyak {p_murah:.2f}% mobil dengan harga beli murah (buying=low) tetap Unacceptable.'
        ] + self.ringkas_perbandingan() + [self.CATATAN_SIMULASI]

    def rekap_teks(self):
        n_unacc, p_unacc = self.persen_unacc()
        a, b = self.safety_high_unacc()
        n_low, n_low_unacc = self.safety_low()
        n_murah, murah_unacc, p_murah = self.buying_low()
        baris = [
            f'=== REKAP ANALISIS CAR EVALUATION ({NAMA_KELOMPOK}) ===',
            f'Total data: {len(self.df)} {self.jumlah_per_tahun().to_dict()}',
            f'Jumlah per kelas: {self.jumlah_kelas().to_dict()}',
            f'Persen unacc: {p_unacc:.2f}% ({n_unacc})',
            f'Safety=high -> unacc: {a} dari {b}',
            f'Good/vgood: {len(self.mobil_bagus())}',
            f'Safety=low -> unacc: {n_low_unacc} dari {n_low}',
            f'Buying=low -> unacc: {murah_unacc} dari {n_murah} ({p_murah:.2f}%)',
            '',
            self.JUDUL_PERBANDINGAN
        ] + self.ringkas_perbandingan() + ['', self.CATATAN_SIMULASI]
        return '\n'.join(baris) + '\n'


# ==============================================================================
# KELAS MACHINE LEARNING: DECISION TREE
# ==============================================================================
class ModelDecisionTree:
    def __init__(self, df, test_size=0.20, max_depth=8, min_samples_leaf=1):
        self.fitur = AnalisisCar.FITUR
        X = df[self.fitur].astype(str)
        y = df['class'].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.pipeline = Pipeline([
            ('encoder', ColumnTransformer([
                ('kategori', OneHotEncoder(handle_unknown='ignore'), self.fitur)
            ])),
            ('pohon', DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            ))
        ])
        self.pipeline.fit(X_train, y_train)
        self.prediksi_test = self.pipeline.predict(X_test)
        self.prediksi_train = self.pipeline.predict(X_train)

    def evaluasi(self):
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, self.prediksi_test, average='weighted', zero_division=0
        )
        return {
            'Test Accuracy': accuracy_score(self.y_test, self.prediksi_test),
            'Train Accuracy': accuracy_score(self.y_train, self.prediksi_train),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

    def laporan(self):
        return pd.DataFrame(
            classification_report(
                self.y_test, self.prediksi_test, output_dict=True, zero_division=0
            )
        ).T

    def matriks_konfusi(self):
        urutan = AnalisisCar.URUTAN_KELAS
        return confusion_matrix(self.y_test, self.prediksi_test, labels=urutan), urutan

    def feature_importance(self):
        encoder = self.pipeline.named_steps['encoder']
        names = encoder.get_feature_names_out()
        importance = self.pipeline.named_steps['pohon'].feature_importances_
        hasil = pd.DataFrame({'encoded_feature': names, 'importance': importance})
        return pd.Series({
            f: float(
                hasil.loc[
                    hasil['encoded_feature'].str.startswith(f'kategori__{f}_'),
                    'importance'
                ].sum()
            )
            for f in self.fitur
        }).sort_values(ascending=True)

    def tree_figure(self, max_plot_depth=3, dark_mode=True):
        names = self.pipeline.named_steps['encoder'].get_feature_names_out()
        clf = self.pipeline.named_steps['pohon']
        bg = '#1E1F20' if dark_mode else '#FFFFFF'
        fig, ax = plt.subplots(figsize=(18, 8), facecolor=bg)
        ax.set_facecolor(bg)
        plot_tree(
            clf,
            feature_names=names,
            class_names=clf.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=max_plot_depth,
            ax=ax
        )
        fig.tight_layout()
        return fig

    def predict(self, input_dict):
        baru = pd.DataFrame([input_dict], columns=self.fitur)
        hasil = str(self.pipeline.predict(baru)[0])
        proba = self.pipeline.predict_proba(baru)[0]
        peluang = pd.Series(proba, index=self.pipeline.classes_).reindex(
            AnalisisCar.URUTAN_KELAS, fill_value=0
        ) * 100
        return hasil, peluang


# ==============================================================================
# TEMA PREMIUM GEMINI-STYLE (DARK: ABU HITAM | LIGHT: PUTIH BIRU KEABU-ABUAN)
# ==============================================================================
def terapkan_tema_gemini(dark_mode: bool):
    if dark_mode:
        # Gelap: Abu Hitam (Gemini Dark Surface)
        bg_main = "#131314"
        bg_sidebar = "#1E1F20"
        bg_card = "#1E1F20"
        bg_card_hover = "#282A2C"
        nav_box_bg = "#252729"
        nav_box_hover = "#2F3236"
        nav_active_bg = "linear-gradient(135deg, rgba(138, 180, 248, 0.22), rgba(168, 199, 250, 0.12))"
        nav_active_border = "#8AB4F8"
        nav_active_text = "#D2E3FC"
        border_color = "#2E3033"
        text_primary = "#E3E3E3"
        text_secondary = "#9AA0A6"
        accent = "#8AB4F8"
        shadow = "0 6px 24px rgba(0, 0, 0, 0.45)"
        header_rgba = "rgba(19, 19, 20, 0.90)"
    else:
        # Terang: Putih Biru Rada Keabu-abuan (Gemini Soft Cool Light)
        bg_main = "#EFF3F9"
        bg_sidebar = "#E6ECF5"
        bg_card = "#F8FAFD"
        bg_card_hover = "#FFFFFF"
        nav_box_bg = "#DCE4F0"
        nav_box_hover = "#D0DCF0"
        nav_active_bg = "linear-gradient(135deg, #D3E3FD, #E4EEFF)"
        nav_active_border = "#1A73E8"
        nav_active_text = "#041E49"
        border_color = "#CBD6E6"
        text_primary = "#1F2937"
        text_secondary = "#526075"
        accent = "#1A73E8"
        shadow = "0 4px 18px rgba(26, 115, 232, 0.08)"
        header_rgba = "rgba(239, 243, 249, 0.90)"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Background Utama */
        .stApp {{
            background-color: {bg_main};
            color: {text_primary};
        }}

        /* Sidebar Task Bar Gaya Gemini */
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border_color};
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_primary};
        }}

        /* HANYA sembunyikan tombol Deploy & menu titik 3, JANGAN sembunyikan stToolbar karena memuat tombol buka sidebar! */
        [data-testid="stAppDeployButton"],
        .stDeployButton,
        #MainMenu,
        footer {{
            display: none !important;
            visibility: hidden !important;
        }}

        /* Pastikan header dan tombol pembuka sidebar (garis 3) SELALU muncul & mudah diklik */
        header[data-testid="stHeader"] {{
            background: {header_rgba} !important;
            backdrop-filter: blur(8px);
            z-index: 99999 !important;
        }}

        [data-testid="stToolbar"] {{
            visibility: visible !important;
            display: flex !important;
        }}

        button[data-testid="stExpandSidebarButton"],
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapsedControl"] button {{
            visibility: visible !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            width: 42px !important;
            height: 42px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}
        button[data-testid="stExpandSidebarButton"]:hover,
        [data-testid="collapsedControl"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover {{
            border-color: {accent} !important;
            background-color: {bg_card_hover} !important;
            transform: scale(1.04);
        }}
        /* Ganti ikon panah bawaan menjadi ikon garis 3 (hamburger menu) */
        button[data-testid="stExpandSidebarButton"] svg,
        [data-testid="collapsedControl"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg {{
            display: none !important;
        }}
        button[data-testid="stExpandSidebarButton"]::after,
        [data-testid="collapsedControl"] button::after,
        [data-testid="stSidebarCollapsedControl"] button::after {{
            content: "☰";
            font-size: 20px;
            font-weight: 700;
            color: {text_primary};
            line-height: 1;
        }}

        /* Styling Kotak-Kotak Menu di Taskbar Kiri (Premium Gemini Nav Cards) */
        section[data-testid="stSidebar"] div.stButton > button {{
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            background: {nav_box_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 14px !important;
            padding: 13px 16px !important;
            margin-bottom: 6px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
            transition: all 0.18s cubic-bezier(0.2, 0, 0, 1) !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button:hover {{
            background: {nav_box_hover} !important;
            border-color: {accent} !important;
            transform: translateX(3px);
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background: {nav_active_bg} !important;
            border: 1.5px solid {nav_active_border} !important;
            color: {nav_active_text} !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.18) !important;
        }}

        /* Quick Navigation Bar di Atas Banner (Bisa klik langsung juga tanpa buka sidebar) */
        div[data-testid="stHorizontalBlock"].quick-nav-wrap div.stButton > button {{
            background: {bg_card} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            padding: 9px 12px !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            transition: all 0.18s ease !important;
        }}
        div[data-testid="stHorizontalBlock"].quick-nav-wrap div.stButton > button:hover {{
            border-color: {accent} !important;
            background: {bg_card_hover} !important;
        }}
        div[data-testid="stHorizontalBlock"].quick-nav-wrap div.stButton > button[kind="primary"] {{
            background: {nav_active_bg} !important;
            border: 1.5px solid {nav_active_border} !important;
            color: {nav_active_text} !important;
            font-weight: 700 !important;
        }}

        /* Top Banner */
        .top-banner {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 18px;
            padding: 20px 26px;
            margin-bottom: 18px;
            box-shadow: {shadow};
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .top-banner-title {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {text_primary};
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .top-banner-sub {{
            font-size: 0.88rem;
            color: {text_secondary};
            margin-top: 4px;
        }}
        .badge-pill {{
            background: {nav_box_bg};
            color: {text_primary};
            border: 1px solid {border_color};
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        /* KPI Cards */
        .kpi-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: {shadow};
            transition: transform 0.2s ease, border-color 0.2s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: {accent};
            background: {bg_card_hover};
        }}
        .kpi-label {{
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {text_secondary};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: {text_primary};
            line-height: 1.15;
        }}
        .kpi-delta {{
            font-size: 0.8rem;
            margin-top: 8px;
            font-weight: 500;
            color: {text_secondary};
        }}

        /* Section Card Container */
        .dash-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 18px;
            padding: 18px 22px;
            box-shadow: {shadow};
            margin-bottom: 16px;
        }}
        .card-header {{
            font-size: 1.02rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 4px;
        }}
        .card-sub {{
            font-size: 0.82rem;
            color: {text_secondary};
            margin-bottom: 12px;
        }}

        /* Custom Car Spec Cards */
        .car-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 18px;
            box-shadow: {shadow};
            margin-bottom: 14px;
            transition: all 0.2s ease;
        }}
        .car-card:hover {{
            border-color: {accent};
            transform: translateY(-3px);
            background: {bg_card_hover};
        }}
        .car-badge-vgood {{
            background: rgba(16, 185, 129, 0.16);
            color: #10B981;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
        }}
        .car-badge-good {{
            background: rgba(59, 130, 246, 0.16);
            color: #2563EB;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
        }}
        .spec-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 12px;
            font-size: 0.82rem;
        }}
        .spec-item {{
            background: {bg_main};
            padding: 6px 10px;
            border-radius: 10px;
            border: 1px solid {border_color};
            color: {text_secondary};
        }}
        .spec-item strong {{
            color: {text_primary};
        }}
    </style>
    """, unsafe_allow_html=True)


def gaya_plotly(fig, dark_mode: bool, height=340):
    font_color = "#E3E3E3" if dark_mode else "#1F2937"
    grid_color = "rgba(154, 160, 166, 0.14)" if dark_mode else "rgba(31, 41, 55, 0.08)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=font_color, size=12),
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
    )
    return fig


@st.cache_data
def muat_dataset(sumber, tahun):
    return AnalisisCar.baca_excel(sumber, tahun)


# Terapkan tema sesuai pilihan user
terapkan_tema_gemini(st.session_state.dark_mode)


# ==============================================================================
# SIDEBAR TASKBAR PREMIUM GAYA GEMINI (TIAP FITUR PUNYA KOTAK TERSENDIRI)
# ==============================================================================
NAV_ITEMS = [
    ("dashboard", "📊  Dashboard Utama"),
    ("ml", "🤖  Machine Learning"),
    ("explore", "📋  Eksplorasi & Filter Data"),
    ("catalog", "🚘  Katalog Rekomendasi"),
    ("compare", "📅  Perbandingan & Kesimpulan"),
]

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 4px 2px 12px 2px;">
        <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em;">✨ CarEval Suite</div>
        <div style="font-size: 0.78rem; opacity: 0.72;">Analytics & Decision Tree • {NAMA_KELOMPOK}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("MENU FITUR")
    for page_id, label in NAV_ITEMS:
        is_active = (st.session_state.active_page == page_id)
        if st.button(
            label,
            key=f"side_nav_{page_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            st.session_state.active_page = page_id
            st.rerun()

    st.divider()
    st.caption("TAMPILAN & DATASET")

    mode_label = "🌙 Gelap (Abu Hitam)" if st.session_state.dark_mode else "☀️ Terang (Putih Biru Abu)"
    new_dark = st.toggle(mode_label, value=st.session_state.dark_mode, key="side_theme_toggle")
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    with st.expander("📂 Kelola Dataset (.xlsx)", expanded=False):
        st.caption("Otomatis memakai `14_Car_Evaluation.xlsx` atau upload manual:")
        up1 = st.file_uploader("Dataset 2025", type=['xlsx'], key='up1')
        up2 = st.file_uploader("Dataset 2026", type=['xlsx'], key='up2')
        file_asli_upload = st.file_uploader("Dataset Asli (14_Car_Evaluation.xlsx)", type=['xlsx'], key='asli')

    with st.expander("🚀 Publikasi Online (GitHub)", expanded=False):
        st.caption("Tombol Deploy bawaan disembunyikan agar UI bersih. Untuk publikasi ke internet:")
        st.markdown(
            "1. Upload folder project ini ke **GitHub**.\n"
            "2. Buka [share.streamlit.io](https://share.streamlit.io) dan pilih repository Anda.\n"
            "3. Set main file: `car_evaluation_decision_tree.py`."
        )


# ==============================================================================
# PEMUATAN & VALIDASI DATA
# ==============================================================================
upload = {TAHUN_1: up1, TAHUN_2: up2}
sumber = {}
for tahun in [TAHUN_1, TAHUN_2]:
    if upload[tahun] is not None:
        sumber[tahun] = upload[tahun].getvalue()
    elif os.path.exists(FILE_LOKAL[tahun]):
        sumber[tahun] = FILE_LOKAL[tahun]

mode_simulasi = False
try:
    if len(sumber) == 2:
        daftar_df = [muat_dataset(sumber[t], t) for t in [TAHUN_1, TAHUN_2]]
    elif file_asli_upload is not None or os.path.exists(FILE_ASLI):
        data_asli = muat_dataset(
            file_asli_upload.getvalue() if file_asli_upload is not None else FILE_ASLI,
            'asli'
        )
        tengah = len(data_asli) // 2
        pertama = data_asli.iloc[:tengah].copy()
        kedua = data_asli.iloc[tengah:].copy()
        pertama['tahun'], kedua['tahun'] = TAHUN_1, TAHUN_2
        daftar_df = [pertama, kedua]
        mode_simulasi = True
    else:
        st.error("⚠️ File dataset `14_Car_Evaluation.xlsx` tidak ditemukan. Buka taskbar kiri (☰) → Kelola Dataset.")
        st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca Excel: {e}")
    st.stop()

analisis = AnalisisCar(AnalisisCar.gabung(daftar_df))
df = analisis.df
masalah = analisis.validasi()
if masalah:
    for pesan in masalah:
        st.error(pesan)
    st.stop()

nama_halaman_aktif = dict(NAV_ITEMS).get(st.session_state.active_page, "Dashboard Utama").strip()

# ==============================================================================
# TOP HEADER BANNER
# ==============================================================================
st.markdown(f"""
<div class="top-banner">
    <div>
        <div class="top-banner-title">Car Evaluation Analytics & Decision Tree</div>
        <div class="top-banner-sub">Klik ikon <strong>☰</strong> di pojok kiri atas untuk membuka/menutup taskbar kiri • Halaman aktif: <strong>{nama_halaman_aktif}</strong></div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="badge-pill">📁 {len(df):,} Records</span>
        <span class="badge-pill">👥 {NAMA_KELOMPOK}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if mode_simulasi:
    st.info("ℹ️ **Mode Simulasi Aktif:** Data `14_Car_Evaluation.xlsx` dibagi menjadi dua bagian seimbang (Paruh 1 = 2025, Paruh 2 = 2026).")


# ==============================================================================
# HALAMAN 1: DASHBOARD UTAMA
# ==============================================================================
if st.session_state.active_page == 'dashboard':
    total_mobil = len(df)
    n_unacc, p_unacc = analisis.persen_unacc()
    n_bagus = len(analisis.mobil_bagus())
    p_bagus = (n_bagus / total_mobil * 100) if total_mobil else 0
    n_high_unacc, n_high_total = analisis.safety_high_unacc()
    p_high_acc = ((n_high_total - n_high_unacc) / n_high_total * 100) if n_high_total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Kombinasi Mobil</div>
            <div class="kpi-value">{total_mobil:,}</div>
            <div class="kpi-delta">6 Atribut Evaluasi • 4 Kelas</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Unacceptable (unacc)</div>
            <div class="kpi-value" style="color:#EF4444;">{p_unacc:.1f}%</div>
            <div class="kpi-delta">{n_unacc:,} dari {total_mobil:,} spesifikasi</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Good & Very Good</div>
            <div class="kpi-value" style="color:#10B981;">{n_bagus:,}</div>
            <div class="kpi-delta">{p_bagus:.2f}% kategori kualitas tinggi</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Layak saat Safety High</div>
            <div class="kpi-value" style="color:#3B82F6;">{p_high_acc:.1f}%</div>
            <div class="kpi-delta">{n_high_total - n_high_unacc} dari {n_high_total} mobil safety=high</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        st.markdown('<div class="dash-card"><div class="card-header">Distribusi Kelas Berdasarkan Harga Beli (Buying)</div><div class="card-sub">Perbandingan jumlah kelayakan mobil pada tiap tingkat harga</div>', unsafe_allow_html=True)
        ct_buy = analisis.tabel_silang('buying').reset_index()
        ct_melt = ct_buy.melt(id_vars='buying', var_name='Kelas', value_name='Jumlah')
        fig_bar = px.bar(
            ct_melt,
            y='buying',
            x='Jumlah',
            color='Kelas',
            orientation='h',
            barmode='group',
            color_discrete_map=AnalisisCar.MAP_WARNA_KELAS,
            labels={'buying': 'Harga Beli', 'Jumlah': 'Jumlah Mobil'}
        )
        st.plotly_chart(gaya_plotly(fig_bar, st.session_state.dark_mode, height=310), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dash-card"><div class="card-header">Tren Kelayakan vs Tingkat Keselamatan (Safety)</div><div class="card-sub">Perubahan proporsi kelayakan seiring naiknya standar safety</div>', unsafe_allow_html=True)
        ct_saf = analisis.tabel_silang('safety', persen=True).reset_index()
        ct_saf_melt = ct_saf.melt(id_vars='safety', var_name='Kelas', value_name='Persentase')
        fig_line = px.line(
            ct_saf_melt,
            x='safety',
            y='Persentase',
            color='Kelas',
            markers=True,
            color_discrete_map=AnalisisCar.MAP_WARNA_KELAS,
            labels={'safety': 'Tingkat Safety', 'Persentase': 'Persentase (%)'}
        )
        fig_line.update_traces(line=dict(width=3), marker=dict(size=9))
        st.plotly_chart(gaya_plotly(fig_line, st.session_state.dark_mode, height=310), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 1.15])

    with col_b1:
        st.markdown('<div class="dash-card"><div class="card-header">Proporsi Kelas Evaluasi</div><div class="card-sub">Komposisi keseluruhan dataset Car Evaluation</div>', unsafe_allow_html=True)
        s_kelas = analisis.jumlah_kelas().reset_index()
        s_kelas.columns = ['Kelas', 'Jumlah']
        fig_donut = px.pie(
            s_kelas,
            values='Jumlah',
            names='Kelas',
            hole=0.58,
            color='Kelas',
            color_discrete_map=AnalisisCar.MAP_WARNA_KELAS
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(gaya_plotly(fig_donut, st.session_state.dark_mode, height=310), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b2:
        st.markdown('<div class="dash-card"><div class="card-header">Peta Konsentrasi: Biaya Perawatan vs Harga Beli</div><div class="card-sub">Ukuran titik menunjukkan jumlah mobil yang layak (acc, good, vgood)</div>', unsafe_allow_html=True)
        layak_df = df[df['class'] != 'unacc']
        grp = layak_df.groupby(['buying', 'maint', 'class']).size().reset_index(name='Jumlah')
        fig_scatter = px.scatter(
            grp,
            x='buying',
            y='maint',
            size='Jumlah',
            color='class',
            color_discrete_map=AnalisisCar.MAP_WARNA_KELAS,
            category_orders={
                'buying': AnalisisCar.URUTAN_LEVEL,
                'maint': AnalisisCar.URUTAN_LEVEL
            },
            size_max=34,
            labels={'buying': 'Harga Beli (buying)', 'maint': 'Biaya Perawatan (maint)'}
        )
        st.plotly_chart(gaya_plotly(fig_scatter, st.session_state.dark_mode, height=310), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# HALAMAN 2: MACHINE LEARNING (DECISION TREE)
# ==============================================================================
elif st.session_state.active_page == 'ml':
    st.markdown('<div class="dash-card"><div class="card-header">⚙️ Hyperparameter & Konfigurasi Model</div><div class="card-sub">Atur rasio data uji dan kompleksitas Decision Tree secara real-time</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    persen_uji = p1.slider('Data Uji / Test Split (%)', 10, 40, 20, 5)
    max_depth = p2.slider('Max Depth (Kedalaman Pohon)', 2, 20, 8, 1)
    leaf = p3.slider('Min Samples Leaf', 1, 20, 1, 1)
    st.markdown('</div>', unsafe_allow_html=True)

    model = ModelDecisionTree(
        df, test_size=persen_uji / 100, max_depth=max_depth, min_samples_leaf=leaf
    )
    metrik = model.evaluasi()

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="kpi-card"><div class="kpi-label">Test Accuracy</div><div class="kpi-value" style="color:#10B981;">{metrik["Test Accuracy"]*100:.2f}%</div><div class="kpi-delta">Train Acc: {metrik["Train Accuracy"]*100:.2f}%</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="kpi-card"><div class="kpi-label">Precision (Weighted)</div><div class="kpi-value">{metrik["Precision"]*100:.2f}%</div><div class="kpi-delta">Ketepatan prediksi positif</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="kpi-card"><div class="kpi-label">Recall (Weighted)</div><div class="kpi-value">{metrik["Recall"]*100:.2f}%</div><div class="kpi-delta">Sensitivitas deteksi kelas</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="kpi-card"><div class="kpi-label">F1-Score (Weighted)</div><div class="kpi-value" style="color:#3B82F6;">{metrik["F1-Score"]*100:.2f}%</div><div class="kpi-delta">Data Uji: {len(model.X_test):,} sampel</div></div>', unsafe_allow_html=True)

    st.write("")

    g1, g2, g3 = st.columns([1, 1.15, 1.15])

    with g1:
        st.markdown('<div class="dash-card"><div class="card-header">Model Accuracy Gauge</div><div class="card-sub">Skor akurasi pada data uji</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(metrik["Test Accuracy"] * 100, 2),
            delta={'reference': round(metrik["Train Accuracy"] * 100, 2), 'suffix': '% vs Train'},
            number={'suffix': '%', 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10B981"},
                'bgcolor': "rgba(154,160,166,0.15)",
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [60, 85], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
                ],
            }
        ))
        st.plotly_chart(gaya_plotly(fig_gauge, st.session_state.dark_mode, height=290), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="dash-card"><div class="card-header">Column Importance</div><div class="card-sub">Kontribusi setiap atribut dalam keputusan</div>', unsafe_allow_html=True)
        imp = model.feature_importance().reset_index()
        imp.columns = ['Fitur', 'Importance']
        fig_imp = px.bar(
            imp,
            x='Importance',
            y='Fitur',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#3B82F6', '#10B981']
        )
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(gaya_plotly(fig_imp, st.session_state.dark_mode, height=290), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g3:
        st.markdown('<div class="dash-card"><div class="card-header">Confusion Matrix</div><div class="card-sub">Aktual vs Prediksi pada data uji</div>', unsafe_allow_html=True)
        cm, labels = model.matriks_konfusi()
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(gaya_plotly(fig_cm, st.session_state.dark_mode, height=290), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sim_col, rep_col = st.columns([1.15, 1])

    with sim_col:
        st.markdown('<div class="dash-card"><div class="card-header">🔍 Simulasi Prediksi Kelayakan Mobil</div><div class="card-sub">Pilih spesifikasi mobil untuk memprediksi kelas dan probabilitasnya</div>', unsafe_allow_html=True)
        with st.form('form_prediksi_pro'):
            f1, f2, f3 = st.columns(3)
            in_buying = f1.selectbox('Harga Beli (buying)', AnalisisCar.URUTAN_LEVEL, index=0)
            in_maint = f2.selectbox('Biaya Perawatan (maint)', AnalisisCar.URUTAN_LEVEL, index=0)
            in_doors = f3.selectbox('Jumlah Pintu (doors)', ['2', '3', '4', '5more'], index=2)
            in_persons = f1.selectbox('Kapasitas (persons)', ['2', '4', 'more'], index=1)
            in_lug = f2.selectbox('Bagasi (lug_boot)', ['small', 'med', 'big'], index=2)
            in_safety = f3.selectbox('Keselamatan (safety)', AnalisisCar.URUTAN_SAFETY, index=2)
            submit_pred = st.form_submit_button('🚀 Jalankan Prediksi', type='primary', use_container_width=True)

        if submit_pred:
            input_mobil = {
                'buying': in_buying, 'maint': in_maint, 'doors': in_doors,
                'persons': in_persons, 'lug_boot': in_lug, 'safety': in_safety
            }
            kelas, peluang = model.predict(input_mobil)
            warna_res = AnalisisCar.MAP_WARNA_KELAS[kelas]
            st.markdown(f"""
            <div style="padding:14px; border-radius:12px; border:1px solid {warna_res}; background:rgba(59,130,246,0.06); margin-top:10px;">
                <span style="font-size:0.85rem; opacity:0.85;">Hasil Prediksi Model:</span><br>
                <strong style="font-size:1.35rem; color:{warna_res};">{kelas.upper()} — {AnalisisCar.LABEL_KELAS[kelas]}</strong>
            </div>
            """, unsafe_allow_html=True)
            df_prob = peluang.reset_index()
            df_prob.columns = ['Kelas', 'Probabilitas']
            fig_prob = px.bar(
                df_prob, x='Probabilitas', y='Kelas', orientation='h',
                color='Kelas', color_discrete_map=AnalisisCar.MAP_WARNA_KELAS
            )
            st.plotly_chart(gaya_plotly(fig_prob, st.session_state.dark_mode, height=200), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rep_col:
        st.markdown('<div class="dash-card"><div class="card-header">📑 Classification Report Detail</div><div class="card-sub">Metrik Precision, Recall, dan F1-Score per kelas</div>', unsafe_allow_html=True)
        st.dataframe(model.laporan().round(3), use_container_width=True, height=320)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🌳 Lihat Visualisasi Struktur Pohon Keputusan (Decision Tree)"):
        tampilan_depth = st.slider('Kedalaman level pohon yang ditampilkan', 1, 4, 3)
        fig_tree = model.tree_figure(max_plot_depth=tampilan_depth, dark_mode=st.session_state.dark_mode)
        st.pyplot(fig_tree)
        plt.close(fig_tree)


# ==============================================================================
# HALAMAN 3: EKSPLORASI DATA & FILTER
# ==============================================================================
elif st.session_state.active_page == 'explore':
    st.markdown('<div class="dash-card"><div class="card-header">🔎 Filter & Pencarian Spesifikasi Dataset</div><div class="card-sub">Gunakan filter di bawah untuk menyaring dataset secara interaktif</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    f_class = fc1.multiselect('Kelas Evaluasi', AnalisisCar.URUTAN_KELAS, default=AnalisisCar.URUTAN_KELAS)
    f_buying = fc2.multiselect('Harga Beli (buying)', AnalisisCar.URUTAN_LEVEL, default=AnalisisCar.URUTAN_LEVEL)
    f_safety = fc3.multiselect('Keselamatan (safety)', AnalisisCar.URUTAN_SAFETY, default=AnalisisCar.URUTAN_SAFETY)
    f_tahun = fc4.multiselect('Tahun Simulasi', analisis.daftar_tahun, default=analisis.daftar_tahun)

    df_filtered = df[
        df['class'].isin(f_class) &
        df['buying'].isin(f_buying) &
        df['safety'].isin(f_safety) &
        df['tahun'].isin(f_tahun)
    ]
    st.caption(f"Menampilkan **{len(df_filtered):,}** dari **{len(df):,}** baris data.")
    st.dataframe(df_filtered, use_container_width=True, height=380)
    st.markdown('</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown('<div class="dash-card"><div class="card-header">Crosstab: Harga Beli vs Kelas (Proporsi %)</div>', unsafe_allow_html=True)
        st.dataframe(analisis.tabel_silang('buying', persen=True).round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with t_col2:
        st.markdown('<div class="dash-card"><div class="card-header">Crosstab: Keselamatan vs Kelas (Jumlah)</div>', unsafe_allow_html=True)
        st.dataframe(analisis.tabel_silang('safety', persen=False), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# HALAMAN 4: KATALOG REKOMENDASI MOBIL
# ==============================================================================
elif st.session_state.active_page == 'catalog':
    st.markdown('<div class="dash-card"><div class="card-header">🚘 Katalog Mobil Terbaik (Good & Very Good)</div><div class="card-sub">Filter kombinasi spesifikasi ideal berdasarkan preferensi harga, keselamatan, dan kapasitas</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    p_harga = r1.selectbox('Filter Harga (buying)', ['Semua'] + AnalisisCar.URUTAN_LEVEL)
    p_safety = r2.selectbox('Filter Safety', ['Semua'] + AnalisisCar.URUTAN_SAFETY)
    p_persons = r3.selectbox('Kapasitas Penumpang', ['Semua', '2', '4', 'more'])
    p_tahun = r4.selectbox('Sumber Tahun', ['Semua'] + analisis.daftar_tahun)
    st.markdown('</div>', unsafe_allow_html=True)

    hasil_rek = analisis.rekomendasi(p_harga, p_safety, p_persons, p_tahun)

    if hasil_rek.empty:
        st.warning("Tidak ditemukan mobil berkategori Good atau Very Good dengan kombinasi filter tersebut.")
    else:
        st.success(f"Ditemukan **{len(hasil_rek)}** spesifikasi mobil unggulan (Good / Very Good).")
        tampil_kartu = hasil_rek.head(24).reset_index(drop=True)
        for i in range(0, len(tampil_kartu), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if i + idx < len(tampil_kartu):
                    row = tampil_kartu.iloc[i + idx]
                    badge_cls = "car-badge-vgood" if row['class'] == 'vgood' else "car-badge-good"
                    label_cls = "VERY GOOD ★" if row['class'] == 'vgood' else "GOOD ✓"
                    with col:
                        st.markdown(f"""
                        <div class="car-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:700; font-size:1rem;">Spesifikasi #{i+idx+1}</span>
                                <span class="{badge_cls}">{label_cls}</span>
                            </div>
                            <div style="font-size:0.78rem; opacity:0.7; margin-top:2px;">Tahun Simulasi: {row['tahun']}</div>
                            <div class="spec-grid">
                                <div class="spec-item">💰 Buying: <strong>{row['buying'].upper()}</strong></div>
                                <div class="spec-item">🔧 Maint: <strong>{row['maint'].upper()}</strong></div>
                                <div class="spec-item">🛡️ Safety: <strong>{row['safety'].upper()}</strong></div>
                                <div class="spec-item">👥 Persons: <strong>{row['persons']}</strong></div>
                                <div class="spec-item">🚪 Doors: <strong>{row['doors']}</strong></div>
                                <div class="spec-item">🧳 Bagasi: <strong>{row['lug_boot'].upper()}</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with st.expander("📄 Lihat Semua Hasil dalam Bentuk Tabel Lengkap"):
            st.dataframe(hasil_rek, use_container_width=True)


# ==============================================================================
# HALAMAN 5: PERBANDINGAN 2025 VS 2026 & KESIMPULAN
# ==============================================================================
elif st.session_state.active_page == 'compare':
    st.markdown(f'<div class="dash-card"><div class="card-header">📅 {AnalisisCar.JUDUL_PERBANDINGAN}</div><div class="card-sub">Parameter: {AnalisisCar.KRITERIA_A} dan penyempitan {AnalisisCar.KRITERIA_B}</div>', unsafe_allow_html=True)
    hasil_banding = analisis.kriteria_per_tahun()
    b_col1, b_col2 = st.columns([1.2, 1])

    with b_col1:
        df_b_plot = hasil_banding.copy()
        df_b_plot.index.name = 'Tahun'
        df_b_plot = df_b_plot.reset_index()
        df_b_melt = df_b_plot.melt(id_vars='Tahun', var_name='Kriteria', value_name='Jumlah')
        fig_comp = px.bar(
            df_b_melt, x='Tahun', y='Jumlah', color='Kriteria', barmode='group',
            text_auto=True, color_discrete_sequence=['#3B82F6', '#10B981']
        )
        st.plotly_chart(gaya_plotly(fig_comp, st.session_state.dark_mode, height=300), use_container_width=True)

    with b_col2:
        st.write("**Tabel Ringkasan Kriteria:**")
        st.dataframe(hasil_banding, use_container_width=True)
        st.write("**Rincian Kelas (buying=low & safety=high):**")
        st.dataframe(analisis.detail_kelas_per_tahun(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-card"><div class="card-header">📌 Kesimpulan Utama Analisis</div><div class="card-sub">Ringkasan temuan dari seluruh tahapan eksplorasi dan evaluasi</div>', unsafe_allow_html=True)
    for kalimat in analisis.kesimpulan():
        st.markdown(f"- {kalimat}")
    st.write("")
    st.download_button(
        label='📥 Unduh Laporan Rekap (.txt)',
        data=analisis.rekap_teks(),
        file_name='rekap_hasil_kelompok7.txt',
        mime='text/plain',
        type='primary'
    )
    st.markdown('</div>', unsafe_allow_html=True)
