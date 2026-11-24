import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

TAHUN_1 = 2025
TAHUN_2 = 2026
FILE_LOKAL = {
    TAHUN_1: f'Car_Evaluation_{TAHUN_1}.xlsx',
    TAHUN_2: f'Car_Evaluation_{TAHUN_2}.xlsx',
}
FILE_ASLI = '14_Car_Evaluation.xlsx'
NAMA_KELOMPOK = 'Kelompok 7'


@st.cache_data(show_spinner=False)
def muat_dataset(sumber, label_sumber):
    if isinstance(sumber, bytes):
        df = pd.read_excel(io.BytesIO(sumber))
    else:
        df = pd.read_excel(sumber)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if isinstance(label_sumber, int):
        df['tahun'] = label_sumber
    elif 'tahun' not in df.columns:
        df['tahun'] = TAHUN_1
    return df


def tampilkan_grafik(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


class AnalisisCar:
    KOLOM_WAJIB = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    URUTAN_LEVEL = ['low', 'med', 'high', 'vhigh']
    URUTAN_SAFETY = ['low', 'med', 'high']
    URUTAN_KELAS = ['unacc', 'acc', 'good', 'vgood']
    CATATAN_SIMULASI = (
        'Catatan: Data 2025 & 2026 merupakan hasil pembagian simulasi dari dataset asli '
        '(paruh pertama = 2025, paruh kedua = 2026), bukan data tahun sebenarnya.'
    )
    JUDUL_PERBANDINGAN = 'Perbandingan Kriteria (2025 vs 2026)'

    def __init__(self, df):
        self.df = df.copy()
        for col in self.KOLOM_WAJIB:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        self.daftar_tahun = sorted(self.df['tahun'].dropna().unique().tolist())

    @classmethod
    def gabung(cls, daftar_df):
        return pd.concat(daftar_df, ignore_index=True)

    def validasi(self):
        masalah = []
        kurang = [k for k in self.KOLOM_WAJIB if k not in self.df.columns]
        if kurang:
            masalah.append(f'Kolom wajib tidak ditemukan: {", ".join(kurang)}')
        if self.df.empty:
            masalah.append('Dataset kosong.')
        return masalah

    def baris_kembar(self):
        return int(self.df.duplicated(subset=self.KOLOM_WAJIB).sum())

    def jumlah_per_tahun(self):
        return self.df['tahun'].value_counts().sort_index().to_dict()

    def jumlah_kelas(self):
        s = self.df['class'].value_counts()
        urut = [k for k in self.URUTAN_KELAS if k in s.index]
        sisa = [k for k in s.index if k not in urut]
        return s.reindex(urut + sisa, fill_value=0)

    def persen_unacc(self):
        total = len(self.df)
        n_unacc = int((self.df['class'] == 'unacc').sum())
        persen = (n_unacc / total * 100) if total else 0.0
        return n_unacc, persen

    def mobil_bagus(self):
        return self.df[self.df['class'].isin(['good', 'vgood'])].copy()

    def tabel_silang(self, kolom, persen=False):
        ct = pd.crosstab(self.df[kolom], self.df['class'])
        urut_baris = [k for k in self.URUTAN_LEVEL if k in ct.index]
        if kolom == 'safety':
            urut_baris = [k for k in self.URUTAN_SAFETY if k in ct.index]
        if urut_baris:
            sisa = [k for k in ct.index if k not in urut_baris]
            ct = ct.reindex(urut_baris + sisa)
        urut_kol = [k for k in self.URUTAN_KELAS if k in ct.columns]
        if urut_kol:
            sisa_k = [k for k in ct.columns if k not in urut_kol]
            ct = ct[urut_kol + sisa_k]
        if persen:
            ct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
        return ct

    def safety_high_unacc(self):
        sub = self.df[self.df['safety'] == 'high']
        return int((sub['class'] == 'unacc').sum()), len(sub)

    def safety_low(self):
        sub = self.df[self.df['safety'] == 'low']
        return len(sub), int((sub['class'] == 'unacc').sum())

    def buying_low(self):
        sub = self.df[self.df['buying'] == 'low']
        total = len(sub)
        n_unacc = int((sub['class'] == 'unacc').sum())
        persen = (n_unacc / total * 100) if total else 0.0
        return total, n_unacc, persen

    def ciri_khas(self):
        bagus = self.mobil_bagus()
        hasil = {}
        for col in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
            hasil[col] = bagus[col].value_counts()
        return hasil

    def rekomendasi(self, buying_level, tahun_pilihan='Semua'):
        sub = self.mobil_bagus()
        sub = sub[sub['buying'] == buying_level]
        if tahun_pilihan != 'Semua':
            sub = sub[sub['tahun'] == tahun_pilihan]
        return sub.reset_index(drop=True)

    def kriteria_per_tahun(self):
        rows = []
        for t in self.daftar_tahun:
            sub = self.df[self.df['tahun'] == t]
            c1 = int(((sub['buying'] == 'low') & (sub['safety'] == 'high')).sum())
            c2 = int((
                (sub['buying'] == 'low')
                & (sub['safety'] == 'high')
                & (sub['class'] == 'acc')
            ).sum())
            rows.append({
                'Tahun': t,
                'buying=low & safety=high': c1,
                'buying=low & safety=high & class=acc': c2,
            })
        return pd.DataFrame(rows).set_index('Tahun')

    def detail_kelas_per_tahun(self):
        sub = self.df[(self.df['buying'] == 'low') & (self.df['safety'] == 'high')]
        return pd.crosstab(sub['tahun'], sub['class'])

    def sebaran_buying_per_tahun(self):
        ct = pd.crosstab(self.df['tahun'], self.df['buying'])
        urut = [k for k in self.URUTAN_LEVEL if k in ct.columns]
        return ct[urut] if urut else ct

    def ringkas_perbandingan(self):
        df_b = self.kriteria_per_tahun()
        pesan = []
        for tahun, row in df_b.iterrows():
            pesan.append(
                f"Tahun {tahun}: terdapat {int(row['buying=low & safety=high'])} mobil dengan "
                f"buying=low & safety=high, dan {int(row['buying=low & safety=high & class=acc'])} "
                f"di antaranya berkelas acc."
            )
        return pesan

    def penjelasan_perbedaan(self):
        return (
            'Pada dataset evaluasi mobil ini, kelas tidak hanya ditentukan oleh buying dan safety, '
            'tetapi juga dipengaruhi oleh maint, doors, persons, dan lug_boot (serta pembagian simulasi data).'
        )

    def grafik_distribusi(self):
        s = self.jumlah_kelas()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(s.index, s.values, color='#38bdf8')
        ax.set_title('Distribusi Kelas Mobil (Gabungan)')
        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah mobil')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    def grafik_perbandingan(self, df_banding):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        df_banding.plot(kind='bar', ax=ax, color=['#38bdf8', '#34d399'])
        ax.set_title('Jumlah Mobil Memenuhi Kriteria (Simulasi)')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Jumlah mobil')
        ax.legend(title='Kriteria', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    def grafik_tabel(self, tabel, judul, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        tabel.plot(kind='bar', ax=ax)
        ax.set_title(judul)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title='Kelas', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    def kesimpulan(self):
        _, p_unacc = self.persen_unacc()
        n_baik = len(self.mobil_bagus())
        n_murah, n_murah_unacc, p_murah_unacc = self.buying_low()
        return [
            f'Sebagian besar mobil dalam dataset berkelas unacc ({p_unacc:.2f}%).',
            f'Terdapat {n_baik} mobil dengan kategori kualitas Good atau Very Good.',
            f'Dari {n_murah} mobil dengan buying=low, sebanyak {n_murah_unacc} ({p_murah_unacc:.2f}%) masih tergolong unacc.',
            'Keselamatan (safety) dan kapasitas penumpang (persons) sangat berpengaruh pada kelayakan mobil.',
        ]

    def rekap_teks(self):
        lines = [
            f'REKAP ANALISIS CAR EVALUATION - {NAMA_KELOMPOK}',
            f'Total data: {len(self.df)}',
            '',
            'Ringkasan:',
        ]
        for k in self.kesimpulan():
            lines.append(f'- {k}')
        lines.append('')
        lines.append(self.CATATAN_SIMULASI)
        return '\n'.join(lines)


class ModelDecisionTree:
    FITUR = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    def __init__(self, df, test_size=0.2, max_depth=8, min_samples_leaf=1, random_state=42):
        self.X_raw = df[self.FITUR].astype(str)
        self.y_raw = df['class'].astype(str)
        self.X_encoded = pd.get_dummies(self.X_raw, drop_first=False)
        self.kolom_fitur = self.X_encoded.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_encoded,
            self.y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_raw,
        )
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

    def evaluasi(self):
        return {
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Precision (Weighted)': precision_score(
                self.y_test, self.y_pred, average='weighted', zero_division=0
            ),
            'Recall (Weighted)': recall_score(
                self.y_test, self.y_pred, average='weighted', zero_division=0
            ),
            'F1-Score (Weighted)': f1_score(
                self.y_test, self.y_pred, average='weighted', zero_division=0
            ),
        }

    def laporan(self):
        lap = classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)
        return pd.DataFrame(lap).transpose()

    def confusion_figure(self):
        label_kelas = sorted(self.y_raw.unique().tolist())
        cm = confusion_matrix(self.y_test, self.y_pred, labels=label_kelas)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(label_kelas)))
        ax.set_yticks(range(len(label_kelas)))
        ax.set_xticklabels(label_kelas)
        ax.set_yticklabels(label_kelas)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix')
        for i in range(len(label_kelas)):
            for j in range(len(label_kelas)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    def feature_importance(self):
        imp = pd.Series(self.clf.feature_importances_, index=self.kolom_fitur)
        kelompok = {}
        for f in self.FITUR:
            cocok = [k for k in self.kolom_fitur if k.startswith(f + '_')]
            kelompok[f] = float(imp[cocok].sum()) if cocok else 0.0
        return pd.Series(kelompok).sort_values(ascending=False)

    def tree_figure(self, max_plot_depth=3):
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(
            self.clf,
            max_depth=max_plot_depth,
            feature_names=self.kolom_fitur,
            class_names=self.clf.classes_.tolist(),
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
        )
        plt.tight_layout()
        return fig

    def predict(self, input_dict):
        df_in = pd.DataFrame([input_dict])[self.FITUR].astype(str)
        enc_in = pd.get_dummies(df_in).reindex(columns=self.kolom_fitur, fill_value=0)
        pred_kelas = self.clf.predict(enc_in)[0]
        probs = self.clf.predict_proba(enc_in)[0]
        seri_prob = pd.Series(probs * 100, index=self.clf.classes_).sort_values(ascending=False)
        return pred_kelas, seri_prob


st.set_page_config(
    page_title='Car Evaluation Dashboard + Decision Tree',
    page_icon='🚗',
    layout='wide',
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 14px 18px;
        border-radius: 12px;
    }
    .badge-kelas {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        background-color: #0f172a;
        border: 1px solid #38bdf8;
        color: #7dd3fc;
        margin: 6px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title('🚗 Navigasi')
    menu = st.radio(
        'Pilih halaman:',
        ['Dashboard Utama', 'Analisis Data', 'Machine Learning', 'Prediksi Mobil'],
    )
    st.divider()
    st.subheader('📂 Sumber Dataset')
    upload = {
        TAHUN_1: st.file_uploader('Dataset 2025 (.xlsx)', type=['xlsx'], key='up1'),
        TAHUN_2: st.file_uploader('Dataset 2026 (.xlsx)', type=['xlsx'], key='up2'),
    }
    file_asli_upload = st.file_uploader(
        'Alternatif: 14_Car_Evaluation.xlsx', type=['xlsx'], key='asli'
    )

st.title('🚗 Car Evaluation Dashboard + Decision Tree')
st.caption('Analisis harga, keselamatan, kualitas mobil, dan prediksi kelas dengan Decision Tree.')

sumber = {}
for tahun in [TAHUN_1, TAHUN_2]:
    if upload[tahun] is not None:
        sumber[tahun] = upload[tahun].getvalue()
    elif os.path.exists(FILE_LOKAL[tahun]):
        sumber[tahun] = FILE_LOKAL[tahun]

try:
    if len(sumber) == 2:
        daftar_df = [muat_dataset(sumber[t], t) for t in [TAHUN_1, TAHUN_2]]
    elif file_asli_upload is not None or os.path.exists(FILE_ASLI):
        data_asli = muat_dataset(
            file_asli_upload.getvalue() if file_asli_upload is not None else FILE_ASLI,
            'asli',
        )
        tengah = len(data_asli) // 2
        pertama = data_asli.iloc[:tengah].copy()
        kedua = data_asli.iloc[tengah:].copy()
        pertama['tahun'], kedua['tahun'] = TAHUN_1, TAHUN_2
        daftar_df = [pertama, kedua]
        st.warning(
            'Mode simulasi otomatis: data asli dibagi menurut urutan baris '
            '(paruh pertama = 2025, paruh kedua = 2026), BUKAN data tahun sebenarnya.'
        )
    else:
        st.warning('Upload kedua file tahunan atau file asli 14_Car_Evaluation.xlsx di sidebar.')
        st.stop()
except Exception as e:
    st.error(f'Terjadi kesalahan saat membaca Excel: {e}')
    st.stop()

analisis = AnalisisCar(AnalisisCar.gabung(daftar_df))
df = analisis.df
masalah = analisis.validasi()
if masalah:
    for pesan in masalah:
        st.error(pesan)
    st.stop()
if analisis.baris_kembar():
    st.warning(
        f'Ada {analisis.baris_kembar()} baris spesifikasi identik. '
        'Periksa sumber data jika ini tidak disengaja.'
    )

st.divider()

if menu == 'Dashboard Utama':
    st.header('📊 Dashboard Utama')
    per_tahun = analisis.jumlah_per_tahun()
    n_unacc, p_unacc = analisis.persen_unacc()
    n_baik = len(analisis.mobil_bagus())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Total Data Gabungan', f'{len(df):,}')
    k2.metric('Unacceptable (unacc)', f'{p_unacc:.2f}%', f'{n_unacc:,} data')
    k3.metric('Good / Very Good', f'{n_baik:,}', f'{n_baik / len(df) * 100:.2f}%')
    k4.metric(
        f'Data {TAHUN_1} / {TAHUN_2}',
        f'{per_tahun.get(TAHUN_1, 0):,} / {per_tahun.get(TAHUN_2, 0):,}',
    )

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.subheader('Distribusi Kelas Mobil')
            tampilkan_grafik(analisis.grafik_distribusi())
            st.dataframe(analisis.jumlah_kelas().rename('Jumlah Mobil'), use_container_width=True)
    with g2:
        with st.container(border=True):
            st.subheader('Perbandingan Kriteria (2025 vs 2026)')
            hasil_banding = analisis.kriteria_per_tahun()
            tampilkan_grafik(analisis.grafik_perbandingan(hasil_banding))
            st.caption(AnalisisCar.CATATAN_SIMULASI)

    with st.container(border=True):
        st.subheader('Ringkasan & Kesimpulan')
        for kalimat in analisis.kesimpulan():
            st.write('- ' + kalimat)
        st.download_button(
            'Unduh rekap hasil (.txt)',
            data=analisis.rekap_teks(),
            file_name='rekap_hasil.txt',
        )

    with st.expander('Lihat dataset gabungan dan nilai unik'):
        st.dataframe(df, use_container_width=True)
        for kolom in df.columns:
            st.write(f'**{kolom}**: {sorted(df[kolom].unique())}')

elif menu == 'Analisis Data':
    st.header('🔎 Analisis Data & Eksplorasi')
    sub1, sub2, sub3, sub4 = st.tabs([
        'Crosstab Harga & Safety',
        'Filter Good / VGood',
        'Analisis Harga Murah',
        'Rekomendasi & 2025 vs 2026',
    ])

    with sub1:
        with st.container(border=True):
            st.subheader('Crosstab buying vs class')
            tabel_buying = analisis.tabel_silang('buying')
            st.dataframe(tabel_buying, use_container_width=True)
            tampilkan_grafik(analisis.grafik_tabel(
                tabel_buying, 'Harga Beli vs Kelas', 'Harga beli', 'Jumlah mobil'
            ))
        with st.container(border=True):
            st.subheader('Crosstab safety vs class')
            tabel_safety = analisis.tabel_silang('safety')
            st.dataframe(tabel_safety, use_container_width=True)
            tampilkan_grafik(analisis.grafik_tabel(
                tabel_safety, 'Keselamatan vs Kelas', 'Safety', 'Jumlah mobil'
            ))
            n, total = analisis.safety_high_unacc()
            st.info(f'Safety=high yang unacc: {n} dari {total} ({n / total * 100 if total else 0:.2f}%).')

    with sub2:
        with st.container(border=True):
            st.subheader('Mobil good atau vgood')
            bagus = analisis.mobil_bagus()
            st.write(f'Jumlah: **{len(bagus)}** mobil.')
            st.dataframe(bagus, use_container_width=True)

        with st.container(border=True):
            st.subheader('Ciri khas good/vgood')
            daftar = list(analisis.ciri_khas().items())
            for mulai in range(0, len(daftar), 3):
                kolom_ui = st.columns(3)
                for ui, (nama, sebaran) in zip(kolom_ui, daftar[mulai:mulai + 3]):
                    ui.write(f'**{nama}**')
                    ui.dataframe(sebaran.rename('Jumlah'), use_container_width=True)

            n_low, n_unacc = analisis.safety_low()
            st.info(
                f'Safety low: {n_unacc} dari {n_low} unacc '
                f'({n_unacc / n_low * 100 if n_low else 0:.2f}%).'
            )

    with sub3:
        with st.container(border=True):
            st.subheader('Mobil murah (buying=low)')
            n, n_unacc, persen = analisis.buying_low()
            st.info(f'Dari {n} mobil murah, {n_unacc} unacc ({persen:.2f}%).')
            st.write(
                '**Jawaban:** Mobil murah belum tentu acceptable.'
                if n_unacc else '**Jawaban:** Semua mobil murah pada data ini acceptable.'
            )
            proporsi = analisis.tabel_silang('buying', persen=True)
            st.dataframe(proporsi.round(2), use_container_width=True)
            tampilkan_grafik(analisis.grafik_tabel(
                proporsi, 'Persentase Kelas per Harga', 'Harga beli', 'Persentase (%)'
            ))

    with sub4:
        with st.container(border=True):
            st.subheader('Rekomendasi mobil (good / vgood)')
            kiri, kanan = st.columns(2)
            pilihan = kiri.selectbox('Pilih kategori harga:', analisis.URUTAN_LEVEL)
            pilihan_tahun = kanan.selectbox('Pilih sumber tahun:', ['Semua'] + analisis.daftar_tahun)
            hasil = analisis.rekomendasi(pilihan, pilihan_tahun)
            if hasil.empty:
                st.warning('Tidak ada mobil Good atau Very Good pada pilihan ini.')
            else:
                st.success(f'Ditemukan {len(hasil)} pilihan good/vgood.')
                st.dataframe(hasil, use_container_width=True)

        with st.container(border=True):
            st.subheader(AnalisisCar.JUDUL_PERBANDINGAN)
            st.warning(AnalisisCar.CATATAN_SIMULASI)
            hasil_banding = analisis.kriteria_per_tahun()
            st.dataframe(hasil_banding, use_container_width=True)
            tampilkan_grafik(analisis.grafik_perbandingan(hasil_banding))
            st.write('Rincian kelas pada buying=low & safety=high:')
            st.dataframe(analisis.detail_kelas_per_tahun(), use_container_width=True)
            st.write('Sebaran buying tiap tahun:')
            st.dataframe(analisis.sebaran_buying_per_tahun(), use_container_width=True)
            for kalimat in analisis.ringkas_perbandingan():
                st.write('- ' + kalimat)
            st.info(analisis.penjelasan_perbedaan())

elif menu == 'Machine Learning':
    st.header('🤖 Decision Tree Classifier')
    st.write(
        'Model memprediksi **class** dari enam atribut mobil. '
        'Data dipecah menjadi latih/uji dengan stratifikasi kelas dan random_state=42.'
    )
    with st.container(border=True):
        opsi1, opsi2, opsi3 = st.columns(3)
        persen_uji = opsi1.slider('Data uji (%)', min_value=10, max_value=40, value=20, step=5)
        max_depth = opsi2.slider('Kedalaman maksimum tree', min_value=2, max_value=25, value=8, step=1)
        leaf = opsi3.slider('Minimal sampel per leaf', min_value=1, max_value=20, value=1, step=1)

    try:
        model = ModelDecisionTree(
            df, test_size=persen_uji / 100, max_depth=max_depth, min_samples_leaf=leaf
        )
    except ValueError as e:
        st.error(f'Gagal melatih Decision Tree: {e}')
        st.stop()

    st.caption(
        f'Data latih: {len(model.X_train):,} | '
        f'Data uji: {len(model.X_test):,} | '
        'Encoder: One-Hot | Algoritma: DecisionTreeClassifier'
    )
    hasil_metrik = model.evaluasi()
    koloms = st.columns(4)
    for kolom, (nama, nilai) in zip(koloms, hasil_metrik.items()):
        kolom.metric(nama, f'{nilai * 100:.2f}%')

    with st.container(border=True):
        st.subheader('Evaluasi model')
        kiri, kanan = st.columns([1, 1])
        with kiri:
            st.write('**Classification Report**')
            st.dataframe(model.laporan().round(3), use_container_width=True)
        with kanan:
            tampilkan_grafik(model.confusion_figure())

    with st.container(border=True):
        st.subheader('Feature Importance')
        penting = model.feature_importance()
        st.bar_chart(penting)
        st.dataframe(penting.rename('Importance').to_frame(), use_container_width=True)

    with st.container(border=True):
        st.subheader('Visualisasi Decision Tree')
        st.caption(
            'Agar terbaca, tampilan pohon dibatasi hingga beberapa level; '
            'model tetap memanfaatkan kedalaman yang dipilih di atas.'
        )
        tampilan_depth = st.slider('Kedalaman pohon yang ditampilkan', 1, 4, 3)
        tampilkan_grafik(model.tree_figure(max_plot_depth=tampilan_depth))

elif menu == 'Prediksi Mobil':
    st.header('🔮 Prediksi Kualitas Mobil')
    model = ModelDecisionTree(df)

    with st.container(border=True):
        with st.form('form_prediksi'):
            kiri, kanan, ketiga = st.columns(3)
            input_mobil = {
                'buying': kiri.selectbox('Harga beli (buying)', AnalisisCar.URUTAN_LEVEL),
                'maint': kanan.selectbox('Biaya perawatan (maint)', AnalisisCar.URUTAN_LEVEL),
                'doors': ketiga.selectbox('Jumlah pintu (doors)', ['2', '3', '4', '5more']),
                'persons': kiri.selectbox('Kapasitas penumpang (persons)', ['2', '4', 'more']),
                'lug_boot': kanan.selectbox('Bagasi (lug_boot)', ['small', 'med', 'big']),
                'safety': ketiga.selectbox('Keselamatan (safety)', AnalisisCar.URUTAN_SAFETY),
            }
            tombol_prediksi = st.form_submit_button('Prediksi Kelas Mobil', type='primary')

    if tombol_prediksi:
        kelas, probabilitas = model.predict(input_mobil)
        keterangan = {
            'unacc': 'Unacceptable (tidak layak)',
            'acc': 'Acceptable (cukup layak)',
            'good': 'Good (baik)',
            'vgood': 'Very Good (sangat baik)',
        }
        with st.container(border=True):
            st.markdown(
                f'<span class="badge-kelas">{kelas.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.success(f'Prediksi kelas: **{kelas.upper()}** — {keterangan[kelas]}')
            st.write('**Estimasi probabilitas per kelas (menurut model):**')
            st.bar_chart(probabilitas)
            st.dataframe(probabilitas.round(2).rename('Probabilitas (%)').to_frame())
            st.caption(
                'Prediksi adalah keluaran model dari dataset evaluasi sintetis, '
                'bukan inspeksi keselamatan atau rekomendasi membeli mobil nyata.'
            )

st.divider()
st.caption(f'Tugas Besar Python - {NAMA_KELOMPOK}')
