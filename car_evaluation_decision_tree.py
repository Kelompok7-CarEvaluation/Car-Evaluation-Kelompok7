import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

NAMA_KELOMPOK = 'Kelompok 7'
TAHUN_1, TAHUN_2 = '2025', '2026'
FILE_LOKAL = {TAHUN_1: 'Car_Evaluation_2025.xlsx', TAHUN_2: 'Car_Evaluation_2026.xlsx'}
FILE_ASLI = '14_Car_Evaluation.xlsx'


class AnalisisCar:
    NAMA_KOLOM = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    FITUR = NAMA_KOLOM[:-1]
    URUTAN_KELAS = ['unacc', 'acc', 'good', 'vgood']
    URUTAN_LEVEL = ['low', 'med', 'high', 'vhigh']
    URUTAN_SAFETY = ['low', 'med', 'high']
    WARNA_KELAS = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
    KRITERIA_A = 'buying=low & safety=high'
    KRITERIA_B = 'buying=low & safety=high & class=acc'
    JUDUL_PERBANDINGAN = 'Perbandingan jumlah mobil yang memenuhi kriteria, 2025 vs 2026'
    CATATAN_SIMULASI = (
        'Catatan: pembagian 2025/2026 hanya simulasi. Dataset asli Car Evaluation '
        'tidak mencantumkan tahun atau penjualan. Angka yang dibandingkan adalah '
        'jumlah kombinasi spesifikasi yang memenuhi kriteria, bukan unit terjual.'
    )

    def __init__(self, df):
        self.df = df
        self.daftar_tahun = sorted(df['tahun'].unique())

    @staticmethod
    def baca_excel(sumber, tahun):
        if isinstance(sumber, bytes):
            sumber = io.BytesIO(sumber)
        df = pd.read_excel(sumber, header=None, names=AnalisisCar.NAMA_KOLOM,
                           engine='openpyxl').dropna(how='all')
        for kolom in AnalisisCar.NAMA_KOLOM:
            df[kolom] = (df[kolom].astype('string').str.strip().str.lower()
                         .str.replace(r'\.1$', '', regex=True))
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

    def baris_kembar(self):
        return int(self.df.duplicated(subset=self.NAMA_KOLOM, keep=False).sum())

    def jumlah_per_tahun(self):
        return self.df['tahun'].value_counts().sort_index()

    def jumlah_kelas(self):
        return self.df['class'].value_counts().reindex(self.URUTAN_KELAS, fill_value=0)

    def persen_unacc(self):
        n = int((self.df['class'] == 'unacc').sum())
        return n, n / len(self.df) * 100

    def _urutan(self, kolom):
        return self.URUTAN_SAFETY if kolom == 'safety' else self.URUTAN_LEVEL

    def tabel_silang(self, kolom, persen=False):
        tabel = pd.crosstab(self.df[kolom], self.df['class'],
                            normalize='index' if persen else False)
        if persen:
            tabel *= 100
        return tabel.reindex(index=self._urutan(kolom),
                             columns=self.URUTAN_KELAS, fill_value=0)

    def safety_high_unacc(self):
        tinggi = self.df[self.df['safety'] == 'high']
        return int((tinggi['class'] == 'unacc').sum()), len(tinggi)

    def mobil_bagus(self):
        return self.df[self.df['class'].isin(['good', 'vgood'])]

    def ciri_khas(self):
        bagus = self.mobil_bagus()
        return {k: bagus[k].value_counts() for k in self.FITUR}

    def safety_low(self):
        rendah = self.df[self.df['safety'] == 'low']
        return len(rendah), int((rendah['class'] == 'unacc').sum())

    def buying_low(self):
        murah = self.df[self.df['buying'] == 'low']
        n_unacc = int((murah['class'] == 'unacc').sum())
        return len(murah), n_unacc, n_unacc / len(murah) * 100 if len(murah) else 0.0

    def rekomendasi(self, harga, tahun='Semua'):
        pilih = self.df[(self.df['buying'] == harga) &
                        (self.df['class'].isin(['good', 'vgood']))].copy()
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
        return pd.DataFrame({self.KRITERIA_A: a['tahun'].value_counts(),
                             self.KRITERIA_B: b['tahun'].value_counts()}).reindex(
                                 self.daftar_tahun).fillna(0).astype(int)

    def detail_kelas_per_tahun(self):
        a = self.df[(self.df['buying'] == 'low') & (self.df['safety'] == 'high')]
        return pd.crosstab(a['tahun'], a['class']).reindex(
            index=self.daftar_tahun, columns=self.URUTAN_KELAS, fill_value=0)

    def sebaran_buying_per_tahun(self):
        return pd.crosstab(self.df['tahun'], self.df['buying']).reindex(
            index=self.daftar_tahun, columns=self.URUTAN_LEVEL, fill_value=0)

    def ringkas_perbandingan(self):
        return [f'Tahun {tahun}: {nilai[self.KRITERIA_A]} mobil memenuhi {self.KRITERIA_A}, '
                f'{nilai[self.KRITERIA_B]} di antaranya class=acc.'
                for tahun, nilai in self.kriteria_per_tahun().iterrows()]

    def penjelasan_perbedaan(self):
        isi = []
        for tahun in self.daftar_tahun:
            level = sorted(self.df[self.df['tahun'] == tahun]['buying'].unique(),
                           key=self.URUTAN_LEVEL.index)
            isi.append(f"{tahun}: {', '.join(level)}")
        return ('Isi kolom buying setiap file: ' + '; '.join(isi) +
                '. Perbedaan berasal dari pembagian dataset simulasi, bukan perubahan pasar.')

    def kesimpulan(self):
        total = len(self.df)
        baik = len(self.mobil_bagus())
        n_unacc, p_unacc = self.persen_unacc()
        n_low, n_low_unacc = self.safety_low()
        _, _, p_murah = self.buying_low()
        return [
            f'Dari {total:,} kombinasi mobil, {baik:,} ({baik / total * 100:.2f}%) '
            f'Good/Very Good dan {n_unacc:,} ({p_unacc:.2f}%) Unacceptable.',
            f'Dari {n_low:,} mobil safety rendah, {n_low_unacc:,} '
            f'({n_low_unacc / n_low * 100 if n_low else 0:.2f}%) Unacceptable.',
            f'Sebanyak {p_murah:.2f}% mobil buying=low tetap Unacceptable.'
        ] + self.ringkas_perbandingan() + [self.CATATAN_SIMULASI]

    def rekap_teks(self):
        n_unacc, p_unacc = self.persen_unacc()
        a, b = self.safety_high_unacc()
        n_low, n_low_unacc = self.safety_low()
        n_murah, murah_unacc, p_murah = self.buying_low()
        baris = [
            f'Total data: {len(self.df)} {self.jumlah_per_tahun().to_dict()}',
            f'Jumlah per kelas: {self.jumlah_kelas().to_dict()}',
            f'Persen unacc: {p_unacc:.2f}% ({n_unacc})',
            f'Safety=high -> unacc: {a} dari {b}',
            f'Good/vgood: {len(self.mobil_bagus())}',
            f'Safety=low -> unacc: {n_low_unacc} dari {n_low}',
            f'Buying=low -> unacc: {murah_unacc} dari {n_murah} ({p_murah:.2f}%)',
            '', self.JUDUL_PERBANDINGAN
        ] + self.ringkas_perbandingan() + ['', self.CATATAN_SIMULASI]
        return '\n'.join(baris) + '\n'

    def grafik_distribusi(self):
        jumlah = self.jumlah_kelas()
        fig, ax = plt.subplots()
        batang = ax.bar(jumlah.index, jumlah.values, color=self.WARNA_KELAS)
        ax.bar_label(batang)
        ax.set(title='Distribusi Kelas Mobil (gabungan)', xlabel='Kelas', ylabel='Jumlah mobil')
        return fig

    def grafik_tabel(self, tabel, judul, xlabel, ylabel):
        fig, ax = plt.subplots()
        tabel.plot(kind='bar', ax=ax, color=self.WARNA_KELAS)
        ax.set(title=judul, xlabel=xlabel, ylabel=ylabel)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title='Kelas', loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        return fig

    def grafik_perbandingan(self, hasil):
        fig, ax = plt.subplots()
        hasil.plot(kind='bar', ax=ax, color=['#5bc0de', '#5cb85c'])
        for wadah in ax.containers:
            ax.bar_label(wadah)
        ax.set(title='Jumlah Mobil Memenuhi Kriteria (simulasi)', xlabel='Tahun', ylabel='Jumlah')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title='Kriteria', loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        return fig


class ModelDecisionTree:
    """Data disiapkan dari gabungan kedua file; tahun tidak ikut memprediksi kelas."""

    def __init__(self, df, test_size=0.20, max_depth=8, min_samples_leaf=1):
        self.fitur = AnalisisCar.FITUR
        X = df[self.fitur].astype(str)
        y = df['class'].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.pipeline = Pipeline([
            ('encoder', ColumnTransformer([
                ('kategori', OneHotEncoder(handle_unknown='ignore'), self.fitur)
            ])),
            ('pohon', DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42))
        ])
        self.pipeline.fit(X_train, y_train)
        self.prediksi_test = self.pipeline.predict(X_test)

    def evaluasi(self):
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, self.prediksi_test, average='weighted', zero_division=0)
        return {'Accuracy': accuracy_score(self.y_test, self.prediksi_test),
                'Precision': precision, 'Recall': recall, 'F1-score': f1}

    def laporan(self):
        return pd.DataFrame(classification_report(
            self.y_test, self.prediksi_test, output_dict=True, zero_division=0)).T

    def confusion_figure(self):
        urutan = AnalisisCar.URUTAN_KELAS
        matriks = confusion_matrix(self.y_test, self.prediksi_test, labels=urutan)
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay(matriks, display_labels=urutan).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title('Confusion Matrix - Data Uji')
        fig.tight_layout()
        return fig

    def feature_importance(self):
        encoder = self.pipeline.named_steps['encoder']
        names = encoder.get_feature_names_out()
        importance = self.pipeline.named_steps['pohon'].feature_importances_
        hasil = pd.DataFrame({'encoded_feature': names, 'importance': importance})
        return pd.Series({f: float(hasil.loc[
            hasil['encoded_feature'].str.startswith(f'kategori__{f}_'),
            'importance'].sum()) for f in self.fitur}).sort_values(ascending=False)

    def tree_figure(self, max_plot_depth=3):
        names = self.pipeline.named_steps['encoder'].get_feature_names_out()
        clf = self.pipeline.named_steps['pohon']
        fig, ax = plt.subplots(figsize=(19, 9))
        plot_tree(clf, feature_names=names, class_names=clf.classes_,
                  filled=True, rounded=True, fontsize=8,
                  max_depth=max_plot_depth, ax=ax)
        fig.tight_layout()
        return fig

    def predict(self, input_dict):
        baru = pd.DataFrame([input_dict], columns=self.fitur)
        hasil = str(self.pipeline.predict(baru)[0])
        proba = self.pipeline.predict_proba(baru)[0]
        peluang = pd.Series(proba, index=self.pipeline.classes_).reindex(
            AnalisisCar.URUTAN_KELAS, fill_value=0) * 100
        return hasil, peluang


@st.cache_data
def muat_dataset(sumber, tahun):
    return AnalisisCar.baca_excel(sumber, tahun)


def tampilkan_grafik(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.set_page_config(
    page_title='Car Evaluation Dashboard + Decision Tree',
    page_icon='🚗',
    layout='wide',
)

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
    k4.metric(f'Data {TAHUN_1} / {TAHUN_2}',
              f'{per_tahun.get(TAHUN_1, 0):,} / {per_tahun.get(TAHUN_2, 0):,}')

    g1, g2 = st.columns(2)
    with g1:
        st.subheader('Distribusi Kelas Mobil')
        tampilkan_grafik(analisis.grafik_distribusi())
        st.dataframe(analisis.jumlah_kelas().rename('Jumlah Mobil'), use_container_width=True)
    with g2:
        st.subheader('Perbandingan Kriteria (2025 vs 2026)')
        hasil_banding = analisis.kriteria_per_tahun()
        tampilkan_grafik(analisis.grafik_perbandingan(hasil_banding))
        st.caption(AnalisisCar.CATATAN_SIMULASI)

    with st.expander('Lihat dataset gabungan dan nilai unik'):
        st.dataframe(df, use_container_width=True)
        for kolom in df.columns:
            st.write(f'**{kolom}**: {sorted(df[kolom].unique())}')

    st.subheader('Ringkasan & Kesimpulan')
    for kalimat in analisis.kesimpulan():
        st.write('- ' + kalimat)
    st.download_button(
        'Unduh rekap hasil (.txt)',
        data=analisis.rekap_teks(),
        file_name='rekap_hasil.txt',
    )

elif menu == 'Analisis Data':
    st.header('🔎 Analisis Data & Eksplorasi')
    sub1, sub2, sub3, sub4 = st.tabs([
        'Crosstab Harga & Safety',
        'Filter Good / VGood',
        'Analisis Harga Murah',
        'Rekomendasi & 2025 vs 2026',
    ])

    with sub1:
        st.subheader('Crosstab buying vs class')
        tabel_buying = analisis.tabel_silang('buying')
        st.dataframe(tabel_buying, use_container_width=True)
        tampilkan_grafik(analisis.grafik_tabel(
            tabel_buying, 'Harga Beli vs Kelas', 'Harga beli', 'Jumlah mobil'
        ))

        st.subheader('Crosstab safety vs class')
        tabel_safety = analisis.tabel_silang('safety')
        st.dataframe(tabel_safety, use_container_width=True)
        tampilkan_grafik(analisis.grafik_tabel(
            tabel_safety, 'Keselamatan vs Kelas', 'Safety', 'Jumlah mobil'
        ))
        n, total = analisis.safety_high_unacc()
        st.info(f'Safety=high yang unacc: {n} dari {total} ({n / total * 100 if total else 0:.2f}%).')

    with sub2:
        st.subheader('Mobil good atau vgood')
        bagus = analisis.mobil_bagus()
        st.write(f'Jumlah: **{len(bagus)}** mobil.')
        st.dataframe(bagus, use_container_width=True)

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

        st.divider()
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

    st.subheader('Evaluasi model')
    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.write('**Classification Report**')
        st.dataframe(model.laporan().round(3), use_container_width=True)
    with kanan:
        tampilkan_grafik(model.confusion_figure())

    st.subheader('Feature Importance')
    penting = model.feature_importance()
    st.bar_chart(penting)
    st.dataframe(penting.rename('Importance').to_frame(), use_container_width=True)

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
        st.success(f'Prediksi kelas: **{kelas.upper()}** - {keterangan[kelas]}')
        st.write('**Estimasi probabilitas per kelas (menurut model):**')
        st.bar_chart(probabilitas)
        st.dataframe(probabilitas.round(2).rename('Probabilitas (%)').to_frame())
        st.caption(
            'Prediksi adalah keluaran model dari dataset evaluasi sintetis, '
            'bukan inspeksi keselamatan atau rekomendasi membeli mobil nyata.'
        )

st.divider()
st.caption(f'Tugas Besar Python - {NAMA_KELOMPOK}')
