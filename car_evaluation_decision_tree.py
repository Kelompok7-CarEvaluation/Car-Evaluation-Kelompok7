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
        ax.legend(title='Kelas
