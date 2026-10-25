# Studio 2 — Klasifikasi Sampah · Kelompok 18 / Learning By Google

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Faiz Akbar Al Kalabadzi|123140091|Preprocessing dan Hyperparameter Tuning|
| 2  | Ahmad Aufa Mahdi Salam |123140092|Perancang Arsitektur|
| 3  | M. Reyshandi           |123140037|Training|

## Ringkasan Pendekatan

- **Arsitektur / backbone**: MobileNetV3-Large dengan pendekatan transfer learning berbasis bobot pretrained ImageNet V2.
- **Strategi pelatihan**: Model menerapkan full fine-tuning tanpa freezing backbone, dengan ptimizer AdamW dan learning rate konstan sebesar 9 × 10⁻⁶ selama 20 epoch. Fungsi loss yang digunakan adalah Cross Entropy dengan label smoothing sebesar 0,1. Scheduler learning rate tidak digunakan. Augmentasi data training meliputi Random Resized Crop berukuran 224 × 224 dengan skala 0,5–1,0, Random Horizontal Flip, Random Vertical Flip dengan probabilitas 0,1, Color Jitter, dan Random Rotation sebesar ±15°, dilanjutkan normalisasi menggunakan statistik ImageNet. Pelatihan dijalankan menggunakan GPU CUDA dengan mixed precision.
- **Data tambahan**: tidak ada
- **Ukuran input & preprocessing**: ukuran input gambar sebesar 224 × 224 piksel dengan 3 kanal warna RGB. Sebelum diproses oleh model, gambar melalui tahap preprocessing dan augmentasi, yang meliputi Random Resized Crop, Random Horizontal Flip, Random Vertical Flip, Color Jitter, rotasi hingga 15°, konversi gambar menjadi tensor, serta normalisasi menggunakan nilai mean dan standard deviation ImageNet.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | |0.9028|0.8857|
| Preview 2 (21 Okt) | |0.9028|0.8857|
| Final (25 Okt)     | |0.9458| |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
