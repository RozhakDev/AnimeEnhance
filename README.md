# AnimeEnhance - Anime Image Super-Resolution Pipeline (4x)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20L4%20(24GB)-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Model](https://img.shields.io/badge/Model-RealESRGAN__x4plus__anime__6B-orange.svg)](https://github.com/xinntao/Real-ESRGAN)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AnimeEnhance adalah pipeline PyTorch untuk memperbesar resolusi gambar anime 4x menggunakan model Real-ESRGAN. Gambar diperbesar dengan menjaga ketajaman garis (*line art*) dan warna asli tanpa modul tambahan yang merusak wajah karakter.

## Overview

Gambar anime beresolusi rendah sering kali pecah atau buram saat diperbesar dengan metode konvensional. AnimeEnhance mengatasi kendala tersebut dengan merekonstruksi gambar hingga skala 4x menggunakan model deep learning Real-ESRGAN (RRDBNet 6 blok). Pipeline ini berfokus mempertahankan ketajaman garis gambar (*line art*), detail mata, dan konsistensi warna asli tanpa menambahkan distorsi sintetis pada karakter.

Sistem bekerja dengan prinsip pemrosesan non-destruktif di mana file sumber diperlakukan secara *read-only*. Alur restorasi memadukan penyimpanan disk lokal Google Colab untuk komputasi cepat dan Google Drive untuk penyimpanan permanen. Untuk mencegah kehabisan memori GPU pada gambar berdimensi besar, sistem menerapkan teknik partisi gambar (*tiled inference*) dengan pemrosesan presisi FP16 yang efisien.

## Features

### Model and Inference

- Model `RealESRGAN_x4plus_anime_6B` (arsitektur RRDBNet 6 blok, 4.47M parameter).
- Pembagian gambar per bagian (tile 512px, padding 32px) untuk menjaga konsumsi memori VRAM GPU.
- Menggunakan presisi FP16 untuk komputasi cepat pada akselerator CUDA.
- Pemisahan dan rekonstruksi kanal transparansi (*alpha channel*) via interpolasi bicubic untuk file PNG.
- Kode PyTorch murni tanpa dependensi usang.

### Validation and Safety

- Format file yang didukung: PNG, JPG, JPEG, dan WebP.
- Batasan keamanan: ukuran file maksimal 50 MB dan dimensi maksimal 8192px untuk mencegah kehabisan RAM.
- Koreksi rotasi otomatis berdasarkan metadata EXIF gambar.

### Output and Verification

- File gambar PNG 4x disimpan dengan nama unik (`<nama>_x4_<id>.png`).
- Manifes metadata JSON per eksekusi (mencatat checksum SHA-256, durasi, dan pemakaian puncak VRAM).
- Verifikasi otomatis pasca-simpan untuk memastikan file output dapat dibaca normal.
- Visualisasi perbandingan sebelum-sesudah dan crop detail 35% langsung di notebook.

## Requirements

### Hardware

- GPU: NVIDIA L4 (24 GB VRAM) di Google Colab, atau GPU NVIDIA CUDA lain (disarankan minimal 8 GB VRAM).
- RAM: Minimal 12 GB.

### Environment

- OS: Linux (Ubuntu 22.04 LTS / Colab) atau Windows 10/11.
- Python: Versi 3.10 ke atas.
- Interface: VS Code via remote Colab atau Google Colab langsung.

### Dependencies

Daftar paket pada `requirements.txt`:

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `numpy>=1.24.0`
- `Pillow>=9.5.0`
- `PyYAML>=6.0`
- `matplotlib>=3.7.0`

## Installation

1. Clone repositori:
   
   ```bash
   git clone https://github.com/RozhakDev/AnimeEnhance.git
   cd AnimeEnhance
   ```

2. Pasang dependensi:
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Atur konfigurasi parameter di file `configs/default_config.yaml`:
   
   ```yaml
   pipeline:
     name: "AnimeEnhance"
     device: "cuda"
   
   inference:
     tile_size: 512
     tile_padding: 32
     half_precision: true
   ```

2. Letakkan file gambar di folder input:
   
   ```bash
   cp /path/gambar.png data/input/
   ```

3. Jalankan seluruh sel notebook secara berurutan:
   
   ```text
   notebooks/01_AnimeEnhancePipeline.ipynb
   ```
   
   Notebook akan memvalidasi gambar, memuat model, menjalankan inferensi 4x, menampilkan preview perbandingan visual, menyimpan output, dan membersihkan memori GPU secara otomatis.

4. Periksa hasil di folder `data/output/`:
   
   - `<nama>_x4_<id>.png`: Gambar hasil restorasi 4x.
   - `manifest_<id>.json`: Catatan log metadata dan verifikasi file.

## Acknowledgements

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) oleh Xintao Wang dkk. untuk riset super-resolution dan bobot model.
- [PyTorch](https://pytorch.org/) untuk framework deep learning dan akselerasi CUDA.
- [Pillow](https://python-pillow.org/) dan [NumPy](https://numpy.org/) untuk pengolahan gambar dan manipulasi array.

## License

Proyek ini menggunakan lisensi [MIT](LICENSE). Bobot model Real-ESRGAN mengikuti lisensi hulu (BSD 3-Clause).
