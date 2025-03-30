# Laporan Proyek Machine Learning - T. Muhammad Caesar Maulana

## Domain Proyek

Emas telah lama digunakan sebagai alat tukar dan simbol kekayaan di berbagai belahan dunia. Saat ini, emas memiliki peran penting dalam ekonomi global, terutama sebagai aset lindung nilai terhadap inflasi dan ketidakstabilan pasar. Bank sentral banyak negara menyimpan emas sebagai cadangan devisa untuk menjamin stabilitas ekonomi dan kepercayaan pasar.

Fluktuasi harga emas dipengaruhi oleh berbagai faktor seperti nilai tukar mata uang, inflasi, suku bunga, dan kondisi geopolitik. Oleh karena itu, melakukan prediksi terhadap pergerakan harga emas dapat membantu investor dan pembuat kebijakan dalam mengambil keputusan yang lebih tepat.

**Referensi**:
- [Inflation Responses to Commodity Price Shocks–How and Why Do Countries Differ?](https://www.imf.org/external/pubs/ft/wp/2012/wp12225.pdf)
- [Sejarah dan Peran Emas: Dari Simbol Kekayaan Kuno hingga Investasi Modern - Galeri 24](https://galeri24.co.id/post/sejarah-dan-peran-emas-dari-simbol-kekayaan-kuno-hingga-investasi-modern)
- [5 Peran Emas dalam Perencanaan Keuangan Jangka Panjang - Treasury](https://www.treasury.id/5-peran-emas-dalam-perencanaan-keuangan-jangka-panjang)
- [Emas dan Ekonomi Global: Hubungan yang Kompleks - Blog IndoGold](https://blog.indogold.id/emas-dan-ekonomi-global-hubungan-yang-kompleks/)
- [Fungsi Emas Sebagai Aset Lindung Nilai - Bareksa.com](https://www.bareksa.com/berita/reksa-dana/2020-09-07/fungsi-emas-sebagai-aset-lindung-nilai)

---

## Business Understanding

### Problem Statements
1. Bagaimana faktor-faktor ekonomi seperti harga minyak, indeks saham, dan nilai tukar mata uang mempengaruhi harga emas?
2. Bagaimana membangun model prediksi harga emas yang akurat menggunakan teknik machine learning?
3. Fitur-fitur apa yang paling signifikan dalam memprediksi harga emas?

### Goals
1. Menganalisis pengaruh faktor-faktor ekonomi terhadap harga emas.
2. Membangun model prediksi harga emas yang akurat menggunakan berbagai algoritma machine learning.
3. Mengidentifikasi fitur-fitur yang paling berpengaruh dalam prediksi harga emas.

### Solution Statements
1. Menggunakan algoritma regresi seperti Linear SVR, Lasso, Ridge, dan Bayesian Ridge untuk memprediksi harga emas.
2. Melakukan tuning hyperparameter untuk meningkatkan performa model.
3. Menggunakan teknik seleksi fitur untuk mengidentifikasi fitur yang paling signifikan.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini diperoleh dari [Kaggle Gold Price Prediction Dataset](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset). Dataset ini mencakup data historis harga emas dan berbagai faktor ekonomi yang mempengaruhinya, seperti harga minyak, indeks saham, dan nilai tukar mata uang.

Data dalam penelitian ini dikumpulkan dari berbagai sumber dalam rentang waktu **18 November 2011** hingga **1 Januari 2019**. Dataset ini memiliki **1718 baris** dan **80 kolom**, yang terdiri dari berbagai atribut ekonomi dan keuangan yang berpengaruh terhadap harga emas.

Fitur yang dikumpulkan mencakup berbagai faktor ekonomi seperti **harga minyak mentah, indeks saham (S&P 500 dan Dow Jones), nilai tukar mata uang Euro-USD, harga logam mulia lainnya (Perak, Platinum, Palladium, Rhodium), indeks dolar AS, serta data dari Gold Miners ETF dan Eldorado Gold Corporation**.

### Variabel-variabel pada Dataset:
- **Open, High, Low, Close, Adj Close**: Harga pembukaan, tertinggi, terendah, penutupan, dan penutupan yang disesuaikan untuk emas.
- **Volume**: Volume perdagangan emas.
- **SP_open, SP_high, SP_low, SP_close, SP_Ajclose**: Harga pembukaan, tertinggi, terendah, penutupan, dan penutupan yang disesuaikan untuk indeks S&P 500.
- **DJ_open, DJ_high, DJ_low, DJ_close, DJ_Ajclose**: Harga pembukaan, tertinggi, terendah, penutupan, dan penutupan yang disesuaikan untuk indeks Dow Jones.
- **EU_Price, EU_open, EU_high, EU_low**: Harga dan nilai tukar Euro-USD.
- **OF_Price, OF_Open, OF_High, OF_Low**: Harga minyak mentah Brent.
- **SF_Price, SF_Open, SF_High, SF_Low**: Harga futures perak.
- **USB_Price, USB_Open, USB_High, USB_Low**: Harga obligasi US 10 tahun.
- **PLT_Price, PLT_Open, PLT_High, PLT_Low**: Harga platinum.
- **PLD_Price, PLD_Open, PLD_High, PLD_Low**: Harga palladium.
- **RHO_PRICE**: Harga rhodium.
- **USDI_Price, USDI_Open, USDI_High, USDI_Low**: Indeks dolar AS.
- **GDX_Open, GDX_High, GDX_Low, GDX_Close, GDX_Adj Close**: Harga ETF penambang emas.
- **USO_Open, USO_High, USO_Low, USO_Close, USO_Adj Close**: Harga ETF minyak.


### Pengaruh Harga Indeks Terhadap Harga Emas
Analisis ini bertujuan untuk memahami hubungan antara harga emas (GLD) dan indeks saham seperti S&P 500 (SPY) serta Dow Jones (DJ). Tujuan utama adalah mengidentifikasi pola pergerakan harga emas seiring dengan perubahan harga indeks saham.

Langkah-langkah yang dilakukan:
1. Mengambil data harga penutupan yang telah disesuaikan (Adjusted Close) untuk GLD, SPY, dan DJ.
2. Menyusun DataFrame baru yang berisi ketiga harga tersebut.
3. Membuat visualisasi dalam bentuk plot untuk memahami tren harga secara keseluruhan.
4. Menambahkan label sumbu dan legenda agar grafik lebih informatif.
5. Menampilkan plot hasil analisis.

```python
GLD_adj_close = df['Adj Close']
```

### Menghitung Return Harian dari Semua Fitur
Return harian dihitung untuk memahami perubahan relatif harga suatu aset dari satu hari ke hari berikutnya. Rumus perhitungannya adalah:

\[
\text{Return harian} = \frac{\text{Harga hari ini}}{\text{Harga hari sebelumnya}} - 1
\]

Langkah-langkah yang dilakukan:
1. Mengambil data harga Adjusted Close untuk berbagai aset seperti GLD, SPY, DJ, minyak mentah (USO), dan logam lainnya.
2. Menggunakan fungsi untuk menghitung return harian berdasarkan harga hari ini dibandingkan dengan harga sebelumnya.
3. Menyusun DataFrame yang berisi return harian semua fitur.
4. Membuat visualisasi return harian untuk 100 data terakhir.
5. Menampilkan grafik untuk analisis pergerakan return harian.

```python
def compute_daily_returns(df):
    return (df / df.shift(1)) - 1
```

### Menghitung Return Harian dari Indeks Saham
Tujuan dari analisis ini adalah memahami volatilitas harian dari indeks saham yang relevan dengan harga emas, seperti **Gold ETF (GLD), S&P 500 (SPY), dan Dow Jones (DJ)**.

Langkah-langkah yang dilakukan:
1. Menyusun DataFrame untuk return harian indeks saham.
2. Membuat visualisasi return harian 100 data terakhir untuk melihat pola pergerakan harga.
3. Menampilkan grafik untuk analisis lebih lanjut.

```python
df_s = pd.DataFrame({'GLD': GLD_daily_return, 'SPY': SPY_daily_return, 'DJ': DJ_adj_return})
```

### Scatterplot untuk Hubungan Antar Fitur
Scatterplot digunakan untuk memahami hubungan antara harga emas dan faktor ekonomi lainnya.

Langkah-langkah yang dilakukan:
1. Membuat scatterplot antara harga emas dan berbagai fitur lainnya seperti indeks saham, harga minyak, dan harga logam lainnya.
2. Memvisualisasikan hubungan ini untuk melihat apakah ada pola korelasi tertentu.
3. Menampilkan grafik untuk analisis lebih lanjut.

```python
df_d.plot(kind='scatter', x='SPY', y='GLD', title='Hubungan antara SPY dan GLD')
```

### Statistical Measures (Mean, Standard Deviation, Kurtosis)
Statistik deskriptif digunakan untuk memahami distribusi data return harian, termasuk **mean**, **standard deviation**, dan **kurtosis**.

- **Mean**: Rata-rata return harian.
- **Standard Deviation**: Ukuran volatilitas return harian.
- **Kurtosis**: Mengukur keparahan ekor distribusi data.

Langkah-langkah yang dilakukan:
1. Menghitung nilai mean, standard deviation, dan kurtosis untuk setiap fitur return harian.
2. Membuat histogram untuk memvisualisasikan distribusi return harian.
3. Menambahkan garis vertikal pada histogram untuk menunjukkan nilai rata-rata dan deviasi standar.
4. Menampilkan grafik hasil analisis.

```python
mean = df_d['GLD'].mean()
std = df_d['GLD'].std()
kurt = df_d['GLD'].kurtosis()
```

### Plotting Correlation Matrix
Matriks korelasi digunakan untuk memahami hubungan antar fitur dalam dataset.

Langkah-langkah yang dilakukan:
1. Menghitung korelasi antara semua fitur dalam dataset.
2. Membuat heatmap untuk memvisualisasikan korelasi antar fitur.
3. Menganalisis fitur mana yang memiliki korelasi tinggi atau rendah terhadap harga emas.
4. Menampilkan hasil dalam bentuk grafik bar untuk mempermudah interpretasi.

```python
sns.heatmap(df.corr(), annot=True)
```

![Heatmap Korelasi](images/heatmap_correlation.jpg)

### Plotting Indikator Teknikal
Analisis indikator teknikal membantu memahami tren harga emas dengan menggunakan beberapa metode seperti:

- **Simple Moving Average (SMA)**: Rata-rata pergerakan harga dalam jangka waktu tertentu.
- **Bollinger Bands (BB)**: Mengukur volatilitas harga.
- **Moving Average Convergence Divergence (MACD)**: Menentukan arah dan kekuatan tren harga.
- **Relative Strength Index (RSI)**: Mengidentifikasi kondisi overbought atau oversold.
- **Standard Deviation (STDEV)**: Mengukur sebaran harga terhadap rata-rata.

Langkah-langkah yang dilakukan:
1. Menghitung indikator teknikal untuk GLD.
2. Membuat plot untuk setiap indikator.
3. Menampilkan hasil analisis teknikal dalam bentuk visualisasi.

```python
SMA_GLD = calculate_SMA(GLD_adj_close)
```

### Menghitung Selisih Open-Close dan High-Low
Selisih antara harga pembukaan dan penutupan serta harga tertinggi dan terendah dapat memberikan wawasan tentang volatilitas harian.

Langkah-langkah yang dilakukan:
1. Menghitung selisih harga Open-Close.
2. Menghitung selisih harga High-Low.
3. Membuat plot untuk memahami pola volatilitas harian.
4. Menampilkan hasil dalam bentuk grafik.

```python
Open_Close = df.Open - df.Close
High_Low = df.High - df.Low
```

---

## Data Preparation

### 1. Mengambil data harga penutupan yang disesuaikan (*Adjusted Close*) dari berbagai sumber
```python
GLD_adj_close = df['Adj Close']
```

### 2. Menghitung indikator teknikal

#### a. Menghitung Simple Moving Average (SMA)
```python
SMA_GLD = df['Adj Close'].rolling(window=15).mean()
```

#### b. Menghitung Moving Average Convergence Divergence (MACD)
```python
DIF, MACD = calculate_MACD(df['Adj Close'])
```

#### c. Menghitung Relative Strength Index (RSI)
```python
RSI = calculate_RSI(df['Adj Close'])
```

#### d. Menghitung Bollinger Bands (BB)
```python
upper_band, lower_band = calculate_BB(df['Adj Close'])
```

#### e. Menghitung Standar Deviasi (STDEV)
```python
STDEV = df['Adj Close'].rolling(window=5).std()
```

### 3. Menghitung selisih harga *Open-Close* dan *High-Low*
```python
Open_Close = df['Open'] - df['Close']
High_Low = df['High'] - df['Low']
```

### 4. Menyusun dataset dengan menambahkan indikator teknikal sebagai fitur
```python
test = df.copy()
test['SMA'] = SMA_GLD
test['Upper_band'] = upper_band
test['Lower_band'] = lower_band
test['DIF'] = DIF
test['MACD'] = MACD
test['RSI'] = RSI
test['STDEV'] = STDEV
test['Open_Close'] = Open_Close
test['High_Low'] = High_Low
```

### 5. Menghapus baris awal yang memiliki nilai *null* akibat penghitungan indikator
```python
test = test[33:]
```

### 6. Menentukan kolom target (*Adj Close*)
```python
target_adj_close = test[['Adj Close']]
```

### 7. Memilih kolom fitur yang akan digunakan
```python
feature_columns = ['Open', 'High', 'Low', 'Volume', 'SMA', 'Upper_band', 'Lower_band', 'DIF', 'MACD', 'RSI', 'STDEV', 'Open_Close', 'High_Low']
```

### 8. Normalisasi data fitur menggunakan *MinMaxScaler*
```python
scaler = MinMaxScaler()
feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
```

### 9. Menggeser target satu hari ke depan untuk prediksi harga di hari berikutnya
```python
target_adj_close = target_adj_close.shift(-1)
```

### 10. Membagi dataset menjadi set pelatihan dan validasi (90 hari terakhir untuk validasi)
```python
validation_y = target_adj_close[-90:-1]
validation_X = feature_minmax_transform[-90:-1]
```

### 11. Menghapus data validasi dari dataset utama
```python
feature_minmax_transform = feature_minmax_transform[:-90]
target_adj_close = target_adj_close[:-90]
```

### 12. Melakukan pembagian data pelatihan dan pengujian menggunakan *TimeSeriesSplit*
```python
ts_split = TimeSeriesSplit(n_splits=10)
```

---

## Modeling

### 1. Decision Tree Regressor
**Cara Kerja:**  
Membangun struktur pohon dengan membagi data secara rekursif berdasarkan fitur yang memberikan pemisahan terbaik.

**Parameter:**  
- `criterion='squared_error'` (default): Menggunakan squared error untuk mengukur kualitas split.
- `splitter='best'` (default): Memilih split terbaik berdasarkan impurity.
- `max_depth=None` (default): Tidak ada batasan kedalaman pohon.
- `min_samples_split=2` (default): Minimal 2 sampel untuk membagi node.
- `min_samples_leaf=1` (default): Minimal 1 sampel per leaf.
- `max_features=None` (default): Menggunakan semua fitur untuk split.
- `random_state=0`: Menentukan seed agar hasil dapat direproduksi.

| Kelebihan | Kekurangan |
|-----------|------------|
| Mudah diinterpretasi | Rentan overfitting |
| Tidak memerlukan normalisasi data | Kurang stabil terhadap perubahan kecil data |
| Cepat dalam prediksi | Performa buruk pada hubungan linear |

### 2. Support Vector Regressor (SVR) Linear
**Cara Kerja:**  
Mencari hyperplane optimal dalam ruang fitur yang meminimalkan error prediksi.

**Parameter:**  
- `kernel='linear'`: Menggunakan kernel linear untuk memisahkan data.
- `C=1.0` (default): Parameter regularisasi.
- `epsilon=0.1` (default): Menentukan margin error.
- `degree=3` (default, tidak berlaku untuk linear kernel): Derajat polinomial untuk kernel 'poly'.
- `gamma='scale'` (default, tidak berlaku untuk linear kernel): Skala otomatis untuk kernel RBF dan poly.

| Kelebihan | Kekurangan |
|-----------|------------|
| Efektif untuk high-dimensional space | Komputasi mahal untuk dataset besar |
| Robust terhadap outlier | Sulit memilih kernel yang tepat |
| Generalisasi baik dengan parameter tepat | Sensitif terhadap scaling data |

### 3. Random Forest Regressor
**Cara Kerja:**  
Membangun banyak pohon keputusan dengan teknik bagging.

**Parameter:**  
- `n_estimators=50`: Jumlah pohon dalam ensemble.
- `criterion='squared_error'` (default): Metode pengukuran split.
- `max_depth=None` (default): Tidak ada batasan kedalaman.
- `min_samples_split=2` (default): Minimal 2 sampel untuk split.
- `min_samples_leaf=1` (default): Minimal 1 sampel per leaf.
- `max_features='auto'` (default): Memilih subset fitur secara otomatis.
- `random_state=0`: Menentukan seed.

| Kelebihan | Kekurangan |
|-----------|------------|
| Mengurangi overfitting | Waktu training lebih lama |
| Dapat menangani missing values | Kurang interpretatif |
| Robust terhadap noise data | Memori besar untuk banyak pohon |

### 4. Lasso Regression
**Cara Kerja:**  
Regresi linear dengan penalti L1 untuk seleksi fitur.

**Parameter:**  
- `alpha=1.0` (default): Koefisien regulasi L1.
- `fit_intercept=True` (default): Memasukkan intersep dalam model.
- `max_iter=3000`: Jumlah iterasi maksimal.
- `tol=0.0001` (default): Toleransi konvergensi.
- `selection='cyclic'` (default): Urutan update koefisien.

| Kelebihan | Kekurangan |
|-----------|------------|
| Seleksi fitur otomatis | Tidak stabil pada fitur berkorelasi |
| Baik untuk high-dimensional data | Sulit memilih alpha optimal |
| Interpretasi mudah | Underfit jika terlalu banyak fitur relevan |

### 5. Ridge Regression
**Cara Kerja:**  
Regresi linear dengan penalti L2 untuk menangani multikolinearitas.

**Parameter:**  
- `alpha=1.0` (default): Koefisien regulasi L2.
- `fit_intercept=True` (default): Memasukkan intersep dalam model.
- `solver='auto'` (default): Memilih solver terbaik secara otomatis.
- `gcv_mode='auto'`: Mode validasi silang.

| Kelebihan | Kekurangan |
|-----------|------------|
| Stabil untuk data berkorelasi | Tidak melakukan seleksi fitur |
| Mencegah overfitting | Sensitif terhadap outlier |
| Performa baik pada data noisy | Kurang efektif untuk feature selection |

### 6. Bayesian Ridge Regression
**Cara Kerja:**  
Pendekatan Bayesian yang memodelkan distribusi probabilitas parameter.

**Parameter:**  
- `alpha_1=1e-6`: Hyperprior alpha pertama.
- `alpha_2=1e-6`: Hyperprior alpha kedua.
- `lambda_1=1e-6` (default): Hyperprior lambda pertama.
- `lambda_2=1e-6` (default): Hyperprior lambda kedua.
- `n_iter=300` (default): Jumlah iterasi maksimum.

| Kelebihan | Kekurangan |
|-----------|------------|
| Memberikan interval prediksi | Komputasi lebih intensif |
| Otomatis menangani overfitting | Sulit menginterpretasikan prior |
| Robust terhadap small datasets | Hyperparameter sensitif |

### 7. Gradient Boosting Regressor
**Cara Kerja:**  
Membangun model secara bertahap dengan mengoreksi residual.

**Parameter:**  
- `n_estimators=70`: Jumlah pohon dalam boosting.
- `learning_rate=0.1`: Kecepatan pembelajaran.
- `max_depth=4`: Kedalaman maksimal pohon.
- `loss='squared_error'` (default): Fungsi loss.
- `subsample=1.0` (default): Proporsi data untuk setiap pohon.
- `min_samples_split=2` (default): Minimum sampel untuk split.

| Kelebihan | Kekurangan |
|-----------|------------|
| Akurasi tinggi | Sensitif terhadap overfitting |
| Fleksibel dengan berbagai loss function | Waktu training lama |
| Handles mixed data types | Hyperparameter sensitif |

### 8. SGD Regressor
**Cara Kerja:**  
Optimisasi iteratif dengan gradient descent stokastik.

**Parameter:**  
- `max_iter=1000`: Jumlah iterasi maksimal.
- `tol=1e-3`: Toleransi konvergensi.
- `alpha=0.0001` (default): Koefisien regulasi L2.
- `penalty='l2'` (default): Jenis regularisasi.
- `learning_rate='invscaling'` (default): Metode penurunan learning rate.

| Kelebihan | Kekurangan |
|-----------|------------|
| Efisien untuk data besar | Sensitif terhadap feature scaling |
| Fleksibel dengan berbagai regularisasi | Sulit memilih learning rate |
| Dapat handle online learning | Konvergensi sulit diverifikasi |

### 9. Ensemble Model
**Cara Kerja:**  
Menggabungkan prediksi dari Lasso, Bayesian Ridge, dan Ridge Regression.

**Parameter:**  
Tidak ada parameter tambahan selain model penyusun.

| Kelebihan | Kekurangan |
|-----------|------------|
| Meningkatkan akurasi | Kehilangan interpretabilitas |
| Mengurangi overfitting | Komputasi lebih intensif |
| Robust terhadap noise | Kompleksitas maintenance tinggi |

---

## Evaluation

### Metrik Evaluasi
Dua metrik utama digunakan untuk mengevaluasi performa model:

1. **RMSE (Root Mean Squared Error)**  
   - Mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan variabel target
   - Semakin kecil nilai semakin baik
   - Formula: `√(1/n * Σ(aktual - prediksi)²)`

2. **R² Score (Koefisien Determinasi)**  
   - Mengukur proporsi variasi dalam data yang dapat dijelaskan oleh model
   - Rentang nilai 0-1 (semakin mendekati 1 semakin baik)
   - Formula: `1 - (Σ(aktual - prediksi)² / Σ(aktual - rata_rata)²)`

### Analisis Performa Model
Berdasarkan problem statements bisnis, berikut evaluasi model:

#### 1. Prediksi Harga Emas (Goal 2)
| Model                | RMSE   | R² Score | Keterangan |
|----------------------|--------|----------|------------|
| Decision Tree        | 1.217  | 0.659    | Benchmark  |
| Linear SVR           | 0.814  | 0.847    |            |
| Random Forest        | 0.808  | 0.850    |            |
| **Lasso**            | **0.712** | **0.883**| **Optimal**|
| Ridge                | 0.719  | 0.881    |            |
| Bayesian Ridge       | 0.720  | 0.881    |            |
| **Ensemble**         | **0.701** | **0.887**| **Terbaik**|

#### 2. Signifikansi Fitur (Goal 3)  
- Fitur paling berpengaruh berdasarkan Lasso:  
  1. Harga tertinggi (High) – **27.17**  
  2. Harga pembukaan (Open) – **23.47**  
  3. Harga terendah (Low) – **17.31**  
  4. Selisih harga pembukaan & penutupan (Open_Close) – **-4.72** (pengaruh negatif)  
  5. Harga penutupan GDX (GDX_Close) – **3.33**  

#### 3. Pengaruh Faktor Ekonomi (Goal 1)
Analisis korelasi menunjukkan:
- Indeks saham (SPY/DJ) berkorelasi positif moderat (0.4-0.6)
- Nilai tukar USD (USDI) berkorelasi negatif (-0.3)
- Harga minyak (USO) berkorelasi positif lemah (0.2)

### Visualisasi Kunci
#### Performa Model Sebelum Seleksi Fitur
![Hasil Prediksi Sebelum Feature Selection](images/prediction_before_fs.jpg)

*Gambar 1: Perbandingan prediksi vs aktual sebelum seleksi fitur*

![Perbandingan RMSE Sebelum Feature Selection](images/rmse_before_fs.jpg)

*Gambar 2: Perbandingan RMSE antar model sebelum seleksi fitur*

#### Performa Model Setelah Seleksi Fitur
![Hasil Prediksi Setelah Feature Selection](images/prediction_after_fs.jpg)

*Gambar 3: Perbandingan prediksi vs aktual setelah seleksi fitur*

![Perbandingan RMSE Setelah Feature Selection](images/rmse_after_fs.jpg)

*Gambar 4: Perbandingan RMSE antar model setelah seleksi fitur*

### Rekomendasi Bisnis
Berdasarkan evaluasi:

1. **Model Produksi**  
   - Gunakan **Ensemble Model** (RMSE 0.701) untuk prediksi harian
   - **Lasso Regression** (RMSE 0.712) sebagai alternatif lebih sederhana

2. **Faktor Pengaruh Utama**  
   Fokus monitor:
   - Harga pembukaan emas
   - Trend minyak mentah
   - Pergerakan USD bonds

3. **Improvement**  
   - Tambahkan data makroekonomi terkini
   - Eksperimen dengan model sequence-based (LSTM) untuk capture pola temporal

---
