import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="Credit Score Analysis",
    page_icon="💰",
    layout="wide"
)

# --- Sidebar: Pemilihan Model ---
st.sidebar.header("⚙️ Konfigurasi Model")
model_options = {
    "LogisticRegression": "model6.pkl", # Diganti dari LinearRegression karena outputnya Klasifikasi (0/1)
    "XGBoost": "model5.pkl",          
    "RandomForest": "model4.pkl",
    "LogisticRegression2": "model3.pkl"     
}

selected_model_name = st.sidebar.selectbox(
    "Pilih Model Machine Learning:",
    options=list(model_options.keys()),
    index=0 # Default ke Logistic untuk testing
)

selected_model_file = model_options[selected_model_name]

# --- 1. Load Model & Encoder ---
@st.cache_resource
def load_assets(model_path):
    try:
        model = joblib.load(model_path)
        encoder = joblib.load('ohe_encoder.pkl')
        
        # Coba load scaler jika ada (SANGAT PENTING untuk Logistic Regression)
        scaler = None
        try:
            scaler = joblib.load('scaler.pkl') 
        except:
            pass # Lanjut tanpa scaler jika tidak ada (tapi bahaya untuk Logistic Reg)
            
        return model, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"File aset tidak ditemukan: {e}")
        return None, None, None

model, encoder, scaler = load_assets(selected_model_file)

# --- 2. Input Form ---
if model and encoder:
    st.title("💰 Aplikasi Prediksi Risiko Kredit")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Utama")
        loan_amnt = st.number_input("Jumlah Pinjaman ($)", min_value=500.0, value=5000.0)
        int_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=10.0)
        installment = st.number_input("Cicilan Bulanan ($)", min_value=0.0, value=150.0)
        annual_inc = st.number_input("Pendapatan Tahunan ($)", min_value=1000.0, value=50000.0)
        term_numeric = st.selectbox("Durasi (Bulan)", options=[36, 60])

    with col2:
        st.subheader("Profil Peminjam")
        
        # --- INPUT GRADE MANUAL (Dikembalikan) ---
        col_grade1, col_grade2 = st.columns(2)
        with col_grade1:
            grade_option = st.selectbox("Grade (Peringkat)", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            # Mapping Grade Huruf ke Angka (1-7)
            grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            grade = grade_mapping[grade_option]
            
        with col_grade2:
            # Subgrade 1-5 untuk setiap Grade
            sub_grade_num = st.selectbox(f"Sub Grade ({grade_option})", options=[1, 2, 3, 4, 5])
            # Mapping total 1-35 (A1=1 ... G5=35)
            sub_grade = (grade - 1) * 5 + sub_grade_num
        
        emp_length = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=40, value=2)
        home_ownership = st.selectbox("Kepemilikan Rumah", options=['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE'])
        verification_status = st.selectbox("Status Verifikasi", options=['Verified', 'Source Verified', 'Not Verified'])
        purpose = st.selectbox("Tujuan Pinjaman", options=[
            'debt_consolidation', 'credit_card', 'home_improvement', 'other', 
            'major_purchase', 'medical', 'small_business', 'car', 'vacation', 
            'moving', 'wedding', 'house', 'renewable_energy', 'educational'
        ])
    
    # --- FITUR TAMBAHAN (Opsi Lanjutan) ---
    st.markdown("---")
    with st.expander("📂 Opsi Lanjutan (Riwayat Kredit & Aset)", expanded=True):
        col3, col4, col5 = st.columns(3)
        
        with col3:
            delinq_2yrs = st.number_input("Jml Tunggakan (2th Terakhir)", min_value=0, value=0, help="Jumlah pelanggaran/tunggakan 30+ hari dalam 2 tahun terakhir.")
            inq_last_6mths = st.number_input("Inquiry (6bln Terakhir)", min_value=0, value=0, help="Jumlah pengecekan kredit dalam 6 bulan terakhir.")
            pub_rec = st.number_input("Public Records (Kebangkrutan)", min_value=0, value=0, help="Jumlah catatan publik yang buruk (kebangkrutan, dll).")
        
        with col4:
            open_acc = st.number_input("Jml Akun Kredit Terbuka", min_value=0, value=5, help="Jumlah jalur kredit yang sedang aktif.")
            total_acc = st.number_input("Total Akun Kredit", min_value=0, value=10, help="Total seluruh akun kredit (aktif + tutup).")
            revol_util = st.number_input("Revol Util (%)", min_value=0.0, value=30.0, help="Persentase pemakaian kartu kredit.")

        with col5:
            revol_bal = st.number_input("Saldo Revolving ($)", min_value=0.0, value=1000.0, help="Total saldo tagihan kartu kredit saat ini.")
            last_pymnt_amnt = st.number_input("Pembayaran Terakhir ($)", min_value=0.0, value=0.0)

    # Tombol Prediksi
    if st.button("🔍 Analisa Risiko", type="primary"):
        try:
            # --- A. Preprocessing Data Input ---
            # Hitung DTI
            dti = (installment * 12 / annual_inc) * 100 if annual_inc > 0 else 0
            
            # Buat DataFrame Input
            input_data = {
                'loan_amnt': [loan_amnt],
                'int_rate': [int_rate],
                'installment': [installment],
                'annual_inc': [annual_inc],
                'dti': [dti],
                'delinq_2yrs': [delinq_2yrs],
                'inq_last_6mths': [inq_last_6mths],
                'open_acc': [open_acc],
                'pub_rec': [pub_rec],
                'revol_bal': [revol_bal],
                'revol_util': [revol_util],
                'total_acc': [total_acc],
                'last_pymnt_amnt': [last_pymnt_amnt],
                'term_numeric': [term_numeric],
                'emp_length': [emp_length],
                
                # Menggunakan Grade & Subgrade Manual dari User
                'grade': [grade],           
                'sub_grade': [sub_grade],
                
                # Kolom Kategori untuk Encoder
                'home_ownership': [home_ownership],
                'verification_status': [verification_status],
                'purpose': [purpose]
            }
            
            raw_df = pd.DataFrame(input_data)

            # --- B. Encoding ---
            cat_columns = ['home_ownership', 'verification_status', 'purpose']
            encoded_features = encoder.transform(raw_df[cat_columns])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_columns))
            
            # --- C. Gabungkan Data ---
            num_columns = [
                'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
                'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 
                'revol_bal', 'revol_util', 'total_acc', 'last_pymnt_amnt', 
                'term_numeric', 'emp_length', 'grade', 'sub_grade'
            ]
            
            # Gabungkan Numerik + OneHotEncoded
            final_df = pd.concat([raw_df[num_columns], encoded_df], axis=1)

            # --- PENTING: SCALING ---
            # Model Linear/Logistic Regression SANGAT sensitif terhadap skala data.
            # Jika model dilatih dengan data yang dinormalisasi (StandardScaler/MinMax), input WAJIB discale juga.
            if scaler:
               final_df[num_columns] = scaler.transform(final_df[num_columns])
            else:
               # Peringatan khusus jika menggunakan Logistic Regression tanpa scaler
               if "Logistic" in selected_model_name:
                   st.warning("⚠️ PERINGATAN: 'scaler.pkl' tidak ditemukan. Model Logistic Regression akan gagal (selalu prediksi 100% Bad) jika menerima data mentah yang tidak discaling.")

            # --- D. Prediksi ---
            # Trik untuk menangani LinearRegression vs Classifier
            if hasattr(model, "predict_proba"):
                # Untuk LogisticRegression, Random Forest, XGBoost
                prediction_class = model.predict(final_df)[0] # Hasil 0 atau 1
                probability = model.predict_proba(final_df)[0] # Hasil [0.8, 0.2]
                
                # Tampilkan Debugging untuk melihat kenapa 100%
                with st.expander("🕵️ Debugging Model"):
                    st.write(f"Model Probability: {probability}")
                    if probability[1] > 0.99:
                        st.write("❗ Probabilitas sangat tinggi (100%). Kemungkinan besar karena data tidak di-scale.")

            else:
                # Fallback jika model benar-benar Linear Regression murni (Regressor)
                prediction_score = model.predict(final_df)[0]
                
                # Kita harus buat threshold manual untuk regresi
                # Biasanya: < 0.5 = Good (0), >= 0.5 = Bad (1)
                THRESHOLD = 0.5 
                prediction_class = 1 if prediction_score >= THRESHOLD else 0
                # Simulasi probabilitas agar tidak error
                prob_bad = max(0.0, min(1.0, prediction_score)) # Clip 0-1
                probability = [1-prob_bad, prob_bad] 

            # --- E. Tampilkan Hasil ---
            st.write("---")
            st.subheader("Hasil Analisa")

            # Logic Tampilan: Kelas 0 = Good, Kelas 1 = Bad
            if prediction_class == 0: 
                st.success(f"✅ **Low Risk (Credit Approved)**")
                st.write(f"Confidence Score (Peluang Lunas): {probability[0]:.2%}")
            else: 
                st.error(f"⚠️ **High Risk (Credit Rejected)**")
                st.write(f"Confidence Score (Peluang Gagal Bayar): {probability[1]:.2%}")
                
                st.warning("Aplikasi memprediksi risiko tinggi.")

            # Tampilkan Data Akhir yang masuk ke Model
            with st.expander("Lihat Data Final (Input ke Model)"):
                st.dataframe(final_df)
                st.write(f"Jumlah Fitur: {final_df.shape[1]}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.code("Cek apakah jumlah kolom di 'final_df' sama dengan jumlah kolom saat training.")

else:
    st.warning("Menunggu file model (pkl) dan encoder...")