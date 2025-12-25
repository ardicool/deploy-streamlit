import streamlit as st
import pandas as pd
import joblib
import numpy as np
import math

# Konfigurasi Halaman
st.set_page_config(
    page_title="Credit Score Analysis",
    page_icon="💰",
    layout="wide"
)

# --- Sidebar: Pemilihan Model ---
st.sidebar.header("⚙️ Konfigurasi Model")
model_options = {
    "RandomForest": "model4.pkl",
    "XGBoost": "model5.pkl",
    "LinearRegression": "model6.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Pilih Model Machine Learning:",
    options=list(model_options.keys()),
    index=0 # Default ke RandomForest
)

selected_model_file = model_options[selected_model_name]

# --- 1. Load Model & Encoder ---
@st.cache_resource
def load_assets(model_path):
    try:
        # Load model sesuai pilihan user
        model = joblib.load(model_path)
        # Encoder tetap sama untuk semua model
        encoder = joblib.load('ohe_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error(f"File '{model_path}' atau 'ohe_encoder.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None, None

model, encoder = load_assets(selected_model_file)

# --- Fungsi Helper: Kalkulasi Grade Otomatis ---
def calculate_grade_subgrade(dti_val):
    """
    Menghitung Grade dan Sub Grade berdasarkan DTI.
    Grade A: < 5%
    Grade B: 5–10%
    Grade C: 10–15%
    Grade D: 15–20%
    Grade E: > 20%
    Subgrade diinterpolasi per 1% (misal 0-1% = A1, 1-2% = A2, dst)
    """
    if dti_val < 5:
        g = 'A'
        # Logika: 0-0.99 -> 1, 1-1.99 -> 2, dst
        sg_num = int(dti_val) + 1
    elif dti_val < 10:
        g = 'B'
        sg_num = int(dti_val - 5) + 1
    elif dti_val < 15:
        g = 'C'
        sg_num = int(dti_val - 10) + 1
    elif dti_val < 20:
        g = 'D'
        sg_num = int(dti_val - 15) + 1
    else:
        g = 'E'
        # Untuk > 20, kita interpolasi 20-25 jadi E1-E5. Di atas 25 tetap E5.
        sg_num = int(dti_val - 20) + 1
    
    # Clamp sg_num agar selalu antara 1-5
    sg_num = max(1, min(5, sg_num))
    
    return g, f"{g}{sg_num}"

# Callback function untuk update Grade saat DTI berubah manual
def update_grade_dti():
    dti_val = st.session_state.dti_input
    new_grade, new_sub_grade = calculate_grade_subgrade(dti_val)
    st.session_state.grade_val = new_grade
    st.session_state.sub_grade_val = new_sub_grade

# Callback function untuk update DTI (dan Grade) saat Income/Debt berubah
def update_dti_logic():
    inc = st.session_state.get('annual_inc_input', 0.0)
    debt = st.session_state.get('monthly_debt_input', 0.0)
    
    if inc > 0:
        # Rumus DTI: (Total Hutang Bulanan / Pendapatan Bulanan) * 100
        monthly_inc = inc / 12
        new_dti = (debt / monthly_inc) * 100
        
        # Update nilai DTI di session state
        st.session_state.dti_input = new_dti
        
        # Karena DTI berubah, Grade juga harus diupdate
        new_grade, new_sub_grade = calculate_grade_subgrade(new_dti)
        st.session_state.grade_val = new_grade
        st.session_state.sub_grade_val = new_sub_grade

# --- Inisialisasi Session State ---
if 'grade_val' not in st.session_state:
    st.session_state.grade_val = 'B' # Default
if 'sub_grade_val' not in st.session_state:
    st.session_state.sub_grade_val = 'B3' # Default

# --- 2. Judul & Deskripsi ---
st.title("🏦 Credit Score Prediction App")
st.markdown(f"""
Aplikasi ini memprediksi apakah pinjaman berisiko **Default** (Gagal Bayar) atau **Fully Paid** (Lancar).
**Model Aktif:** `{selected_model_name}` ({selected_model_file})
""")

if model is not None and encoder is not None:

    # Container Input (Menggantikan st.form agar bisa interaktif)
    st.subheader("📋 Informasi Peminjam")
    
    col1, col2, col3 = st.columns(3)

    # --- INPUT NUMERIKAL ---
    with col1:
        st.markdown("**Data Finansial**")
        loan_amnt = st.number_input("Jumlah Pinjaman (Loan Amount)", min_value=0.0, value=10000.0)
        int_rate = st.number_input("Suku Bunga (Interest Rate %)", min_value=0.0, value=10.0)
        installment = st.number_input("Cicilan Bulanan (Loan Ini)", min_value=0.0, value=300.0)
        
        # Input Annual Income dengan Trigger Calculation
        annual_inc = st.number_input(
            "Pendapatan Tahunan", 
            min_value=0.0, 
            value=50000.0,
            key='annual_inc_input',
            on_change=update_dti_logic
        )

        # Input Baru: Total Debt untuk kalkulasi DTI
        monthly_debt = st.number_input(
            "Total Hutang Bulanan (Semua)", 
            min_value=0.0, 
            value=625.0, # Contoh default (15% DTI dari 50k income)
            key='monthly_debt_input',
            on_change=update_dti_logic,
            help="Total cicilan hutang per bulan (termasuk pinjaman ini dan lainnya). Digunakan untuk menghitung DTI otomatis."
        )
        
        # DTI (Sekarang otomatis terisi, tapi bisa diedit manual)
        dti = st.number_input(
            "Debt-to-Income Ratio (DTI)", 
            min_value=0.0, 
            value=15.0, 
            key='dti_input',
            on_change=update_grade_dti, # Jika user edit manual DTI, grade tetap terupdate
            help="Otomatis dihitung: (Total Hutang Bulanan / (Pendapatan Tahunan / 12)) * 100"
        )

    with col2:
        st.markdown("**Riwayat Kredit**")
        delinq_2yrs = st.number_input("Pelanggaran 2th Terakhir", min_value=0, value=0)
        inq_last_6mths = st.number_input("Inquiry 6bln Terakhir", min_value=0, value=0)
        open_acc = st.number_input("Jumlah Akun Terbuka", min_value=0, value=5)
        pub_rec = st.number_input("Catatan Publik Buruk", min_value=0, value=0)
        revol_bal = st.number_input("Saldo Revolving", min_value=0.0, value=1000.0)

    with col3:
        st.markdown("**Lainnya**")
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, value=30.0)
        total_acc = st.number_input("Total Akun Kredit", min_value=0, value=10)
        last_pymnt_amnt = st.number_input("Pembayaran Terakhir", min_value=0.0, value=0.0)
        term_numeric = st.selectbox("Tenor (Bulan)", options=[36, 60], help="Konversi term ke angka")

    st.divider()
    
    st.subheader("📑 Data Kategori")
    cat_col1, cat_col2 = st.columns(2)

    # Daftar opsi lengkap untuk Sub Grade
    all_sub_grades = [
        'A1','A2','A3','A4','A5',
        'B1','B2','B3','B4','B5',
        'C1','C2','C3','C4','C5',
        'D1','D2','D3','D4','D5',
        'E1','E2','E3','E4','E5',
        'F1','F2','F3','F4','F5',
        'G1','G2','G3','G4','G5'
    ]

    with cat_col1:
        # Grade otomatis terupdate lewat key='grade_val' dari session_state
        grade = st.selectbox(
            "Grade (Auto from DTI)", 
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            key='grade_val' 
        )
        
        # Sub Grade otomatis terupdate lewat key='sub_grade_val' dari session_state
        # Pastikan nilai default ada di list options
        try:
             idx = all_sub_grades.index(st.session_state.sub_grade_val)
        except ValueError:
             idx = 0

        sub_grade = st.selectbox(
            "Sub Grade (Auto from DTI)", 
            options=all_sub_grades,
            key='sub_grade_val'
        )
        
        emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
        emp_length = st.selectbox("Lama Bekerja", options=emp_length_options)
    
    with cat_col2:
        home_ownership = st.selectbox("Kepemilikan Rumah", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        verification_status = st.selectbox("Status Verifikasi", options=['Verified', 'Source Verified', 'Not Verified'])
        purpose = st.selectbox("Tujuan Pinjaman", options=['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business'])

    # Tombol Submit
    st.write("")
    submitted = st.button("🔍 Analisa Risiko Kredit", type="primary")

    # --- 3. Logika Pemrosesan & Prediksi ---
    if submitted:
        # --- A. Definisikan Mapping Sesuai Training ---
        grade_map = {'A':0,'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
        
        sub_grade_map = {
            'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 
            'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9, 
            'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14, 
            'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19, 
            'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24, 
            'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29, 
            'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34
        }

        # Mapping Emp Length sesuai logika kode Anda (<1 -> 0, 10+ -> 10)
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
        }

        # --- B. Proses Data ---
        # 1. Konversi data kategori yang di-map ke angka
        grade_val = grade_map.get(grade, 0)
        sub_grade_val = sub_grade_map.get(sub_grade, 0)
        emp_length_val = emp_length_map.get(emp_length, 0)

        # 2. Siapkan Dictionary Data
        data_input = {
            # Numerikal Asli
            'loan_amnt': [loan_amnt], 'int_rate': [int_rate], 'installment': [installment],
            'annual_inc': [annual_inc], 'dti': [dti], 'delinq_2yrs': [delinq_2yrs],
            'inq_last_6mths': [inq_last_6mths], 'open_acc': [open_acc], 'pub_rec': [pub_rec],
            'revol_bal': [revol_bal], 'revol_util': [revol_util], 'total_acc': [total_acc],
            'last_pymnt_amnt': [last_pymnt_amnt], 'term_numeric': [term_numeric],
            
            # Kategori yang sudah jadi Angka (Emp Length)
            'emp_length': [emp_length_val],
            
            # Kategori untuk OHE (Perhatikan: Grade & Subgrade masuk sini tapi sbg Angka)
            'grade': [grade_val], 
            'sub_grade': [sub_grade_val], 
            
            # Kategori String (Murni OHE)
            'home_ownership': [home_ownership], 
            'verification_status': [verification_status],
            'purpose': [purpose]
        }
        
        raw_df = pd.DataFrame(data_input)

        try:
            # --- C. Encoding & Penggabungan ---
            
            # 1. Kolom untuk OHE (Sesuai snippet Anda: grade & subgrade masuk OHE meski sudah angka)
            cat_columns_ohe = ['home_ownership','verification_status','purpose','grade','sub_grade']
            
            # 2. Transform OHE
            encoded_array = encoder.transform(raw_df[cat_columns_ohe])
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_columns_ohe))

            # 3. Kolom Numerikal (Fitur sisa yang TIDAK masuk OHE)
            # Pastikan 'emp_length' ada di sini karena dia bukan bagian dari cat_columns_ohe
            num_columns = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 
                           'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
                           'total_acc', 'last_pymnt_amnt', 'term_numeric', 'emp_length']

            # 4. Gabungkan (Concat)
            final_df = pd.concat([raw_df[num_columns], encoded_df], axis=1)

            # --- D. Prediksi ---
            prediction = model.predict(final_df)[0]
            
            # Cek apakah model support predict_proba
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(final_df)[0]
            else:
                # Fallback untuk model tanpa predict_proba (misal LinearRegression biasa)
                probability = [0, 0] 

            # Tampilkan Hasil
            st.write("---")
            st.subheader("Hasil Analisa")

            if prediction == 0: 
                st.success(f"✅ **Low Risk (Credit Approved)**")
                if hasattr(model, "predict_proba"):
                    st.write(f"Confidence Score: {probability[0]:.2%}")
            else: 
                st.error(f"⚠️ **High Risk (Credit Rejected)**")
                if hasattr(model, "predict_proba"):
                    st.write(f"Confidence Score: {probability[1]:.2%}")
                st.warning("Aplikasi ini memiliki probabilitas tinggi untuk gagal bayar.")

            with st.expander("Lihat Data yang Diproses Model"):
                st.dataframe(final_df)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.info("Tips: Pastikan 'ohe_encoder.pkl' dilatih dengan input Grade/Subgrade yang sudah di-mapping ke angka.")

else:
    st.warning("Menunggu file model dan encoder...")