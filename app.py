import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Memuat model regresi linear
lin_reg_loaded = joblib.load('lin_reg_model.joblib')

# Judul aplikasi
st.title("📈 Prediksi Gaji Berdasarkan Lama Bekerja")

# Tabs untuk input manual dan file
tab1, tab2 = st.tabs(["🎯 Prediksi Manual", "📂 Prediksi dari File"])

# Tab Prediksi Manual
with tab1:
    years_experience = st.number_input("Masukkan jumlah tahun bekerja:", min_value=0.0, step=0.1)
    if st.button("Prediksi Gaji"):
        gaji = lin_reg_loaded.predict([[years_experience]])
        st.success(f"Gaji seseorang setelah bekerja selama {years_experience} tahun adalah **${gaji[0]:,.2f}**")

# Tab Prediksi dari File
with tab2:
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel (harus punya kolom `Tahun_bekerja`)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Baca file sesuai tipe
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if 'Tahun_bekerja' not in df.columns:
            st.error("File tidak memiliki kolom 'Tahun_bekerja'.")
        else:
            st.write("Data yang diunggah:")
            st.dataframe(df)

            # Prediksi gaji
            try:
                predictions = lin_reg_loaded.predict(df[['Tahun_bekerja']])
                df['Prediksi_Gaji'] = predictions
                st.write("Hasil prediksi gaji:")
                st.dataframe(df)

                # Download hasil
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=csv,
                    file_name='hasil_prediksi.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
