import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Prediksi Harga Mobil", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Automobile_Price_Clean.csv")

@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(columns=["price"])
    y = df["price"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(n_estimators=100, random_state=42))])
    model.fit(X, y)
    return model, X, numeric, categorical

st.title("🚗 Aplikasi Prediksi Harga Mobil")
st.write("Masukkan spesifikasi mobil untuk mendapatkan perkiraan harga berdasarkan dataset Automobile Price.")
try:
    model, X, numeric_cols, categorical_cols = train_model()
except Exception as e:
    st.error(f"Model tidak dapat dimuat: {e}")
    st.stop()

st.caption(f"Model Random Forest • 100 trees • Dilatih menggunakan {len(X)} data mobil")
with st.form("car_form"):
    st.subheader("Spesifikasi mobil")
    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(X.columns):
        container = left if i % 2 == 0 else right
        with container:
            if col in categorical_cols:
                opts = sorted(X[col].dropna().astype(str).unique().tolist())
                mode = str(X[col].mode(dropna=True).iloc[0]) if not X[col].mode(dropna=True).empty else opts[0]
                default_idx = opts.index(mode) if mode in opts else 0
                values[col] = st.selectbox(col.replace('-', ' ').title(), opts, index=default_idx)
            else:
                series = pd.to_numeric(X[col], errors="coerce").dropna()
                default = float(series.mean()) if not series.empty else 0.0
                lo = float(series.min()) if not series.empty else 0.0
                hi = float(series.max()) if not series.empty else 100.0
                if lo == hi:
                    values[col] = st.number_input(col.replace('-', ' ').title(), value=default)
                else:
                    values[col] = st.number_input(col.replace('-', ' ').title(), min_value=lo, max_value=hi, value=min(max(default, lo), hi), step=0.1, format="%.2f")
    submitted = st.form_submit_button("Prediksi Harga Mobil", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([values], columns=X.columns)
    prediction = float(model.predict(input_df)[0])
    st.success("Prediksi berhasil dibuat")
    st.metric("Perkiraan harga mobil", f"${prediction:,.2f}")
    st.caption("Hasil merupakan estimasi model dari dataset latihan, bukan harga pasar atau penawaran resmi.")

with st.expander("Tentang aplikasi"):
    st.write("Aplikasi ini menggunakan Random Forest Regressor dengan 100 pohon. Nilai numerik yang kosong diimputasi menggunakan rata-rata dan nilai kategori yang kosong menggunakan kategori paling sering muncul. Kategori baru ditangani dengan aman oleh encoder.")
