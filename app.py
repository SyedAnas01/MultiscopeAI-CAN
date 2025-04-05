# Universal Cross-Domain Forecasting + Anomaly Detection System (MultiScopeAI)

# === SETUP ===
# Install required libraries
# !pip install pandas numpy matplotlib seaborn scikit-learn prophet streamlit plotly fpdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from prophet import Prophet
import streamlit as st
from fpdf import FPDF
import os

# === STEP 1: Dataset Loading (Preloaded sample datasets) ===

def load_sample_dataset(domain):
    if domain == 'Agriculture':
        url = 'https://raw.githubusercontent.com/OpenDataDE/State_Crop_Production/main/State_Crop_Production_2019.csv'
        df = pd.read_csv(url)
        df = df[df['Data Item'].str.contains('CORN')]
        df = df.groupby('Year').agg({'Value': 'sum'}).reset_index()
        df.rename(columns={'Year': 'date', 'Value': 'value'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y')
    
    elif domain == 'Energy':
        url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv'
        df = pd.read_csv(url)
        df.rename(columns={'timestamp': 'date', 'value': 'value'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
    
    elif domain == 'Retail':
        url = 'https://raw.githubusercontent.com/selva86/datasets/master/SuperstoreSales.csv'
        df = pd.read_csv(url)
        df = df.groupby('Order Date').agg({'Sales': 'sum'}).reset_index()
        df.rename(columns={'Order Date': 'date', 'Sales': 'value'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
    
    elif domain == 'Traffic':
        url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Time-Series-Analysis-with-R/master/Chapter%2005/data/traffic.csv'
        df = pd.read_csv(url)
        df.rename(columns={'time': 'date', 'traffic_volume': 'value'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
    
    else:
        raise ValueError('Unknown domain')

    return df

# === STEP 2: Preprocessing ===

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    return df

# === STEP 3: Forecasting Module (Prophet) ===

def forecast_with_prophet(df):
    data = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast, model

# === STEP 4: Anomaly Detection Module (Isolation Forest) ===

def detect_anomalies(df):
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df[['value']])
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    return df

# === STEP 5: Benchmarking Function ===

def benchmark_model(y_true, y_pred, anomalies_true, anomalies_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    precision = precision_score(anomalies_true, anomalies_pred, zero_division=0)
    recall = recall_score(anomalies_true, anomalies_pred, zero_division=0)
    f1 = f1_score(anomalies_true, anomalies_pred, zero_division=0)
    return {'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

# === STEP 6: PDF Report Generation ===

def export_report_to_pdf(domain, benchmark):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"MultiScopeAI Report - {domain}", ln=True, align='C')
    pdf.cell(200, 10, txt="Benchmarking Results:", ln=True, align='L')
    for key, value in benchmark.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.4f}", ln=True, align='L')

    if not os.path.exists('report'):
        os.makedirs('report')
    plt.savefig("report/forecast.png")
    plt.savefig("report/anomaly.png")

    try:
        pdf.image("report/forecast.png", x=10, y=50, w=180)
        pdf.add_page()
        pdf.image("report/anomaly.png", x=10, y=50, w=180)
    except:
        pdf.cell(200, 10, txt="Image export failed.", ln=True, align='L')

    pdf.output("report/MultiScopeAI_Report.pdf")

# === STEP 7: Streamlit App ===

def run_streamlit_app():
    st.title('MultiScopeAI - Forecasting & Anomaly Detection')

    domain = st.selectbox('Select Domain', ['Agriculture', 'Energy', 'Retail', 'Traffic'])
    df = load_sample_dataset(domain)
    df = preprocess_data(df)

    st.subheader('Data Preview')
    st.write(df.head())

    st.subheader('Forecasting Results')
    forecast, model = forecast_with_prophet(df)
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader('Anomaly Detection Results')
    df_anomaly = detect_anomalies(df)
    fig2, ax = plt.subplots()
    sns.scatterplot(data=df_anomaly, x='date', y='value', hue='is_anomaly', palette='deep', ax=ax)
    st.pyplot(fig2)

    st.subheader('Benchmarking (Quick)')
    merged = df_anomaly.merge(forecast[['ds', 'yhat']], left_on='date', right_on='ds', how='left')
    benchmark = benchmark_model(
        merged['value'].fillna(0),
        merged['yhat'].fillna(0),
        merged['is_anomaly'],
        merged['is_anomaly']
    )
    st.write(benchmark)

    if st.button('Export PDF Report'):
        export_report_to_pdf(domain, benchmark)
        st.success('PDF Report generated! Check the report folder.')

# === RUN STREAMLIT ===
if __name__ == '__main__':
    run_streamlit_app()
