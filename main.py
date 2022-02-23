
import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Prédiction du prix d'un titre bousier avec Prophet")

stocks = ('AAPL', 'AMZN', 'WMT', 'NFLX','MAR','AAL')
selected_stock = st.selectbox('Selectionner le titre à prédir', stocks)

n_days = st.slider('Nombre de jour à prédir:', 1, 365)  # for one week


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Télégargement des données...')
data = load_data(selected_stock)
data.to_csv('out.csv')  
data_load_state.text('Téléchargement des données... fait!')

st.subheader('Données Brutes')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Données de séries chronologiques', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

period = 48 * n_days

future = m.make_future_dataframe(periods=period, freq='30min')

forecast = m.predict(future)

# Show and plot forecast
st.subheader('Données Prédites')
st.write(forecast.tail())

st.write(f'Figure de la prévison pour {n_days} jours')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

