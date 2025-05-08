import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title="💵 Прогноз курса сомони/доллар", layout="wide")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('USD_TJS Historical Data (1).csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')  # убедимся, что данные в правильном порядке
    return df

# Функция RMSE
def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

# Загрузка и подготовка данных
df = load_data()
df_prophet = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})

# Настройки
forecast_horizon = 120
train_pr = df_prophet.iloc[forecast_horizon:]
val_pr = df_prophet.iloc[:forecast_horizon]

# Prophet для RMSE
model = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.001, n_changepoints=7)
model.fit(train_pr)
future = model.make_future_dataframe(periods=forecast_horizon, freq='D', include_history=False)
forecast = model.predict(future)
val_actual = val_pr['y'].values
val_pred = forecast['yhat'].values
model_rmse = rmse(val_actual, val_pred)

# Боковое меню
st.sidebar.title("📌 Меню")
page = st.sidebar.radio("Выберите страницу", ["📈 Презентация модели", "🔮 Прогнозирование"])

# Страница 1: Презентация модели
if page == "📈 Презентация модели":
    st.title("📈 Прогноз курса сомони/доллар")
    st.markdown("""
    **Описание проекта:**  
    Модель прогнозирует курс валюты **сомони к доллару США** на основе временного ряда с помощью **Prophet** .

    **Данные:**
    - Источник: investing.com
    - Объем: 1339 записей
    - Период: 2020 — 2025

    **Модель:**
    - Метод: Facebook Prophet
    - Особенности: Без сезонности, 7 точек изменения тренда
    - Метрика качества: RMSE
    """)
    st.metric(label="📉 RMSE на валидации", value=f"{model_rmse:.2f} сомони")

    st.subheader("📊 Реальные цены")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Цена'))
    fig.update_layout(title="Исторический курс сомони/доллар", xaxis_title="Дата", yaxis_title="Курс")
    st.plotly_chart(fig, use_container_width=True)

# Страница 2: Прогнозирование
elif page == "🔮 Прогнозирование":
    st.title("🔮 Прогноз курса сомони/доллар")
    st.markdown("Модель **Prophet** предсказывает курс на ближайшие 120 дней.")

    model = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.001, n_changepoints=10)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_horizon, freq='D', include_history=False)
    forecast = model.predict(future)

    day_selected = st.slider("📅 День прогноза", min_value=1, max_value=forecast_horizon, value=1)
    if day_selected >= 60:
        st.warning("⚠️ Долгосрочный прогноз может быть менее точным.")

    selected = forecast.iloc[day_selected - 1]
    st.metric(f"Прогноз на день {day_selected}", f"{selected['yhat']:.4f} сомони")
    st.caption(f"Диапазон от {selected['yhat_lower']:.4f} до {selected['yhat_upper']:.4f}")

    st.subheader("📈 График прогноза")
    fig2 = model.plot(forecast)
    st.pyplot(fig2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet', line=dict(color='royalblue')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Таблица прогноза")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Дата',
        'yhat': 'Прогноз',
        'yhat_lower': 'Нижняя граница',
        'yhat_upper': 'Верхняя граница'
    }))
