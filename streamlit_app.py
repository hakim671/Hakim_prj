# файл: app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title="Прогноз курса сомони/доллар", layout="wide")

# Загрузка данных
@st.cache_data
def load_data():
    # Заменить на свой путь к данным
    df = pd.read_csv('USD_TJS Historical Data (1).csv') 
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Функция RMSE
def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

# Основной код
df = load_data()

# Обработка данных для Prophet
df_prophet = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})

# Разделение на train и val
w_hours = 120
train_pr = df_prophet.iloc[w_hours:]
val_pr = df_prophet.iloc[:w_hours]

# Обучение модели
model = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.001, n_changepoints=7)
model.fit(train_pr)

# Создание прогноза
future = model.make_future_dataframe(periods=w_hours, freq='D', include_history=False)
forecast = model.predict(future)

# Вычисление ошибки
val_actual = val_pr['y'].values
val_pred = forecast['yhat'].values
model_rmse = rmse(val_actual, val_pred)

st.sidebar.title("Меню")
page = st.sidebar.radio("Выберите страницу", ["📈 Презентация модели", "🔮 Прогнозирование"])

if page == "📈 Презентация модели":
    st.title("📈 Прогноз курса сомони/доллар")
    st.write("""
    - **Датасет:** 
            - **Источник:** investing.com
            - **Длина:** 1339
            - **Дата:** от 2020 до 2025
            """)
    st.write("""
    Модель основана на библиотеке **Prophet** от Facebook.
    
    - **Тип модели:** Временной ряд, Prophet(Facebook)
    - **Особенности:** Без сезонности, 7 точек изменения тренда
    - **Метрика качества:** RMSE
    """)

    st.metric(label="RMSE на валидации", value=f"{model_rmse:.2f}")

    st.subheader("График настоящих цен")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Price'], label='Реальные цены')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Курс сомони/доллар')
    ax.legend()
    st.pyplot(fig)

elif page == "🔮 Прогнозирование":
    model = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.001, n_changepoints=7)
    model.fit(df_prophet)

    # Создание прогноза
    future = model.make_future_dataframe(periods=w_hours, freq='D', include_history=False)
    forecast = model.predict(future)
    st.title("🔮 Прогнозирование курса сомони/доллар")

    st.write("Модель предсказывает курс на 120 дней вперед.")

    st.subheader("Выберите день для прогноза")

    # Ползунок для выбора дня
    day_selected = st.slider("День прогноза", min_value=1, max_value=w_hours, value=1)

    # Получение прогноза на выбранный день
    selected_forecast = forecast.iloc[day_selected - 1]  # -1, потому что индексация с нуля

    st.write(f"### Прогноз на день {day_selected}:")
    st.metric(label="Прогнозируемый курс", value=f"{selected_forecast['yhat']:.4f} сомони")
    st.caption(f"Диапазон от {selected_forecast['yhat_lower']:.4f} до {selected_forecast['yhat_upper']:.4f} сомони")

    st.subheader("График прогноза")
    fig2 = model.plot(forecast)
    st.pyplot(fig2)

    st.subheader("Таблица прогноза")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Дата',
        'yhat': 'Прогноз',
        'yhat_lower': 'Нижняя граница',
        'yhat_upper': 'Верхняя граница'
    }))

