import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta


data_weather =  pd.read_excel(r'C:\Users\Борис\JupyterProject\WeatherAPI\weatherSPB_2016_2019.xls', sheet_name = 'weather', 
                              parse_dates = ['date_time'], 
                              usecols = ['date_time', 'T', 'U', 'Td'])

# Конвертация поля date_time из типа object в тип datetime64
data_weather['date_time'] = data_weather['date_time'].astype('datetime64[ns]')

# Удаление Series с пустыми значениями
data_weather = data_weather.dropna(how = 'any', axis = 0)

# Запись данных в новые поля month и year
data_weather['month'] = data_weather['date_time'].dt.month
data_weather['year'] = data_weather['date_time'].dt.year


def month_name_rus(num):
    '''Функция, для преобразования числа в название месяца на русском языке'''
    
    ru = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь',
          'октябрь', 'ноябрь', 'декабрь']
    return ru[num - 1]


def min_max_search(df):
    '''Функция для поиска максимального и минимального значения температуры из входного DataFrame'''
    
    # Создание DataFrame min_max_df и запись в него min и max от T
    min_max_df = df.groupby('month', as_index=False).agg({'T': ['min', 'max']})
    
    # Применение функции month_name_rus к полю month
    min_max_df['month'] = min_max_df['month'].apply(month_name_rus)
    
    # Переименование колонок
    min_max_df.columns = ['Месяц', 'Минимальная T', 'Максимальная T']
    
    # Запись в Excel
    min_max_df.to_excel('./output_min_max_t.xls', index=False)

    return min_max_df


def forecast_weather(df):
    '''Функиця прогнозирования температуры на январь 2020 года'''
    
    # Создание нового DataFrame для обучения
    jan_df = input_df_result = df[(df['month'] == 1)]
    jan_df = jan_df.dropna(how = 'any', axis = 0)
    
    # Создание тестовых и обучаемых данных
    features = jan_df.drop(['T','date_time', 'month', 'year'], axis=1)
    temp = jan_df['T']
    
    X_train, X_test, Y_train, Y_test = train_test_split(features, temp, test_size=0.2, random_state=10)
    
    # Создание и тренировка объекта линейной регрессии
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # Коэффициенты детерминации
    print(regressor.score(X_train, Y_train))
    print(regressor.score(X_test, Y_test))
    print(regressor.intercept_)
    
    # Запись в новый DataFrame input_df_result данных о январе 2019 года
    input_df_result = df[(df['month'] == 1)  & (df['year'] == 2019)]
    
    # Объединение данных по полю date_time
    input_df_result = input_df_result.groupby('date_time', as_index=False)
    
    # Добавление года к дате
    input_df_result = input_df_result.first()['date_time'] + timedelta(days=365)
    pred = regressor.predict(X_test)[:31]
    
    # Добавление данных о предсказанной температуре
    input_df_result = pd.concat([input_df_result, pd.DataFrame(pred, columns=["Температура"])], axis=1)
    
    # Запись в Excel
    input_df_result.to_excel('./predict_weather.xls', index=False)


min_max_search(data_weather)
forecast_weather(data_weather)


# Среднее квадратичное отклонение
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, regressor.predict(X_test))
print("MSE: %.4f" % mse)



