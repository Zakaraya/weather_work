"""
Файл weather.py служит для анализа погоды, данные которой находятся в файле weatherSPB_2016_2019.xls.

Функции:
    month_name_rus - служит для перевода численного значения месяца в буквенное.
    min_max_search - служит для поиска минимального и максимального значения во входных данных по каждому месяцу в году.
    forecast_weather - служит для прогнозирования температуры на январь 2020 года.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

# Игнорирование ошибок
warnings.simplefilter(action='ignore', category=FutureWarning)

data_weather = pd.read_excel('./weatherSPB_2016_2019.xls', sheet_name='weather',
                             parse_dates=['date_time'],
                             usecols=['date_time', 'T', 'U', 'Td'])

# Конвертация поля date_time из типа object в тип datetime64
data_weather['date_time'] = data_weather['date_time'].astype('datetime64[ns]')

# Удаление Series с пустыми значениями
data_weather = data_weather.dropna(how='any', axis=0)

# Запись данных в новые поля month и year
data_weather['month'] = data_weather['date_time'].dt.month
data_weather['year'] = data_weather['date_time'].dt.year


def month_name_rus(num):
    """Функция, для преобразования числа в название месяца на русском языке"""

    ru = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь',
          'октябрь', 'ноябрь', 'декабрь']
    return ru[num - 1]


def min_max_search(df):
    """Функция для поиска максимального и минимального значения температуры из входного DataFrame"""

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
    """Функиця прогнозирования температуры на январь 2020 года"""

    # Создание нового DataFrame для обучения
    jan_df = df[(df['month'] == 1)]
    jan_df = jan_df.dropna(how='any', axis=0)

    # Создание тестовых и обучаемых данных
    features = jan_df.drop(['T', 'date_time', 'month', 'year'], axis=1)
    temp = jan_df['T']
    X_train, X_test, Y_train, Y_test = train_test_split(features, temp, test_size=0.2, shuffle=False)

    # Создание и тренировка объекта линейной регрессии
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Создание DataFrame с датами января
    input_df_result = pd.DataFrame(pd.date_range(start='1/1/2020', end='31/1/2020'), columns=['Дата'])

    # Выбор 31 дня из предсказанных данных
    pred = regressor.predict(X_test)[:31]

    # Добавление данных о предсказанной температуре
    input_df_result = pd.concat([input_df_result, pd.DataFrame(pred, columns=["Температура"])], axis=1)

    # Преобразование вывода температуры
    input_df_result['Температура'] = input_df_result['Температура'].map('{:,.1f}'.format).astype('float64')

    # Запись в Excel
    input_df_result.to_excel('./predict_weather.xls', index=False)

    # Средняя квадратичная ошибка
    mse = mean_squared_error(Y_test, regressor.predict(X_test))
    print("MSE", mse)

    return input_df_result


# Вызов функций
min_max_search(data_weather)
forecast_weather(data_weather)

# Построение графика
forecast_weather(data_weather).plot(x='Дата', y=['Температура'])
plt.show()
