import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# РАБОТА №3.6
# Нейронные сети
# Загрузка данных
data = pd.read_csv("kc_house_data.csv")
# Просмотр данных
print("Первые строки данных:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())
# Удаление ненужных столбцов
# id и date не участвуют в обучении
data = data.drop(columns=['id', 'date'])
# Цена дома — целевая переменная
X = data.drop(columns=['price'])
y = data['price']
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание нейронной сети
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
# Обучение модели
print("\nОбучение модели...")
model.fit(X_train, y_train)
# Предсказание
y_pred = model.predict(X_test)

# Анализ модели по СКО


mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("\nСреднеквадратичная ошибка (MSE):")
print(mse)

print("\nСреднеквадратичное отклонение (RMSE):")
print(rmse)


# Сравнение реальных и предсказанных значений


plt.figure(figsize=(10, 6))

plt.plot(
    y_test.values[:100],
    label='Реальные значения'
)

plt.plot(
    y_pred[:100],
    label='Предсказанные значения'
)

plt.title(
    'Сравнение реальных\n'
    'и предсказанных значений'
)

plt.xlabel('Номер объекта')

plt.ylabel('Цена дома')

plt.legend()

plt.grid(True)

plt.show()


# График ошибки обучения

plt.figure(figsize=(8, 5))

plt.plot(model.loss_curve_)

plt.title('График ошибки обучения')

plt.xlabel('Итерация')

plt.ylabel('Ошибка')

plt.grid(True)

plt.show()