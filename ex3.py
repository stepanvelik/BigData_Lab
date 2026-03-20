import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# =========================
# 1. Загрузка данных
# =========================
data = pd.read_csv("kc_house_data.csv")

print("Исходная размерность:", data.shape)

# =========================
# 2. Предобработка (как в методичке)
# =========================
# удаляем лишние признаки
data = data.drop(['id', 'date', 'zipcode', 'lat', 'long', 'sqft_basement'], axis=1)

# удаляем пропуски
data = data.dropna()

print("После очистки:", data.shape)

# сохраняем названия признаков
features = data.columns

# =========================
# 3. PCA БЕЗ стандартизации
# =========================
print("\n===== PCA БЕЗ СТАНДАРТИЗАЦИИ =====")

pca_raw = PCA(svd_solver='full')
pca_raw.fit(data)

print("\nГлавные компоненты:")
print(pca_raw.components_)

print("\nДоля объясненной дисперсии:")
print(pca_raw.explained_variance_ratio_)

# накопленная дисперсия
var_raw = np.cumsum(pca_raw.explained_variance_ratio_)
print("\nНакопленная дисперсия:")
print(var_raw)

# =========================
# 4. PCA СО стандартизацией
# =========================
print("\n===== PCA СО СТАНДАРТИЗАЦИЕЙ =====")

data_scaled = scale(data)

pca_scaled = PCA(svd_solver='full')
pca_scaled.fit(data_scaled)

print("\nГлавные компоненты:")
print(pca_scaled.components_)

print("\nДоля объясненной дисперсии:")
print(pca_scaled.explained_variance_ratio_)

# накопленная дисперсия
var_scaled = np.cumsum(pca_scaled.explained_variance_ratio_)
print("\nНакопленная дисперсия:")
print(var_scaled)

# =========================
# 5. График (как в методичке)
# =========================
plt.figure()
plt.plot(range(1, len(var_scaled) + 1), var_scaled)
plt.xlabel("Количество главных компонент")
plt.ylabel("Накопленная доля дисперсии")
plt.title("PCA (стандартизированные данные)")
plt.grid()
plt.show()

# =========================
# 6. Какие признаки входят в компоненты
# =========================
print("\n===== АНАЛИЗ КОМПОНЕНТ =====")

for i, component in enumerate(pca_scaled.components_):
    print(f"\nГлавная компонента {i + 1}:")

    # сортируем признаки по вкладу
    comp_dict = dict(zip(features, component))
    sorted_features = sorted(comp_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # выводим топ-5 признаков
    for feature, value in sorted_features[:5]:
        print(f"{feature}: {round(value, 4)}")

# =========================
# 7. Сколько компонент нужно (>= 80%)
# =========================
k = np.argmax(var_scaled >= 0.8) + 1
print("\nКоличество компонент для >= 80% дисперсии:", k)