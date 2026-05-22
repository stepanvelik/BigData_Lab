import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# РАБОТА №3.8
# Кластеризация: K-Means и EM
# Распаковка архива
with zipfile.ZipFile("Quake.zip", "r") as zip_ref:
    zip_ref.extractall()
# Загрузка данных
data = pd.read_csv(
    "quake.dat",
    sep=r"\s+",
    engine="python"
)
# Просмотр данных
print("Первые строки данных:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())
# Выбор только числовых признаков
X = data.select_dtypes(include=[np.number])

# Нормализация данных

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Диапазон количества кластеров

K = range(2, 11)

# Списки для хранения результатов

sse = []

ratio_values = []

# Анализ различных значений k

for k in K:
    # K-Means

    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42
    )

    kmeans.fit(X_scaled)

    # Сумма квадратов расстояний (SSE)
    sse.append(kmeans.inertia_)

    # Метки кластеров

    labels = kmeans.labels_

    # Центры кластеров

    centers = kmeans.cluster_centers_

    # Среднее внутрикластерное расстояние

    intra_distance = 0

    for i in range(k):

        cluster_points = X_scaled[labels == i]

        if len(cluster_points) > 0:

            distances = cdist(
                cluster_points,
                [centers[i]]
            )

            intra_distance += np.mean(distances)

    intra_distance /= k

    # Среднее внекластерное расстояние

    inter_distance = np.mean(
        cdist(centers, centers)
    )

    # Отношение расстояний

    ratio = intra_distance / inter_distance

    ratio_values.append(ratio)

# График SSE (метод локтя)


plt.figure(figsize=(8, 5))

plt.plot(
    list(K),
    sse,
    marker='o'
)

plt.title(
    'Сумма квадратов расстояний\n'
    '(Метод локтя)'
)

plt.xlabel('Количество кластеров k')

plt.ylabel('SSE')

plt.grid(True)

plt.show()

# График отношения расстояний

plt.figure(figsize=(8, 5))

plt.plot(
    list(K),
    ratio_values,
    marker='o'
)

plt.title(
    'Отношение внутрикластерного\n'
    'расстояния к внекластерному'
)

plt.xlabel('Количество кластеров k')

plt.ylabel('Отношение расстояний')

plt.grid(True)

plt.show()

# Определение оптимального числа кластеров

optimal_k = list(K)[np.argmin(ratio_values)]

print("\nОптимальное число кластеров:")
print(optimal_k)

# Финальная модель K-Means

kmeans = KMeans(
    n_clusters=optimal_k,
    n_init=20,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# EM алгоритм

em = GaussianMixture(
    n_components=optimal_k,
    n_init=10,
    random_state=42
)

em.fit(X_scaled)

em_labels = em.predict(X_scaled)

# Визуализация кластеров

plt.figure(figsize=(12, 5))

# K-Means

plt.subplot(1, 2, 1)

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmeans_labels
)

plt.title("K-Means")

plt.xlabel("Признак 1")
plt.ylabel("Признак 2")

# EM

plt.subplot(1, 2, 2)

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=em_labels
)

plt.title("EM алгоритм")

plt.xlabel("Признак 1")
plt.ylabel("Признак 2")

plt.tight_layout()

plt.show()