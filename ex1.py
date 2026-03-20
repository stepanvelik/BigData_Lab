import pandas as pd
import numpy as np

# =========================
# ЗАДАНИЕ 1. Анализ выбросов
# =========================
df = pd.read_csv("turkiye-student-evaluation_generic.csv")

cols = [c for c in df.columns if "Q" in c]

# функция удаления выбросов (метод интерквартильного размаха)
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[~((data < lower) | (data > upper)).any(axis=1)]


clean_df = remove_outliers(df[cols])

print("Размер до:", df.shape)
print("После удаления:", clean_df.shape)

# =========================
# ЗАДАНИЕ 2. Корреляции
# =========================

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = clean_df.corr()

plt.figure(figsize=(12,10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Матрица корреляций")
plt.show()

# =========================
# ЗАДАНИЕ 3. Сравнение корреляций по предметам
# =========================
subjects = df['class'].unique()

for s in subjects[:3]:   # первые 3 предмета для примера
    subset = df[df['class'] == s][cols]
    print(f"\nПредмет {s}")
    print(subset.corr().mean().mean())

# =========================
# ЗАДАНИЕ 4. Анализ преподавателей
# =========================

teacher_stats = df.groupby("instr")[cols].mean()
print(teacher_stats)
print("\nОписательная статистика преподавателей:")
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(teacher_stats.describe())

teacher_stats["rating"] = teacher_stats.mean(axis=1)

print("\nРейтинг преподавателей:")
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(teacher_stats.sort_values("rating", ascending=False))

# =========================
# ЗАДАНИЕ 5. Анализ предметов
# =========================

subject_stats = df.groupby("class")[cols].mean()

subject_stats["rating"] = subject_stats.mean(axis=1)

print("\nРейтинг предметов:")
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(subject_stats.sort_values("rating", ascending=False))