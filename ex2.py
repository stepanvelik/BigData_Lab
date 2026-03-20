import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 1. Загрузка данных
# =========================

data = pd.read_csv("kc_house_data.csv")

# признаки из задания
features = [
    'sqft_living',
    'sqft_lot',
    'waterfront',
    'sqft_above',
    'yr_built',
    'yr_renovated',
    'sqft_living15',
    'sqft_lot15',
    'floors'
]

target = 'price'

df = data[features + [target]].copy()

# =========================
# 2. Анализ пропусков
# =========================


# восстановление пропусков
df = df.fillna(df.mean())

# =========================
# 3. Удаление выбросов (IQR)
# =========================

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# =========================
# 4. Feature extraction
# кодирование floors
# =========================

df = pd.get_dummies(df, columns=['floors'], prefix='floors')

# =========================
# 5. Разделение признаков
# =========================

X = df.drop('price', axis=1)
y = df['price']

# =========================
# 6. Разделение выборки
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. Стандартизация данных
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. Построение моделей
# =========================

linear_model = LinearRegression()
ridge_model = Ridge(alpha=3)
lasso_model = Lasso(alpha=120)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# =========================
# 9. Предсказания
# =========================

pred_linear = linear_model.predict(X_test)
pred_ridge = ridge_model.predict(X_test)
pred_lasso = lasso_model.predict(X_test)


# =========================
# 10. Метрики качества
# =========================

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    #Корень из среднеквадратичной ошибки.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return r2, rmse


r2_lin, rmse_lin = metrics(y_test, pred_linear)
r2_ridge, rmse_ridge = metrics(y_test, pred_ridge)
r2_lasso, rmse_lasso = metrics(y_test, pred_lasso)

print("\nКачество регрессии")
print("-----------------------------")

print("Линейная регрессия")
print("R2 =", r2_lin)
print("RMSE =", rmse_lin)

print("\nГребневая регрессия (α = 3)")
print("R2 =", r2_ridge)
print("RMSE =", rmse_ridge)

print("\nLasso (α = 120)")
print("R2 =", r2_lasso)
print("RMSE =", rmse_lasso)

# =========================
# 11. Коэффициенты регрессии
# =========================

coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Linear Regression": linear_model.coef_,
    "Ridge Regression": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\nКоэффициенты регрессии")
print(coef_table)

# =========================
# 12. Визуализация
# =========================

plt.scatter(y_test, pred_linear)

plt.xlabel("Реальная цена")
plt.ylabel("Предсказанная цена")

plt.title("Линейная регрессия")

plt.show()