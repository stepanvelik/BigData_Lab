# Лабораторная работа 3.4

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# 1. Загрузка данных
data = fetch_20newsgroups(subset='all')
X = data.data
y = data.target
target_names = data.target_names

# 2. Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Векторизация текста
vectorizer_multi = CountVectorizer()
X_train_multi = vectorizer_multi.fit_transform(X_train)
X_test_multi = vectorizer_multi.transform(X_test)

# Bernoulli
vectorizer_bern = CountVectorizer(binary=True)
X_train_bern = vectorizer_bern.fit_transform(X_train)
X_test_bern = vectorizer_bern.transform(X_test)

# 4. Обучение моделей
model_multi = MultinomialNB(alpha=1.0)
model_multi.fit(X_train_multi, y_train)

model_bern = BernoulliNB(alpha=1.0)
model_bern.fit(X_train_bern, y_train)

# 5. Проверка априорных вероятностей
print("\n=== Проверка априоров ===")

model_equal = MultinomialNB(alpha=1.0, fit_prior=False)
model_equal.fit(X_train_multi, y_train)
y_pred_equal = model_equal.predict(X_test_multi)
print("Равные априоры accuracy:", accuracy_score(y_test, y_pred_equal))

model_prior = MultinomialNB(alpha=1.0, fit_prior=True)
model_prior.fit(X_train_multi, y_train)
y_pred_prior = model_prior.predict(X_test_multi)
print("Априоры из данных accuracy:", accuracy_score(y_test, y_pred_prior))

print("\n=== Подбор alpha ===")
alphas = [0.01, 0.1, 0.5, 1.0, 2.0]

for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_multi, y_train)
    y_pred = model.predict(X_test_multi)
    acc = accuracy_score(y_test, y_pred)
    print(f"alpha={alpha}: accuracy={acc:.4f}")

# 7. Сравнение моделей
print("\n=== MultinomialNB ===")
y_pred_multi = model_multi.predict(X_test_multi)
print(classification_report(y_test, y_pred_multi))

print("\n=== BernoulliNB ===")
y_pred_bern = model_bern.predict(X_test_bern)
print(classification_report(y_test, y_pred_bern))

# 8. Распределение документов по рубрикам
print("\n=== Распределение документов ===")
counts = np.bincount(y)

for i, count in enumerate(counts):
    print(f"{target_names[i]}: {count}")