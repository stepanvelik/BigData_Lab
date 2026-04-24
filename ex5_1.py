import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import time


# ========================= Часть 1: Banknote Authentication (SVM) =========================

def load_banknote_data(filepath):
    data = pd.read_csv(filepath, header=None, names=['f1', 'f2', 'f3', 'f4', 'class'])
    X = data.iloc[:, :4].values
    y = data.iloc[:, 4].values
    return X, y


def split_as_specified_banknote(X, y):
    indices_class0 = np.where(y == 0)[0]
    indices_class1 = np.where(y == 1)[0]
    test_idx0 = indices_class0[-107:]
    train_idx0 = indices_class0[:-107]
    test_idx1 = indices_class1[-119:]
    train_idx1 = indices_class1[:-119]
    train_idx = np.concatenate([train_idx0, train_idx1])
    test_idx = np.concatenate([test_idx0, test_idx1])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_and_evaluate(clf, X_train, y_train, X_test, y_test, name=""):
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    start_pred = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start_pred
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Train time : {train_time:.4f} сек")
    print(f"Predict time: {pred_time:.4f} сек")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    return acc


def run_banknote_svm():
    print("=" * 60)
    print("ЧАСТЬ 1: SVM на наборе Banknote Authentication")
    print("=" * 60)
    X, y = load_banknote_data("data_banknote_authentication.txt")
    print(f"Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признака")
    print(f"Распределение классов: 0 -> {np.sum(y == 0)}, 1 -> {np.sum(y == 1)}")
    X_train, X_test, y_train, y_test = split_as_specified_banknote(X, y)
    print(f"\nОбучающая выборка: {X_train.shape[0]} объектов")
    print(f"Тестовая выборка: {X_test.shape[0]} объектов")
    print(f"  из них класс 0: {np.sum(y_test == 0)} (последние 107)")
    print(f"  из них класс 1: {np.sum(y_test == 1)} (последние 119)")

    # Линейное ядро
    svm_linear = svm.SVC(kernel='linear', C=1.0, random_state=42)
    train_and_evaluate(svm_linear, X_train, y_train, X_test, y_test, name="SVM Linear (C=1)")

    # Полиномиальное ядро (degree=3)
    svm_poly = svm.SVC(kernel='poly', degree=3, C=1.0, random_state=42)
    train_and_evaluate(svm_poly, X_train, y_train, X_test, y_test, name="SVM Poly (degree=3, C=1)")

    # RBF ядро по умолчанию
    svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    train_and_evaluate(svm_rbf, X_train, y_train, X_test, y_test, name="SVM RBF (C=1, gamma=scale)")

    # GridSearchCV для RBF
    print("\n===== Поиск оптимальных параметров C и gamma (GridSearchCV) =====")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(svm.SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"Лучшая точность на кросс-валидации: {grid.best_score_:.4f}")
    best_svm = grid.best_estimator_
    train_and_evaluate(best_svm, X_train, y_train, X_test, y_test, name="SVM RBF (оптимальные параметры)")

    # Влияние C для линейного SVM
    print("\n===== Влияние параметра C для линейного SVM =====")
    for C in [0.01, 0.1, 1, 10, 100]:
        svm_lin = svm.SVC(kernel='linear', C=C, random_state=42)
        svm_lin.fit(X_train, y_train)
        acc = svm_lin.score(X_test, y_test)
        print(f"C = {C:5.2f} -> точность на тесте: {acc:.4f}")

    # Влияние степени полиномиального ядра
    print("\n===== Влияние степени полиномиального ядра =====")
    for degree in [2, 3, 4]:
        svm_poly_d = svm.SVC(kernel='poly', degree=degree, C=1.0, random_state=42)
        svm_poly_d.fit(X_train, y_train)
        acc = svm_poly_d.score(X_test, y_test)
        print(f"degree = {degree} -> точность на тесте: {acc:.4f}")


# ========================= Часть 2: Adult Income (SVM на разных объемах, Naive Bayes) =========================

def preprocess_adult(df):
    """Предобработка датасета Adult Income"""
    # Удаление строк с пропусками
    df = df.dropna()
    # Кодирование целевой переменной 'class' в 0/1 (>50K = 1, <=50K = 0)
    le = LabelEncoder()
    y = le.fit_transform(df['class'])  # '>50K' -> 1, '<=50K' -> 0
    # Удаляем целевую переменную из признаков
    X_df = df.drop(columns=['class'])
    # Категориальные признаки -> one-hot encoding
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)
    # Преобразование в numpy
    X = X_encoded.values.astype(np.float64)
    return X, y


def run_adult_experiments():
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: Эксперименты на наборе Adult Income")
    print("=" * 60)

    # Загрузка датасета (может занять 10-20 секунд)
    print("Загрузка набора данных Adult Income (fetch_openml)...")
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    df_adult = adult.frame
    print(f"Размер исходного датасета: {df_adult.shape}")
    print("Распределение классов:")
    print(df_adult['class'].value_counts())

    # Предобработка
    X, y = preprocess_adult(df_adult)
    print(f"\nПосле предобработки: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Распределение классов: 0 -> {np.sum(y == 0)}, 1 -> {np.sum(y == 1)}")

    # Нормализация признаков (важно для SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разбиение на обучающую (80%) и тестовую (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Обучающая выборка: {X_train_full.shape[0]}")
    print(f"Тестовая выборка: {X_test.shape[0]}")

    # ----- 2.1. Кривая обучения SVM на разных объемах данных -----
    print("\n----- 2.1. SVM на разных долях обучающей выборки  -----")
    train_sizes = [0.05, 0.1, 0.2, 0.5, 0.8]  # доли от train_full
    train_scores = []
    test_scores = []

    for frac in train_sizes:
        size = int(len(X_train_full) * frac)
        X_train_sub = X_train_full[:size]
        y_train_sub = y_train_full[:size]
        # Линейный SVM для скорости
        clf = svm.LinearSVC(C=1.0, random_state=42, max_iter=5000)
        clf.fit(X_train_sub, y_train_sub)
        train_acc = clf.score(X_train_sub, y_train_sub)
        test_acc = clf.score(X_test, y_test)
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        print(
            f"Доля обучения: {frac * 100:5.1f}% ({size} объектов) -> Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    # Построение графика learning curve
    plt.figure(figsize=(8, 5))
    plt.plot([f * 100 for f in train_sizes], train_scores, 'o-', label='Точность на обучении')
    plt.plot([f * 100 for f in train_sizes], test_scores, 's-', label='Точность на тесте')
    plt.xlabel('Размер обучающей выборки (%)')
    plt.ylabel('Accuracy')
    plt.title('Learning curve: SVM (линейное ядро) на Adult Income')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- 2.2. Наивные байесовские классификаторы с разными alpha -----
    print("\n----- 2.2. Сравнение Наивных байесовских классификаторов -----")

    # Для GaussianNB alpha не используется (только MultinomialNB и BernoulliNB)
    alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    # 2.2.1 GaussianNB (не имеет alpha)
    gnb = GaussianNB()
    gnb.fit(X_train_full, y_train_full)
    y_pred_gnb = gnb.predict(X_test)
    acc_gnb = accuracy_score(y_test, y_pred_gnb)
    print(f"\nGaussianNB (без сглаживания): Accuracy = {acc_gnb:.4f}")
    print(classification_report(y_test, y_pred_gnb, digits=4, zero_division=0))

    # 2.2.2 MultinomialNB требует неотрицательных признаков – масштабируем в [0,1]
    # Применяем MinMaxScaler для MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    scaler_mm = MinMaxScaler()
    X_train_mm = scaler_mm.fit_transform(X_train_full)
    X_test_mm = scaler_mm.transform(X_test)

    print("\nMultinomialNB с разными alpha:")
    for a in alpha_values:
        mnb = MultinomialNB(alpha=a)
        mnb.fit(X_train_mm, y_train_full)
        acc = mnb.score(X_test_mm, y_test)
        print(f"alpha = {a:5.2f} -> Accuracy: {acc:.4f}")

    # 2.2.3 BernoulliNB (требует бинарных признаков) – бинаризуем по медиане
    # Бинаризация: значение > медианы -> 1, иначе 0
    median = np.median(X_train_full, axis=0)
    X_train_bin = (X_train_full > median).astype(int)
    X_test_bin = (X_test > median).astype(int)

    print("\nBernoulliNB с разными alpha:")
    for a in alpha_values:
        bnb = BernoulliNB(alpha=a)
        bnb.fit(X_train_bin, y_train_full)
        acc = bnb.score(X_test_bin, y_test)
        print(f"alpha = {a:5.2f} -> Accuracy: {acc:.4f}")

    # 2.2.4 Лучший результат среди всех NB
    # Для полноты – также обучим лучший MultinomialNB и BernoulliNB с оптимальным alpha = 0.01
    best_mnb = MultinomialNB(alpha=0.01)
    best_mnb.fit(X_train_mm, y_train_full)
    best_bnb = BernoulliNB(alpha=0.01)
    best_bnb.fit(X_train_bin, y_train_full)
    print("\n=== Лучшие модели NB ===")
    print(f"MultinomialNB (alpha=0.01) test accuracy: {best_mnb.score(X_test_mm, y_test):.4f}")
    print(f"BernoulliNB (alpha=0.01) test accuracy: {best_bnb.score(X_test_bin, y_test):.4f}")

    # Сравнение с линейным SVM (для контекста)
    svm_lin_final = svm.LinearSVC(C=1.0, random_state=42, max_iter=5000)
    svm_lin_final.fit(X_train_full, y_train_full)
    svm_acc = svm_lin_final.score(X_test, y_test)
    print(f"\nЛинейный SVM (C=1) на тех же данных: test accuracy = {svm_acc:.4f}")

    return


# ========================= Основной запуск =========================
if __name__ == "__main__":
    # Часть 1: Banknote Authentication
    run_banknote_svm()

    # Часть 2: Adult Income (SVM learning curve + Naive Bayes)
    run_adult_experiments()

    print("\nВсе эксперименты завершены.")