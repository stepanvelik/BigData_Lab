
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# ----------------------------- 1. Загрузка данных -----------------------------
def load_data(filepath):
    data = pd.read_csv(filepath, header=None, names=['f1', 'f2', 'f3', 'f4', 'class'])
    X = data.iloc[:, :4].values
    y = data.iloc[:, 4].values
    return X, y

# ----------------------------- 2. Разделение по условию задачи -----------------
def split_as_specified(X, y):
    indices_class0 = np.where(y == 0)[0]
    indices_class1 = np.where(y == 1)[0]

    # Последние 107 для класса 0
    test_idx0 = indices_class0[-107:]
    train_idx0 = indices_class0[:-107]

    # Последние 119 для класса 1
    test_idx1 = indices_class1[-119:]
    train_idx1 = indices_class1[:-119]

    train_idx = np.concatenate([train_idx0, train_idx1])
    test_idx = np.concatenate([test_idx0, test_idx1])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

# ----------------------------- 3. Обучение и оценка модели --------------------
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

# ----------------------------- 4. Основная часть ------------------------------
def main():
    # Загрузка данных
    X, y = load_data("data_banknote_authentication.txt")
    print(f"Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признака")
    print(f"Распределение классов: 0 -> {np.sum(y==0)}, 1 -> {np.sum(y==1)}")

    # Разделение по условию задачи
    X_train, X_test, y_train, y_test = split_as_specified(X, y)
    print(f"\nОбучающая выборка: {X_train.shape[0]} объектов")
    print(f"Тестовая выборка: {X_test.shape[0]} объектов")
    print(f"  из них класс 0: {np.sum(y_test==0)} (последние 107)")
    print(f"  из них класс 1: {np.sum(y_test==1)} (последние 119)")

    # 4.1. SVM с линейным ядром
    svm_linear = svm.SVC(kernel='linear', C=1.0, random_state=42)
    train_and_evaluate(svm_linear, X_train, y_train, X_test, y_test, name="SVM Linear (C=1)")

    # 4.2. SVM с полиномиальным ядром (степень 3)
    svm_poly = svm.SVC(kernel='poly', degree=3, C=1.0, random_state=42)
    train_and_evaluate(svm_poly, X_train, y_train, X_test, y_test, name="SVM Poly (degree=3, C=1)")

    # 4.3. SVM с RBF ядром (параметры по умолчанию)
    svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    train_and_evaluate(svm_rbf, X_train, y_train, X_test, y_test, name="SVM RBF (C=1, gamma=scale)")

    # 4.4. Поиск оптимальных параметров с помощью кросс-валидации (для RBF)
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

    # 4.5. Дополнительно: попробуем разные значения C для линейного ядра
    print("\n===== Влияние параметра C для линейного SVM =====")
    for C in [0.01, 0.1, 1, 10, 100]:
        svm_lin = svm.SVC(kernel='linear', C=C, random_state=42)
        svm_lin.fit(X_train, y_train)
        acc = svm_lin.score(X_test, y_test)
        print(f"C = {C:5.2f} -> точность на тесте: {acc:.4f}")

    # 4.6. Дополнительно: влияние степени полинома
    print("\n===== Влияние степени полиномиального ядра =====")
    for degree in [2, 3, 4]:
        svm_poly_d = svm.SVC(kernel='poly', degree=degree, C=1.0, random_state=42)
        svm_poly_d.fit(X_train, y_train)
        acc = svm_poly_d.score(X_test, y_test)
        print(f"degree = {degree} -> точность на тесте: {acc:.4f}")

    # 4.7. Визуализация (по первым двум признакам, если нужно - но не обязательно)
    plot_decision_boundary = False
    if plot_decision_boundary:
        # Обучение на первых двух признаках для визуализации
        X_train2 = X_train[:, :2]
        X_test2 = X_test[:, :2]
        best_svm2 = svm.SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], random_state=42)
        best_svm2.fit(X_train2, y_train)
        # Создание сетки
        x_min, x_max = X_train2[:, 0].min() - 1, X_train2[:, 0].max() + 1
        y_min, y_max = X_train2[:, 1].min() - 1, X_train2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = best_svm2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, edgecolors='k', cmap='coolwarm')
        plt.xlabel('Признак 1')
        plt.ylabel('Признак 2')
        plt.title('Разделяющая граница SVM (первые два признака)')
        plt.show()


if __name__ == "__main__":
    main()