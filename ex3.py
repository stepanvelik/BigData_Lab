import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def print_feature_index(feature_names, title):
    print(f"\n===== {title} =====")
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i:>2}. {feature}")


def print_variance_with_index(explained_ratio, cumulative_ratio):
    print("\nДоля объясненной дисперсии:")
    for i, value in enumerate(explained_ratio, start=1):
        print(f"PC{i:>2}: {value:.10f}")

    print("\nНакопленная дисперсия:")
    for i, value in enumerate(cumulative_ratio, start=1):
        print(f"PC1..PC{i:>2}: {value:.10f}")


def print_top_features(pca_model, feature_names, title, top_n=5):
    print(f"\n===== {title} =====")
    for i, component in enumerate(pca_model.components_):
        print(f"\nГлавная компонента {i + 1}:")
        comp_dict = dict(zip(feature_names, component))
        sorted_features = sorted(
            comp_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feature, value in sorted_features[:top_n]:
            print(f"{feature}: {round(value, 4)}")


def run_theory_reproduction():
    print("\n" + "=" * 70)
    print("БЛОК 1. ВОСПРОИЗВЕДЕНИЕ ТЕОРИИ (GiveMeSomeCredit)")
    print("=" * 70)

    data = np.genfromtxt(
        "cs-training.csv",
        delimiter=",",
        skip_header=1,
        usecols=list(range(1, 11)),
    )
    data = data[~np.isnan(data).any(axis=1)]
    data = scale(data)

    pca = PCA(svd_solver="full")
    pca.fit(data)

    var = np.round(np.cumsum(pca.explained_variance_ratio_), decimals=4)

    theory_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]

    print("Размерность данных:", data.shape)
    print_feature_index(theory_features, "СПИСОК ПРИЗНАКОВ (GiveMeSomeCredit)")
    print_variance_with_index(pca.explained_variance_ratio_, var)

    k = np.argmax(var >= 0.8) + 1
    print("\nКоличество компонент для >= 80% дисперсии:", k)
    print("Вывод: основная доля информации сохраняется примерно в первых 6 компонентах.")

    plt.figure()
    plt.plot(np.arange(1, 11), var)
    plt.xlabel("Количество главных компонент")
    plt.ylabel("Накопленная доля дисперсии")
    plt.title("GiveMeSomeCredit: PCA (стандартизированные данные)")
    plt.grid()
    plt.show()


def run_houses_analysis():
    print("\n" + "=" * 70)
    print("БЛОК 2. ОСНОВНАЯ ЧАСТЬ (датасет домов)")
    print("=" * 70)

    data = pd.read_csv("kc_house_data.csv")
    print("Исходная размерность:", data.shape)

    data = data.drop(["id", "date", "zipcode", "lat", "long", "sqft_basement"], axis=1)
    data = data.dropna()
    print("После очистки:", data.shape)

    features = data.columns
    print_feature_index(features, "СПИСОК ПРИЗНАКОВ (датасет домов)")

    print("\n===== PCA БЕЗ СТАНДАРТИЗАЦИИ =====")
    pca_raw = PCA(svd_solver="full")
    pca_raw.fit(data)
    print("\nГлавные компоненты:")
    print(pca_raw.components_)
    var_raw = np.cumsum(pca_raw.explained_variance_ratio_)
    print_variance_with_index(pca_raw.explained_variance_ratio_, var_raw)

    print("\n===== PCA СО СТАНДАРТИЗАЦИЕЙ =====")
    data_scaled = scale(data)
    pca_scaled = PCA(svd_solver="full")
    pca_scaled.fit(data_scaled)
    print("\nГлавные компоненты:")
    print(pca_scaled.components_)
    var_scaled = np.cumsum(pca_scaled.explained_variance_ratio_)
    print_variance_with_index(pca_scaled.explained_variance_ratio_, var_scaled)

    plt.figure()
    plt.plot(range(1, len(var_scaled) + 1), var_scaled)
    plt.xlabel("")
    plt.ylabel("Накопленная доля дисперсии")
    plt.title("Датасет домов: PCA (стандартизированные данные)")
    plt.grid()
    plt.show()

    print_top_features(
        pca_raw,
        features,
        "ПРИЗНАКИ В ГЛАВНЫХ КОМПОНЕНТАХ (НЕСТАНДАРТИЗИРОВАННЫЕ ДАННЫЕ)",
    )
    print_top_features(
        pca_scaled,
        features,
        "ПРИЗНАКИ В ГЛАВНЫХ КОМПОНЕНТАХ (СТАНДАРТИЗИРОВАННЫЕ ДАННЫЕ)",
    )

    k = np.argmax(var_scaled >= 0.8) + 1
    print("\nКоличество компонент для >= 80% дисперсии:", k)


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000)
    run_theory_reproduction()
    run_houses_analysis()