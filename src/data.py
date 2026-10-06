import numpy as np
import pandas as pd


from config import TRAIN_PATH, TEST_PATH


def load_data():
    """Загружает обучающую и тестовую выборки."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def split_features_target(train_df, test_df):
    """Разделяет обучающие признаки, целевую переменную и тестовые признаки."""

    X_train = train_df.drop("SalePrice", axis=1)
    y_train = train_df["SalePrice"]
    X_test = test_df.copy()

    return X_train, y_train, X_test


def log_transform_target(y):
    """Применяет логарифмическое преобразование log1p к целевой переменной."""

    return np.log1p(y)