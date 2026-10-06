from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


none_features = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]


zero_features = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]


def fill_none_features(df, features):
    """Заполняет пропуски значением 'None' в указанных категориальных признаках."""

    df = df.copy()

    for feature in features:
        df[feature] = df[feature].fillna("None")

    return df


def fill_zero_features(df, features):
    """Заполняет пропуски нулём в указанных числовых признаках."""

    df = df.copy()

    for feature in features:
        df[feature] = df[feature].fillna(0)

    return df


def fill_known_missing_values(df):
    """Обрабатывает пропуски с заранее известным смыслом."""

    df = fill_none_features(df, none_features)
    df = fill_zero_features(df, zero_features)

    return df


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    """
    Заполняет пропуски в LotFrontage медианой по Neighborhood.

    Для каждого района медиана LotFrontage рассчитывается на этапе fit().
    Если для конкретного Neighborhood медиана недоступна, используется
    общая медиана LotFrontage, рассчитанная по обучающим данным.
    """

    def fit(self, X, y=None):
        self.neighborhood_medians_ = (
            X.groupby("Neighborhood")["LotFrontage"].median()
        )

        self.global_median_ = X["LotFrontage"].median()

        return self

    def transform(self, X):
        X = X.copy()

        neighborhood_values = X["Neighborhood"].map(
            self.neighborhood_medians_
        )

        X["LotFrontage"] = X["LotFrontage"].fillna(
            neighborhood_values
        )

        X["LotFrontage"] = X["LotFrontage"].fillna(
            self.global_median_
        )

        return X


def get_feature_types(X):
    """Определяет числовые и категориальные признаки."""

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_numeric_pipeline():
    """Создает пайплайн предобработки числовых признаков."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return numeric_pipeline


def build_categorical_pipeline():
    """Создает пайплайн предобработки категориальных признаков."""

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )),
    ])

    return categorical_pipeline


def build_column_transformer(numeric_features, categorical_features):
    """Создает преобразователь для числовых и категориальных признаков."""

    column_transformer = ColumnTransformer([
        (
            "num",
            build_numeric_pipeline(),
            numeric_features,
        ),
        (
            "cat",
            build_categorical_pipeline(),
            categorical_features,
        ),
    ])

    return column_transformer


def build_catboost_column_transformer(
    numeric_features,
    categorical_features,
):
    """Создает преобразователь признаков для CatBoost."""

    column_transformer = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "cat",
            SimpleImputer(strategy="most_frequent"),
            categorical_features,
        ),
    ])

    return column_transformer


def build_catboost_preprocessor(
    numeric_features,
    categorical_features,
):
    """Создает общий пайплайн предобработки признаков для CatBoost."""

    column_transformer = build_catboost_column_transformer(
        numeric_features,
        categorical_features,
    )

    preprocessor = Pipeline([
        ("lot_frontage_imputer", LotFrontageImputer()),
        ("column_transformer", column_transformer),
    ])

    return preprocessor


def build_preprocessor(numeric_features, categorical_features):
    """Создает общий пайплайн предобработки признаков."""

    column_transformer = build_column_transformer(
        numeric_features,
        categorical_features,
    )

    preprocessor = Pipeline([
        ("lot_frontage_imputer", LotFrontageImputer()),
        ("column_transformer", column_transformer),
    ])

    return preprocessor