from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from config import (
    DECISION_TREE_PARAMS,
    ELASTICNET_PARAMS,
    ELASTICNET_MAX_ITER,
    KNN_PARAMS,
    LASSO_PARAMS,
    LASSO_MAX_ITER,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    RIDGE_PARAMS,
    CATBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    CATBOOST_ONEHOT_PARAMS,
    VOTING_LASSO_MAX_ITER,
    N_JOBS,
    N_SPLITS,
)


def build_linear_regression():
    """Создает базовую модель Linear Regression."""

    return LinearRegression()


def build_ridge():
    """Создает модель Ridge Regression с настроенными гиперпараметрами."""

    return Ridge(**RIDGE_PARAMS)


def build_lasso(max_iter=LASSO_MAX_ITER):
    """Создает модель Lasso Regression с настроенными гиперпараметрами."""

    return Lasso(
        **LASSO_PARAMS,
        random_state=RANDOM_STATE,
        max_iter=max_iter,
    )


def build_elasticnet():
    """Создает модель ElasticNet с настроенными гиперпараметрами."""

    return ElasticNet(
        **ELASTICNET_PARAMS,
        random_state=RANDOM_STATE,
        max_iter=ELASTICNET_MAX_ITER,
    )


def build_knn():
    """Создает модель K-Nearest Neighbors с настроенными гиперпараметрами."""

    return KNeighborsRegressor(**KNN_PARAMS)



def build_decision_tree():
    """Создает Decision Tree Regressor с настроенными гиперпараметрами."""

    return DecisionTreeRegressor(
        **DECISION_TREE_PARAMS,
        random_state=RANDOM_STATE,
    )


def build_random_forest():
    """Создает Random Forest Regressor с настроенными гиперпараметрами."""

    return RandomForestRegressor(
        **RANDOM_FOREST_PARAMS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def build_catboost(numeric_features, categorical_features):
    """Создает CatBoost Regressor с настроенными гиперпараметрами."""

    cat_features_idx = list(
        range(
            len(numeric_features),
            len(numeric_features) + len(categorical_features),
        )
    )

    return CatBoostRegressor(
        **CATBOOST_PARAMS,
        random_state=RANDOM_STATE,
        cat_features=cat_features_idx,
        thread_count=N_JOBS,
    )


def build_catboost_onehot():
    """Создает CatBoost Regressor для данных после OneHot Encoding."""

    return CatBoostRegressor(
        **CATBOOST_ONEHOT_PARAMS,
        random_state=RANDOM_STATE,
        thread_count=N_JOBS,
    )


def build_lightgbm():
    """Создает LightGBM Regressor с настроенными гиперпараметрами."""

    return LGBMRegressor(
        **LIGHTGBM_PARAMS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def build_xgboost():
    """Создает XGBoost Regressor с настроенными гиперпараметрами."""

    return XGBRegressor(
        **XGBOOST_PARAMS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def build_voting_regressor():
    """Создает Voting Regressor из пяти настроенных моделей."""

    estimators = [
        ("catboost", build_catboost_onehot()),
        ("xgboost", build_xgboost()),
        ("lightgbm", build_lightgbm()),
        ("random_forest", build_random_forest()),
        ("lasso", build_lasso(max_iter=VOTING_LASSO_MAX_ITER)),
    ]

    return VotingRegressor(
        estimators=estimators,
        n_jobs=N_JOBS,
    )


def build_boosting_voting_regressor():
    """Создает Voting Regressor из трех boosting-моделей."""

    estimators = [
        ("catboost", build_catboost_onehot()),
        ("xgboost", build_xgboost()),
        ("lightgbm", build_lightgbm()),
    ]

    return VotingRegressor(
        estimators=estimators,
        n_jobs=N_JOBS,
    )


def build_stacking_regressor(preprocessor):
    """Создает Stacking Regressor из трех boosting-моделей."""

    estimators = [
        ("catboost", Pipeline([
                ("preprocessor", preprocessor),
                ("model", build_catboost_onehot()),
            ])),
        ("xgboost", Pipeline([
                ("preprocessor", preprocessor),
                ("model", build_xgboost()),
            ])),
        ("lightgbm", Pipeline([
                ("preprocessor", preprocessor),
                ("model", build_lightgbm()),
            ])),
    ]

    return StackingRegressor(
        estimators=estimators,
        final_estimator=build_ridge(),
        cv=N_SPLITS,
        n_jobs=N_JOBS,
    )


def get_model(
    model_name,
    numeric_features=None,
    categorical_features=None,
    preprocessor=None,
):
    """Создает модель по имени из config."""

    model_builders = {
        "linear_regression": build_linear_regression,
        "ridge": build_ridge,
        "lasso": build_lasso,
        "elasticnet": build_elasticnet,
        "knn": build_knn,
        "decision_tree": build_decision_tree,
        "random_forest": build_random_forest,
        "lightgbm": build_lightgbm,
        "xgboost": build_xgboost,
        "catboost_onehot": build_catboost_onehot,
        "voting_regressor": build_voting_regressor,
        "boosting_voting_regressor": build_boosting_voting_regressor,
    }

    if model_name in model_builders:
        return model_builders[model_name]()

    if model_name == "catboost":
        return build_catboost(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

    if model_name == "stacking_regressor":
        return build_stacking_regressor(
            preprocessor=preprocessor,
        )

    raise ValueError(
        f"Неизвестная модель: {model_name}"
    )