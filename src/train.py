import warnings

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from config import (
    CV_SHUFFLE,
    N_JOBS,
    N_SPLITS,
    RANDOM_STATE,
    R2_SCORING,
    RMSE_SCORING,
)



def build_cv():
    """Создает схему K-Fold cross-validation."""

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=RANDOM_STATE if CV_SHUFFLE else None,
    )

    return cv


def evaluate_estimator(estimator, X, y):
    """Оценивает готовый estimator с помощью cross-validation по RMSE и R²."""

    cv = build_cv()

    # Подавляем только известное предупреждение LightGBM об именах признаков.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMRegressor was fitted with feature names"
            ),
            category=UserWarning,
        )

        rmse_scores = -cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=RMSE_SCORING,
            n_jobs=N_JOBS,
        )

        r2_scores = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=R2_SCORING,
            n_jobs=N_JOBS,
        )

    return rmse_scores, r2_scores


def evaluate_model(model, preprocessor, X, y):
    """
    Оценивает модель с preprocessing с помощью cross-validation.

    Предобработка и модель объединяются в Pipeline, поэтому preprocessing
    обучается отдельно на обучающей части каждого fold.
    """

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return evaluate_estimator(pipeline, X, y)