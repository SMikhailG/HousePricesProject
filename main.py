import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from config import (
    FINAL_MODEL,
    MODELS,
    RESULTS_DIR,
    SUBMISSION_PATH,
)
from src.data import (
    load_data,
    log_transform_target,
    split_features_target,
)
from src.dnn import cross_validate_dnn
from src.features import build_features
from src.models import get_model
from src.preprocessing import (
    build_catboost_preprocessor,
    build_preprocessor,
    fill_known_missing_values,
    get_feature_types,
)
from src.train import (
    evaluate_estimator,
    evaluate_model,
)
from src.utils import (
    set_seed,
    setup_logger,
)


logger = setup_logger()


def main():
    """Запускает полный процесс cross-validation выбранных моделей."""

    set_seed()

    # --------------------
    # Данные
    # --------------------
    train_df, test_df = load_data()

    train_df = fill_known_missing_values(
        train_df
    )
    test_df = fill_known_missing_values(
        test_df
    )

    train_df = build_features(
        train_df
    )
    test_df = build_features(
        test_df
    )

    X_train, y_train, X_test = (
        split_features_target(
            train_df,
            test_df,
        )
    )

    # Все модели оцениваем на log1p(SalePrice),
    # как и в notebook.
    y_train_log = log_transform_target(
        y_train
    )

    numeric_features, categorical_features = (
        get_feature_types(X_train)
    )

    # Стандартный preprocessing:
    # median + scaling + OneHot Encoding.
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    # Preprocessing для native CatBoost:
    # категориальные признаки не кодируются через OneHot.
    catboost_preprocessor = (
        build_catboost_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

    results = []

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        RESULTS_DIR / "model_results.csv"
    )

    # --------------------
    # Модели
    # --------------------
    for model_name in MODELS:

        logger.info(
            "Model: %s",
            model_name,
        )

        # DNN использует собственный PyTorch CV.
        if model_name == "dnn":
            dnn_results = cross_validate_dnn(
                X=X_train,
                y=y_train_log,
                preprocessor=preprocessor,
            )

            rmse = dnn_results[
                "RMSE"
            ].mean()

            r2 = dnn_results[
                "R2"
            ].mean()

            std = dnn_results[
                "RMSE"
            ].std(ddof=0)

        # Stacking уже содержит preprocessing
        # внутри каждой base-модели.
        elif model_name == "stacking_regressor":
            model = get_model(
                model_name=model_name,
                preprocessor=preprocessor,
            )

            rmse_scores, r2_scores = (
                evaluate_estimator(
                    estimator=model,
                    X=X_train,
                    y=y_train_log,
                )
            )

            rmse = rmse_scores.mean()
            r2 = r2_scores.mean()
            std = rmse_scores.std()

        # Native CatBoost использует свой preprocessing.
        elif model_name == "catboost":
            model = get_model(
                model_name=model_name,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )

            rmse_scores, r2_scores = (
                evaluate_model(
                    model=model,
                    preprocessor=catboost_preprocessor,
                    X=X_train,
                    y=y_train_log,
                )
            )

            rmse = rmse_scores.mean()
            r2 = r2_scores.mean()
            std = rmse_scores.std()

        # Остальные sklearn-модели используют
        # стандартный OneHot preprocessing.
        else:
            model = get_model(
                model_name=model_name,
            )

            rmse_scores, r2_scores = (
                evaluate_model(
                    model=model,
                    preprocessor=preprocessor,
                    X=X_train,
                    y=y_train_log,
                )
            )

            rmse = rmse_scores.mean()
            r2 = r2_scores.mean()
            std = rmse_scores.std()

        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "R2": r2,
            "STD": std,
        })

        # --------------------
        # Сохранение результатов
        # --------------------
        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            by="RMSE",
            ascending=True,
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        logger.info(
            "%s | RMSE: %.4f | R2: %.4f | STD: %.4f",
            model_name,
            rmse,
            r2,
            std,
        )


    # --------------------
    # Финальная модель
    # --------------------
    logger.info(
        "Final model: %s",
        FINAL_MODEL,
    )

    final_model = get_model(
        model_name=FINAL_MODEL,
    )

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", final_model),
    ])

    # Обучаем финальную модель на всех train-данных.
    final_pipeline.fit(
        X_train,
        y_train_log,
    )

    # Получаем прогноз в логарифмической шкале.
    test_predictions_log = final_pipeline.predict(
        X_test
    )

    # Возвращаем прогноз к исходной шкале SalePrice.
    test_predictions = np.expm1(
        test_predictions_log
    )

    submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": test_predictions,
    })

    if submission["SalePrice"].isna().any():
        raise ValueError(
            "Submission содержит пропуски в SalePrice."
        )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    logger.info(
        "Submission saved to: %s",
        SUBMISSION_PATH,
    )

    
    print("\nModel comparison:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: {results_path}"
    )


if __name__ == "__main__":
    main()