from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Models to evaluate
MODELS = [
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
    "knn",
    "decision_tree",
    "random_forest",
    "catboost",
    "lightgbm",
    "xgboost",
    "catboost_onehot",
    "voting_regressor",
    "boosting_voting_regressor",
    "stacking_regressor",
    "dnn",
]

# Final model
FINAL_MODEL = "catboost_onehot"

# Reproducibility
RANDOM_STATE = 42

# Cross-validation
N_SPLITS = 5
CV_SHUFFLE = True
N_JOBS = 1

# Metrics
RMSE_SCORING = "neg_root_mean_squared_error"
R2_SCORING = "r2"

# Ridge Regression
RIDGE_PARAMS = {
    "alpha": 21.822942025785657,
}

# Lasso Regression
LASSO_PARAMS = {
    "alpha": 0.00042079886696066364,
}
LASSO_MAX_ITER = 10000
VOTING_LASSO_MAX_ITER = 50000

# ElasticNet Regression
ELASTICNET_PARAMS = {
    "alpha": 0.0004425091452638345,
    "l1_ratio": 0.9940130564424344,
}
ELASTICNET_MAX_ITER = 50000

# K-Nearest Neighbors
KNN_PARAMS = {
    "n_neighbors": 8,
    "weights": "distance",
    "metric": "manhattan",
}

# Decision Tree
DECISION_TREE_PARAMS = {
    "max_depth": 15,
    "min_samples_split": 18,
    "min_samples_leaf": 5,
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 447,
    "max_depth": 18,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
}

# CatBoost
CATBOOST_PARAMS = {
    "iterations": 851,
    "depth": 6,
    "learning_rate": 0.09021041082589737,
    "l2_leaf_reg": 2.3725494927763897,
    "verbose": False,
}

# LightGBM
LIGHTGBM_PARAMS = {
    "n_estimators": 257,
    "max_depth": 6,
    "learning_rate": 0.03326586456503854,
    "num_leaves": 55,
    "min_child_samples": 21,
    "verbosity": -1,
}

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 414,
    "max_depth": 4,
    "learning_rate": 0.05748924681991978,
    "subsample": 0.7962072844310213,
    "colsample_bytree": 0.5232252063599989,
    "objective": "reg:squarederror",
}

# CatBoost with OneHot Encoding
CATBOOST_ONEHOT_PARAMS = {
    "iterations": 818,
    "depth": 7,
    "learning_rate": 0.034846684639687375,
    "l2_leaf_reg": 5.786180679994022,
    "verbose": False,
}

# Deep Neural Network
DNN_PARAMS = {
    "hidden_size": 128,
    "activation": "elu",
    "batch_size": 16,
    "epochs": 200,
    "learning_rate": 0.001,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "loss": "mse",
    "scheduler": "cosine_annealing",
    "eta_min": 1e-5,
    "train_shuffle": True,
    "log_every": 10,
}

# Logging
LOGGING_CONFIG = {
    "name": "house_prices",
    "level": "INFO",
    "console": True,
    "file": True,
    "file_name": "training.log",
    "file_mode": "w",
    "format": "%(asctime)s | %(levelname)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}