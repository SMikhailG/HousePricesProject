import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

from src.train import build_cv
from src.utils import make_generator, set_seed, setup_logger

from config import DNN_PARAMS, RANDOM_STATE


logger = setup_logger()


def get_activation(name):
    """Создает функцию активации по имени из config."""

    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }

    if name not in activations:
        raise ValueError(f"Неизвестная функция активации: {name}")

    return activations[name]()


def get_loss(name):
    """Создает функцию потерь по имени из config."""

    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
    }

    if name not in losses:
        raise ValueError(f"Неизвестная функция потерь: {name}")

    return losses[name]()


def get_optimizer(name, model, learning_rate, weight_decay):
    """Создает оптимизатор по настройкам из config."""

    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }

    if name not in optimizers:
        raise ValueError(f"Неизвестный оптимизатор: {name}")

    return optimizers[name](
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def get_scheduler(name, optimizer, t_max, eta_min):
    """Создает scheduler по настройкам из config."""

    if name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    raise ValueError(f"Неизвестный scheduler: {name}")


class HousePriceDNN(nn.Module):
    """Нейронная сеть для прогнозирования логарифма цены дома."""

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                DNN_PARAMS["hidden_size"],
            ),
            get_activation(DNN_PARAMS["activation"]),
            nn.Linear(
                DNN_PARAMS["hidden_size"],
                1,
            ),
        )

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    epochs,
    scheduler=None,
    log_every=10,
):
    """
    Обучает DNN, сохраняет историю обучения
    и восстанавливает лучшие веса по Validation Loss.
    """

    # Возвращаем DataLoader к исходному состоянию generator.
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(
            train_loader.generator.initial_seed()
        )

    train_loss = []
    val_loss = []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):

        # --------------------
        # Обучение
        # --------------------
        model.train()

        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_fn(
                predictions,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            running_train_loss += (
                loss.item() * X_batch.size(0)
            )

        epoch_train_loss = (
            running_train_loss
            / len(train_loader.dataset)
        )

        train_loss.append(
            epoch_train_loss
        )

        # --------------------
        # Валидация
        # --------------------
        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                predictions = model(X_batch)

                loss = loss_fn(
                    predictions,
                    y_batch,
                )

                running_val_loss += (
                    loss.item() * X_batch.size(0)
                )

        epoch_val_loss = (
            running_val_loss
            / len(valid_loader.dataset)
        )

        val_loss.append(
            epoch_val_loss
        )

        # Сохраняем веса с минимальным Validation Loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        
        # Периодически записываем ход обучения в лог.
        if (epoch + 1) % log_every == 0:
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | "
                "Val Loss: %.4f | LR: %.6f",
                epoch + 1,
                epochs,
                epoch_train_loss,
                epoch_val_loss,
                optimizer.param_groups[0]["lr"],
            )


        # Изменяем learning rate после завершения эпохи.
        if scheduler is not None:
            scheduler.step()

    # Используем лучшую версию модели,
    # а не веса последней эпохи.
    model.load_state_dict(
        best_model_state
    )

    return train_loss, val_loss


def cross_validate_dnn(
    X,
    y,
    preprocessor,
):
    """
    Проводит K-Fold cross-validation DNN.

    На каждом fold preprocessing и нейронная сеть
    обучаются заново только на train-части данных.
    """

    cv = build_cv()
    cv_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X),
        start=1,
    ):
        logger.info(
            "DNN Fold %d/%d",
            fold,
            cv.get_n_splits(),
        )

        # Для каждого fold используем свой seed.
        fold_seed = RANDOM_STATE + fold

        # --------------------
        # Разделение данных
        # --------------------
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        # --------------------
        # Preprocessing
        # --------------------
        fold_preprocessor = clone(
            preprocessor
        )

        X_fold_train_processed = (
            fold_preprocessor.fit_transform(
                X_fold_train
            )
        )

        X_fold_valid_processed = (
            fold_preprocessor.transform(
                X_fold_valid
            )
        )

        # --------------------
        # Tensor
        # --------------------
        X_fold_train_tensor = torch.tensor(
            X_fold_train_processed,
            dtype=torch.float32,
        )

        X_fold_valid_tensor = torch.tensor(
            X_fold_valid_processed,
            dtype=torch.float32,
        )

        y_fold_train_tensor = torch.tensor(
            y_fold_train.values,
            dtype=torch.float32,
        ).reshape(-1, 1)

        y_fold_valid_tensor = torch.tensor(
            y_fold_valid.values,
            dtype=torch.float32,
        ).reshape(-1, 1)

        # --------------------
        # DataLoader
        # --------------------
        fold_train_dataset = TensorDataset(
            X_fold_train_tensor,
            y_fold_train_tensor,
        )

        fold_valid_dataset = TensorDataset(
            X_fold_valid_tensor,
            y_fold_valid_tensor,
        )

        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=DNN_PARAMS["batch_size"],
            shuffle=DNN_PARAMS["train_shuffle"],
            generator=make_generator(
                fold_seed
            ),
        )

        fold_valid_loader = DataLoader(
            fold_valid_dataset,
            batch_size=DNN_PARAMS["batch_size"],
            shuffle=False,
        )

        # --------------------
        # Модель
        # --------------------

        # Количество OneHot-признаков может различаться
        # между folds, поэтому input_size определяем здесь.
        input_size = (
            X_fold_train_tensor.shape[1]
        )

        set_seed(
            fold_seed
        )

        fold_model = HousePriceDNN(
            input_size=input_size,
        )

        # --------------------
        # Обучение
        # --------------------
        fold_loss_fn = get_loss(
            DNN_PARAMS["loss"]
        )

        fold_optimizer = get_optimizer(
            name=DNN_PARAMS["optimizer"],
            model=fold_model,
            learning_rate=DNN_PARAMS["learning_rate"],
            weight_decay=DNN_PARAMS["weight_decay"],
        )

        fold_scheduler = get_scheduler(
            name=DNN_PARAMS["scheduler"],
            optimizer=fold_optimizer,
            t_max=DNN_PARAMS["epochs"],
            eta_min=DNN_PARAMS["eta_min"],
        )

        train_model(
            model=fold_model,
            train_loader=fold_train_loader,
            valid_loader=fold_valid_loader,
            loss_fn=fold_loss_fn,
            optimizer=fold_optimizer,
            epochs=DNN_PARAMS["epochs"],
            scheduler=fold_scheduler,
            log_every=DNN_PARAMS["log_every"],
        )

        # --------------------
        # Оценка модели
        # --------------------
        fold_model.eval()

        predictions = []

        with torch.no_grad():
            for X_batch, _ in fold_valid_loader:
                batch_predictions = (
                    fold_model(X_batch)
                )

                predictions.append(
                    batch_predictions
                )

        predictions = torch.cat(
            predictions
        ).numpy().flatten()

        y_true = (
            y_fold_valid_tensor
            .numpy()
            .flatten()
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        )

        fold_r2 = r2_score(
            y_true,
            predictions,
        )

        cv_results.append({
            "Fold": fold,
            "RMSE": fold_rmse,
            "R2": fold_r2,
        })

        logger.info(
            "Fold %d | RMSE: %.4f | R2: %.4f",
            fold,
            fold_rmse,
            fold_r2,
        )

    results_df = pd.DataFrame(
        cv_results
    )

    mean_rmse = results_df["RMSE"].mean()
    mean_r2 = results_df["R2"].mean()

    rmse_std = results_df[
        "RMSE"
    ].std(ddof=0)

    logger.info(
        "DNN CV | Mean RMSE: %.4f | "
        "Mean R2: %.4f | STD: %.4f",
        mean_rmse,
        mean_r2,
        rmse_std,
    )

    return results_df