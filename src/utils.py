import random
import logging

import numpy as np
import torch

from config import (
    LOGGING_CONFIG,
    LOGS_DIR,
    RANDOM_STATE,
)


def set_seed(seed=RANDOM_STATE):
    """Фиксирует генераторы случайных чисел для воспроизводимости результатов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed=RANDOM_STATE):
    """Создает PyTorch Generator с фиксированным seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def setup_logger():
    """
    Создает logger для вывода сообщений в консоль и файл.

    Настройки logger задаются в config.py.
    """

    logger = logging.getLogger(
        LOGGING_CONFIG["name"]
    )

    logger.setLevel(
        LOGGING_CONFIG["level"]
    )

    # Не передаем сообщения родительскому logger,
    # чтобы избежать дублирования вывода.
    logger.propagate = False

    # Не добавляем обработчики повторно,
    # если logger уже был настроен.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        LOGGING_CONFIG["format"],
        datefmt=LOGGING_CONFIG["date_format"],
    )

    # Вывод сообщений в терминал.
    if LOGGING_CONFIG["console"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(
            console_handler
        )

    # Запись сообщений в файл.
    if LOGGING_CONFIG["file"]:
        LOGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            LOGS_DIR / LOGGING_CONFIG["file_name"],
            mode=LOGGING_CONFIG["file_mode"],
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(
            file_handler
        )

    return logger