import ast
import logging
import os
import random

import numpy as np
import torch


def get_logger(filename, verbosity=1, name=None):
    """Create a logger that writes to both a text file and the console."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s]%(message)s")

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(level_dict[verbosity])

    file_handler = logging.FileHandler(filename + ".txt", "a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def setup_seed(seed):
    """Set random seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def normalize_device(device_value):
    device_str = str(device_value)
    if device_str.isdigit():
        return f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    return device_str


def parse_search_values(raw_value):
    parsed = ast.literal_eval(raw_value)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


class EarlyStopping:
    """Stop training when the validation metric no longer improves."""

    def __init__(
        self,
        logger,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logger = logger

    def __call__(self, val_loss, model, epoch):
        del epoch
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            message = (
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            self.trace_func(message)
            self.logger.info(message)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            message = (
                f"Validation score improved "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
            self.trace_func(message)
            self.logger.info(message)

        torch.save(model.state_dict(), os.path.join(self.path, "best_val_epoch.pt"))
        self.val_loss_min = val_loss
