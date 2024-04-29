import logging
import os
from typing import Tuple

MAIN_LOGGER_NAME = "MAIN"


def create_logger(name, level=logging.INFO):
    log_dir = os.path.join(os.getcwd(), "logs/")
    file_dir = os.path.join(log_dir, name + ".log")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    fh = logging.FileHandler(file_dir, "w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.addHandler(sh)
    logger.addHandler(fh)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    return logger


def time_to_largest_unit(time_ts: float) -> Tuple[float, str]:
    """
    Takes seconds and converts it to largest possible unit
    e.g. 3601 seconds -> 1.000277 hours
    """
    units = ["seconds", "minutes", "hours", "days"]
    unit_size = [60, 60, 24, 1]
    for unit, size in zip(units, unit_size):
        if time_ts < size:
            return time_ts, unit
        time_ts /= size
    return time_ts, "years"
