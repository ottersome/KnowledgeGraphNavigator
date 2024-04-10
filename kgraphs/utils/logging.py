import logging
import os


def create_logger(name, level=logging.INFO):
    log_dir = os.path.join(os.getcwd(), "logs/")
    file_dir = os.path.join(log_dir, name + ".log")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    fh = logging.FileHandler(file_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.addHandler(sh)
    logger.addHandler(fh)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    return logger
