import logging
import os
from datetime import datetime
import yaml

# Get absolute project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name, log_dir=None, level=logging.INFO):
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_file = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
