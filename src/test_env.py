from src.utils import load_config, setup_logger

config = load_config()
logger = setup_logger("PHASE0_TEST")

logger.info("Environment setup successful!")
logger.info(f"Project name: {config['project']['name']}")
logger.info(f"Model backbone: {config['model']['backbone']}")
