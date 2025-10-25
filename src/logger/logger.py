import os 
import logging

# Настройка локального логирования
def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Инициализация логирования
setup_logging("./logs/training.log")
logger = logging.getLogger(__name__)