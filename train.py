from src.models.trainer import GigaAMTrainer

import logging

#TODO: проверить настройку логгера, в случае необходимости настроить его по своему
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Пример использования
    """
    # Конфигурация обучения
    trainer = GigaAMTrainer(
        model_type="ctc",  # или "ctc", "rnnt" для fine-tuning
        output_dir="./gigaam_checkpoints",
        learning_rate=1e-4,
        warmup_steps=1000,
        max_steps=50000,
        batch_size=1,
        accumulation_steps=4,
        save_steps=5000,
        eval_steps=1000,
        fp16=True,
        num_workers=4,
    )
   
    # Запуск обучения
    trainer.train()

if __name__ == "__main__":
    main()
