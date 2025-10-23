from utils import fix_torch_seed
from src.models.utils import get_model_vocab

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import os

from models.ctc_model import CTCLightningModule
from data.datasets import CTCDataModule

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

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    fix_torch_seed()

    # Инициализация логирования
    setup_logging(cfg.logging.local_log_file)
    logger = logging.getLogger(__name__)
    
    # Логирование конфигурации
    logger.info("Starting training with config:")
    logger.info(cfg)

    # Инициализация модели
    model = CTCLightningModule(cfg)  # vocab_size должен быть из конфига или датасета

    # Получаем словарь модели
    model_vocab = get_model_vocab(model.model)

    # Инициализация данных
    data_module = CTCDataModule(cfg, model_vocab=model_vocab, logger=logger)

    # Логгер для TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.tensorboard_dir,
        name="ctc_model"
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ctc-best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    # Запуск обучения
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()