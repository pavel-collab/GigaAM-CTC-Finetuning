from utils import fix_torch_seed
from src.models.utils import get_model_vocab

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.model import CTCLightningModule
from src.data.dataset import CTCDataModule
from src.logger.logger import logger

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    fix_torch_seed()
    
    # Логирование конфигурации
    logger.info("Starting training with config:")
    logger.info(cfg)

    # Инициализация модели
    model = CTCLightningModule(cfg)  # vocab_size должен быть из конфига или датасета

    # Получаем словарь модели
    model_vocab = get_model_vocab(model.model)

    # Инициализация данных
    data_module = CTCDataModule(cfg, model_vocab=model_vocab)

    # Логгер для TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.tensorboard_dir,
        name="ctc_model"
    )

    # Callbacks
    '''
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ctc-best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    '''

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=tb_logger,
        #callbacks=[checkpoint_callback]
    )

    # Запуск обучения
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training completed!")
    
    # Сохраняем вручную после обучения
    torch.save(model.state_dict(), "./saved_models/final_model_weights.pth")
    #trainer.save_checkpoint("./saved_models/final_checkpoint.ckpt")

if __name__ == "__main__":
    main()
