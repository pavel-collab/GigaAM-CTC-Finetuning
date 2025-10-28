from src.data.dataset import AudioDataset
from src.data.preprocess import normalize_text
from src.data.utils import collate_fn_wrapper
from src.models.utils import import_gigaam_model, get_model_vocab_idx2char, get_model_vocab_char2idx
from src.utils.utils import calculate_wer
#from gigaam.gigaam.preprocess import FeatureExtractor
from src.models.utils import get_gigaam_logprobs, get_texts_idxs
from src.utils.freeze_weights import (
    freeze_model_completely,
    freeze_model_selective,
    freeze_by_components
)

import logging

import os
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn.functional as F
import warnings

# turn off the UserWarnings because lots of them are talking about
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

class GigaAMTrainer:
    """
    Тренер для дообучения моделей GigaAM
    """
   
    def __init__(
        self,
        model_type: str = "ssl",  # или "ctc", "rnnt" для fine-tuning уже обученных моделей
        output_dir: str = "./checkpoints",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        max_epochs: int = 10,
        batch_size: int = 1,
        accumulation_steps: int = 4,
        save_steps: int = 5000,
        eval_steps: int = 1000,
        fp16: bool = True,
        num_workers: int = 4,
    ):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
       
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.num_workers = num_workers
        self.max_epochs = max_epochs

        # Настройка логирования
        self.logging_steps: int = 100
        self.save_total_limit: int = 5

        self.setup_logging()

        # Логгеры
        self.setup_loggers()
       
        # Счетчики
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
       
        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Использование устройства: {self.device}")
        self.logger.info(f"Количесво эпох: {self.max_epochs}")
       
        # Загрузка модели
        self.logger.info(f"Загрузка модели gigaam типа {model_type}...")
        self.model = import_gigaam_model(model_type=self.model_type, device=self.device)

        self.model.to(self.device)

        self.BLANK_IDX = 33
        self.criterion = nn.CTCLoss(blank=self.BLANK_IDX, reduction='mean', zero_infinity=True)

        # замораживаем веса
        # freeze_model_completely(self.model)
        # freeze_model_selective(self.model)
        # freeze_by_components(self.model)

        self.print_frozen_stats()
       
        # Инициализация оптимизатора (будет настроен в train)
        self.optimizer = None
        self.scheduler = None
       
        # Счетчики
        self.global_step = 0
        self.current_epoch = 0

    def setup_logging(self):
        """Настройка логирования"""
        log_file = self.output_dir / "training.log"

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Создание файлового обработчика
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Форматирование
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Добавление обработчика к логгеру
        self.logger.addHandler(file_handler)
   
    def setup_loggers(self):
        """Настройка TensorBoard"""
        self.tb_writer = None
       
        tb_dir = self.output_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        self.tb_writer = SummaryWriter(tb_dir)
        self.logger.info(f"TensorBoard логи: {tb_dir}")

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        # Консоль
        metrics_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        self.logger.info(f"Шаг {step} [{prefix}] - {metrics_str}")
       
        # TensorBoard - добавление каждой метрики отдельно
        if self.tb_writer:
            for metric_name, metric_value in metrics.items():
                # Создаем полное имя метрики: prefix/metric_name
                full_name = f"{prefix}/{metric_name}" if prefix else metric_name
               
                # Добавляем в TensorBoard
                if isinstance(metric_value, (int, float)):
                    self.tb_writer.add_scalar(full_name, metric_value, step)
                    # Опционально: логируем значение в консоль для отладки
                    if step % 100 == 0:  # Каждые 100 шагов
                        self.logger.debug(f"TensorBoard: {full_name} = {metric_value:.4f}(step {step})")
           
            # Принудительно флешируем писатель для гарантированного сохранения
            self.tb_writer.flush()

    def setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )

        steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        if len(self.train_loader) % self.accumulation_steps != 0:
            steps_per_epoch += 1
       
        # Cosine annealing scheduler с warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-4,
            epochs=self.max_epochs,
            steps_per_epoch=steps_per_epoch,
            # total_steps=(len(self.train_loader) // self.accumulation_steps) * self.max_epochs,
            pct_start=0.1
        )

    def train_step(self, batch) -> float:
        wav_batch, wav_lengths, targets, target_lengths, texts = batch

        wav_batch = wav_batch.to(self.device)
        wav_lengths = wav_lengths.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        # Прямой проход
        features, feat_lengths = self.model.preprocessor(wav_batch, wav_lengths)
        encoded, encoded_len = self.model.encoder(features, feat_lengths)
        logprobs = self.model.head(encoded)

        # Log softmax для CTC
        log_probs = torch.nn.functional.log_softmax(logprobs, dim=-1)
        
        # Перестановка для CTC: (T, B, C)
        log_probs = log_probs.transpose(0, 1)  # (T, B, C)
        
        # Убедитесь, что длины корректны
        input_lengths = encoded_len.clamp(min=1)
        target_lengths = target_lengths.clamp(min=1)
      
        # CTC Loss
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
       
        (loss / self.accumulation_steps).backward()

        return loss.item()
    
    def train(self):
        # Создание датасетов
        self.logger.info("Создание датасетов...")

        char2idx = get_model_vocab_char2idx(self.model)

        train_dataset = AudioDataset(dataset_part="train", normalize_fn=normalize_text)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn_wrapper(x, char2idx),
            # pin_memory=True if torch.cuda.is_available() else False
        )
       
        val_dataset = AudioDataset(dataset_part="validation", normalize_fn=normalize_text)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn_wrapper(x, char2idx),
            # pin_memory=True if torch.cuda.is_available() else False
        )
       
        # Настройка оптимизатора
        self.setup_optimizer()

        self.logger.info("Начало обучения...")
        self.logger.info(f"Батч размер: {self.batch_size}")
        self.logger.info(f"Градиентная аккумуляция: {self.accumulation_steps}")
        self.logger.info(f"Эффективный батч размер: {self.batch_size * self.accumulation_steps}")
       
        running_loss = 0.0
        self.optimizer.zero_grad()

        self.best_wer = float('inf')
       
        while self.current_epoch < self.max_epochs and self.global_step < self.max_steps:
            self.model.train()
            self.current_epoch += 1
            self.optimizer.zero_grad()
           
            pbar = tqdm(self.train_loader, desc=f"Эпоха {self.current_epoch}")
            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                running_loss += loss
               
                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                   
                    self.optimizer.zero_grad()
                    self.global_step += 1
                   
                    # Логирование
                    # Логирование
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = running_loss / self.accumulation_steps
                        lr = self.optimizer.param_groups[0]['lr']
                    
                        metrics = {
                            'loss': avg_loss,
                            'lr': lr,
                        }
                        self.log_metrics(metrics, self.global_step, "train")
                        pbar.set_postfix(metrics)
                        running_loss = 0.0
                
                    # Валидация
                    if self.global_step % self.eval_steps == 0:
                        self.logger.info(f"Валидация на шаге {self.global_step}...")
                        self.validate(self.val_loader)
                
                    # Сохранение чекпоинта
                    #! temporarry comment this line, cause of no so much free space on my test device
                    # if self.global_step % self.save_steps == 0:
                    #     self.save_checkpoint()

                    running_loss = 0.0

            self.validate(self.val_loader)
           
            if self.global_step >= self.max_steps:
                break
       
        self.logger.info("Сохранение финального чекпоинта...")
        self.save_checkpoint(name="final_model")
        self.logger.info("Обучение завершено!")

    def validate(self, val_loader: DataLoader) -> float:
        """
        Валидация модели
       
        Args:
            val_loader: DataLoader для валидационных данных
           
        Returns:
            средний loss на валидации
        """
        self.model.eval()

        idx2char = get_model_vocab_idx2char(self.model)
        wer, refs, hyps = calculate_wer(self.model, val_loader, self.device, idx2char, self.BLANK_IDX)

        if wer < self.best_wer:
            self.best_wer = wer
            checkpoint_path = os.path.join(self.output_dir, f'best_model_wer_{wer*100:.2f}.pt')
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'wer': wer,
                # 'loss': avg_loss,
                'vocab': self.model.decoding.tokenizer.vocab,
                # 'char_to_idx': char_to_idx,
                'blank_idx': self.BLANK_IDX,  # Сохраняем blank index
            }, checkpoint_path)

        self.log_metrics({"wer": wer}, self.global_step, "validation")

        self.model.train()
        return wer
    
    def save_checkpoint(self, name: str = None):
        """
        Сохранение чекпоинта модели
       
        Args:
            name: имя чекпоинта (если None, используется global_step)
        """
        if name is None:
            name = f"checkpoint_step_{self.global_step}"
       
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
       
        # Сохранение state dict модели
        '''
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
        }, checkpoint_path / "pytorch_model.bin")
        '''
        torch.save(self.model.state_dict(), checkpoint_path / 'gigaam_weights.pth')
       
        # Сохранение конфигурации
        config = {
            'model_name': self.model_type,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'learning_rate': self.learning_rate,
        }
       
        with open(checkpoint_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
       
        self.logger.info(f"Чекпоинт сохранен: {checkpoint_path}")

    def print_frozen_stats(self):
        total_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
                self.logger.info(f"❌ Заморожен: {name}")
            else:
                self.logger.info(f"✅ Разморожен: {name}")
        
        self.logger.info(f"\n📊 Статистика:")
        self.logger.info(f"Всего параметров: {total_params:,}")
        self.logger.info(f"Заморожено: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        self.logger.info(f"Обучается: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params*100:.1f}%)")
