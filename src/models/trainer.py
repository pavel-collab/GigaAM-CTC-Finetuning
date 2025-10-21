from src.data.dataset import AudioDataset
from src.data.utils import collate_fn
from src.models.utils import import_gigaam_model
#from gigaam.gigaam.preprocess import FeatureExtractor
from src.models.utils import get_gigaam_logprobs, get_texts_idxs, get_model_vocab
import logging
import sys

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
        """
        Args:
            model_name: имя модели для загрузки ("ssl", "ctc", "rnnt", "v1_ssl", "v2_ctc" и т.д.)
            output_dir: директория для сохранения чекпоинтов
            learning_rate: learning rate
            warmup_steps: количество шагов warmup
            max_steps: максимальное количество шагов обучения
            batch_size: размер батча
            accumulation_steps: шаги градиентной аккумуляции
            save_steps: частота сохранения чекпоинтов
            eval_steps: частота валидации
            fp16: использование mixed precision
            num_workers: количество воркеров для DataLoader
        """
        #? can we exchange all of this assignments to save_hyperparameters?
        #? do we need to use a pytorch lightning wrpaper for it?
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
        
        #! Temporary disable mixed procision       
        # # Настройка mixed precision
        # self.scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None
        # self.use_amp = fp16 and torch.cuda.is_available()
       
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
        """
        Логирование метрик в консоль, TensorBoard и W&B
       
        Args:
            metrics: словарь с метриками {"metric_name": value, ...}
            step: номер шага/эпохи
            prefix: префикс для организации метрик (например "train", "val")
        """
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
        """Настройка оптимизатора и планировщика"""
        # Оптимизатор AdamW с weight decay
        self.optimizer = AdamW(
            self.model.parameters(), #! указываем оптимизатору параметры модели, так мы сможем обновлять ее веса
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )
       
        # Cosine annealing scheduler с warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=1e-7
        )

    def train_step(self, batch) -> float:
        """
        Один шаг обучения
       
        Args:
            batch: батч данных
           
        Returns:
            значение loss
        """
        audios, audio_lengths, texts = batch
        
        #audios = batch['audio'].to(self.device)
        #audio_lengths = batch['num_samples'].to(self.device)
        #texts = batch['transcription']

        #! temporary disable mixed precision
        # # Forward pass с mixed precision
        # if self.use_amp:
        #     with torch.cuda.amp.autocast():
        #         # Здесь должна быть логика вычисления loss
        #         # Для SSL модели - это masked prediction loss
        #         # Для CTC/RNNT - это соответствующие loss функции
               
        #         # Пример для CTC (нужно адаптировать под конкретную задачу)
        #         # outputs = self.model(audios, audio_lengths)
        #         # loss = self.compute_loss(outputs, texts, audio_lengths)
               
        #         # Заглушка - нужно реализовать специфичную логику
        #         loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        # else:
        #     # outputs = self.model(audios, audio_lengths)
        #     # loss = self.compute_loss(outputs, texts, audio_lengths)
        #     loss = torch.tensor(0.0, requires_grad=True, device=self.device)
      
        model_vocab = get_model_vocab(self.model)

        texts = get_texts_idxs(texts, model_vocab)

        transcript_lengths=(len(sample) for sample in texts)
        #! forward pass находится внутри этой функции compute_ctc_loss. Согласен, с точки зрения архитектуры -- не очень, но пока работаем так
        loss = self.compute_ctc_loss(
                    audios, 
                    audio_lengths,
                    texts,
                    transcript_lengths=tuple(transcript_lengths)
                )

        #! temporary disable mixed precision
        # # Backward pass
        # if self.use_amp:
        #     self.scaler.scale(loss / self.accumulation_steps).backward()
        # else:
        #     (loss / self.accumulation_steps).backward()
       
        (loss / self.accumulation_steps).backward()

        return loss.item()
    
    def train(self):
        """
        Основной цикл обучения
       
        Args:
            train_manifest: путь к манифесту тренировочных данных
            val_manifest: путь к манифесту валидационных данных
        """
        # Создание датасетов
        self.logger.info("Создание датасетов...")

        #preprocessor = FeatureExtractor(sample_rate=16000, features=64)
        preprocessor = None

        train_dataset = AudioDataset(preprocessor=preprocessor, dataset_part="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            # pin_memory=True if torch.cuda.is_available() else False
        )
       
        val_dataset = AudioDataset(preprocessor=preprocessor, dataset_part="validation")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            # pin_memory=True if torch.cuda.is_available() else False
        )
       
        # Настройка оптимизатора
        self.setup_optimizer()
       
        # Режим тренировки
        self.model.train()

        self.logger.info("Начало обучения...")
        self.logger.info(f"Батч размер: {self.batch_size}")
        self.logger.info(f"Градиентная аккумуляция: {self.accumulation_steps}")
        self.logger.info(f"Эффективный батч размер: {self.batch_size * self.accumulation_steps}")
       
        running_loss = 0.0
        self.optimizer.zero_grad()
       
        while self.current_epoch < self.max_epochs and self.global_step < self.max_steps:
            self.current_epoch += 1
           
            pbar = tqdm(train_loader, desc=f"Эпоха {self.current_epoch}")
            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                running_loss += loss
               
                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    #! mixed precision is temporary disabled
                    # if self.use_amp:
                    #     self.scaler.unscale_(self.optimizer)
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    #     self.scaler.step(self.optimizer)
                    #     self.scaler.update()
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    #     self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                   
                    # Warmup
                    if self.global_step < self.warmup_steps:
                        lr = self.learning_rate * (self.global_step + 1) / self.warmup_steps
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
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
                        # val_loss = self.validate(val_loader)
                        # self.log_metrics({'loss': val_loss}, self.global_step, "val")
                
                    # Сохранение чекпоинта
                    #! temporarry comment this line, cause of no so much free space on my test device
                    # if self.global_step % self.save_steps == 0:
                    #     self.save_checkpoint()

                    running_loss = 0.0

            self.validate(val_loader)
           
            if self.global_step >= self.max_steps:
                break
       
        self.logger.info("Сохранение финального чекпоинта...")
        self.save_checkpoint(name="final_model")
        self.logger.info("Обучение завершено!")

    #! На данный момент с ctc_loss есть определенные проблемы
    #! Там нужно подавать на вход индексы для токенов транскрипции в словаре модели,
    #! Но там в словаре только обычные русские буквы, нет символов типо : и др
    #! Как вариант, можно убирать все символы нерусского алфавита из строчки транскрипции,
    #! Или считать функцию потерь MSE через логиты    
    def compute_ctc_loss(self, wav_batch, wav_lengths, transcripts, transcript_lengths):
        # Получаем логиты от модели
        logprobs, encoded_len = get_gigaam_logprobs(self.model, wav_batch.to(self.device), wav_lengths.to(self.device))

        '''
        print(f"[DEBUG] transcripts shape {transcripts.shape}")
        print(f"[DEBUG] transcripts_lengths {transcript_lengths}")
        print(f"[DEBUG] logprobs shape {logprobs.shape}")
        print(f"[DEBUG] logprobs len {encoded_len}")
        '''

        # Проверяем и выравниваем длины
        encoded_len = tuple(encoded_len.to('cpu').numpy())
        
        # Убеждаемся, что encoded_len не превышает длину logprobs по времени
        T = logprobs.size(1)  # временная размерность после transpose
        encoded_len = tuple(min(el, T) for el in encoded_len)
        
        #print(f"[DEBUG] new logprobs len {encoded_len}")

        # CTCLoss требует логиты в формате (T, N, C)
        logprobs = logprobs.transpose(0, 1)  # Теперь форма (T, N, C)

        BLANK_IDX = 33
        ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True).to(self.device)

        # Вычисляем потерю
        loss = ctc_loss(
            logprobs,           # (T, N, C)
            transcripts,        # (N, S) -> целочисленные индексы
            encoded_len,        # (N,) -> длины выходных последовательностей
            transcript_lengths  # (N,) -> длины целевых последовательностей
        )

        return loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Валидация модели
       
        Args:
            val_loader: DataLoader для валидационных данных
           
        Returns:
            средний loss на валидации
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация"):
                audios, audio_lengths, texts = batch

                '''
                audios = batch['audio'].to(self.device)
                audio_lengths = batch['num_samples'].to(self.device)
                texts = batch['transcription']
               '''

                #! mixed precision is temporary disabled
                # if self.use_amp:
                #     with torch.cuda.amp.autocast():
                #         # loss = self.compute_loss(...)
                #         loss = torch.tensor(0.0, device=self.device)  # Заглушка
                # else:
                #     # loss = self.compute_loss(...)
                #     loss = torch.tensor(0.0, device=self.device)  # Заглушка
               
                model_vocab = get_model_vocab(self.model)
                texts = get_texts_idxs(texts, model_vocab)

                transcript_lengths=(len(sample) for sample in texts)
                loss = self.compute_ctc_loss(
                            audios, 
                            audio_lengths,
                            texts,
                            transcript_lengths=tuple(transcript_lengths)
                        )

                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_metrics['loss'] = avg_val_loss

        # Логирование метрик валидации в TensorBoard
        self.log_metrics(val_metrics, self.global_step, "val")
       
        self.model.train()
       
        return total_loss / num_batches if num_batches > 0 else 0.0
    
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
