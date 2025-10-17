from src.data.dataset import AudioDataset
from src.data.utils import collate_fn

import torch
import gigaam
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import json

import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GigaAMTrainer:
    """
    Тренер для дообучения моделей GigaAM
    """
   
    def __init__(
        self,
        model_name: str = "ssl",  # или "ctc", "rnnt" для fine-tuning уже обученных моделей
        output_dir: str = "./checkpoints",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        batch_size: int = 8,
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
        self.model_name = model_name
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
       
        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Использование устройства: {self.device}")
       
        # Загрузка модели
        logger.info(f"Загрузка модели {model_name}...")
        #TODO: посмотреть как правильно подгружать модель с предобученными весами, переписать эту часть
        #TODO: более того, нужно проверить, что модель будет не в eval mode, а в train mode, чтобы можно было обновлять веса
        self.model = gigaam.load_model(model_name)
        self.model = self.model.to(self.device)
       
        # Настройка mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None
        self.use_amp = fp16 and torch.cuda.is_available()
       
        # Инициализация оптимизатора (будет настроен в train)
        self.optimizer = None
        self.scheduler = None
       
        # Счетчики
        self.global_step = 0
        self.current_epoch = 0

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

    def train_step(self, batch: Dict) -> float:
        """
        Один шаг обучения
       
        Args:
            batch: батч данных
           
        Returns:
            значение loss
        """
        #TODO: будут использоваться при вычислении функции потерь
        audios = batch['audios'].to(self.device)
        audio_lengths = batch['audio_lengths'].to(self.device)
        texts = batch['texts']

         # Forward pass с mixed precision
        #TODO: здесь нужно реализовать функцию потерь, параметр self.model_name будет влиять на тип вычисляемого лосса
        #TODO: вообще, по хорошему self.model_name лучше переименовать в self.model_type, а еще по хорошему, лучше вообще удалить этот параметр, т к явно исходная модель имеет определенный тип
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Здесь должна быть логика вычисления loss
                # Для SSL модели - это masked prediction loss
                # Для CTC/RNNT - это соответствующие loss функции
               
                # Пример для CTC (нужно адаптировать под конкретную задачу)
                # outputs = self.model(audios, audio_lengths)
                # loss = self.compute_loss(outputs, texts, audio_lengths)
               
                # Заглушка - нужно реализовать специфичную логику
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        else:
            # outputs = self.model(audios, audio_lengths)
            # loss = self.compute_loss(outputs, texts, audio_lengths)
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
       
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss / self.accumulation_steps).backward()
        else:
            (loss / self.accumulation_steps).backward()
       
        return loss.item()
    
    def train(
        self,
        train_manifest: str,
        val_manifest: str = None,
    ):
        """
        Основной цикл обучения
       
        Args:
            train_manifest: путь к манифесту тренировочных данных
            val_manifest: путь к манифесту валидационных данных
        """
        # Создание датасетов
        logger.info("Создание датасетов...")
        #TODO: здесь не указывает предобработчик, а он нужен
        #TODO: загрузить преобработчик из GigaAM, вроде как там за это отвечает FeatureExtractor
        #TODO: как варик -- загрузить GigaAM как git submodule
        train_dataset = AudioDataset(train_manifest)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
       
        val_loader = None
        if val_manifest:
            val_dataset = AudioDataset(val_manifest)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
       
        # Настройка оптимизатора
        self.setup_optimizer()
       
        # Режим тренировки
        self.model.train()

        logger.info("Начало обучения...")
        logger.info(f"Батч размер: {self.batch_size}")
        logger.info(f"Градиентная аккумуляция: {self.accumulation_steps}")
        logger.info(f"Эффективный батч размер: {self.batch_size * self.accumulation_steps}")
       
        running_loss = 0.0
        self.optimizer.zero_grad()
       
        while self.global_step < self.max_steps:
            self.current_epoch += 1
           
            pbar = tqdm(train_loader, desc=f"Эпоха {self.current_epoch}")
            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                running_loss += loss
               
                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
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
                    avg_loss = running_loss / self.accumulation_steps
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': self.global_step
                    })
                    running_loss = 0.0
                   
                   #TODO: сделать отдельную механику, которая будет рисовать графики
                    # Валидация
                    if val_loader and self.global_step % self.eval_steps == 0:
                        val_loss = self.validate(val_loader)
                        logger.info(f"Шаг {self.global_step}: Val Loss = {val_loss:.4f}")
                        self.model.train()
                   
                    # Сохранение чекпоинта
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint()
                   
                    if self.global_step >= self.max_steps:
                        break
           
            if self.global_step >= self.max_steps:
                break
       
        logger.info("Обучение завершено!")
        self.save_checkpoint(name="final_model")

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
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация"):
                audios = batch['audios'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                texts = batch['texts']
               
               #TODO: здесь тоже нужно будет написать вычисление функции потерь
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # loss = self.compute_loss(...)
                        loss = torch.tensor(0.0, device=self.device)  # Заглушка
                else:
                    # loss = self.compute_loss(...)
                    loss = torch.tensor(0.0, device=self.device)  # Заглушка
               
                total_loss += loss.item()
                num_batches += 1
       
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
        }, checkpoint_path / "pytorch_model.bin")
       
        # Сохранение конфигурации
        config = {
            'model_name': self.model_name,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'learning_rate': self.learning_rate,
        }
       
        with open(checkpoint_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
       
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка чекпоинта
       
        Args:
            checkpoint_path: путь к чекпоинту
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
       
        self.model.load_state_dict(checkpoint['model_state_dict'])
       
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
       
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
       
        logger.info(f"Чекпоинт загружен из {checkpoint_path}")
        logger.info(f"Продолжение с шага {self.global_step}, эпоха {self.current_epoch}")