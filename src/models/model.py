from src.models.utils import import_gigaam_model, get_model_vocab, get_gigaam_logprobs
from src.logger.logger import logger

import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

class CTCLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Сохраняем гиперпараметры
        self.save_hyperparameters()

        self.model = import_gigaam_model(model_type=self.config.model.name)

        self.scheduler_type = "cosine"

        BLANK_IDX = 33
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)

        # разморозить все слои для начала
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

        '''
        total_params = 0
        frozen_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
                logger.info(f"❌ Заморожен: {name}")
            else:
                logger.info(f"✅ Разморожен: {name}")
        
        logger.info(f"Статистика:")
        logger.info(f"Всего параметров: {total_params:,}")
        logger.info(f"Заморожено: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logger.info(f"Обучается: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params*100:.1f}%)")
        '''

    '''
    def training_step(self, batch, batch_idx):
        audios, audio_lengths, texts = batch

        model_vocab = get_model_vocab(self.model)

        transcript_lengths=(len(sample) for sample in texts)

        logprobs, encoded_len = get_gigaam_logprobs(self.model, audios, audio_lengths)

        loss =  self._compute_ctc_loss(
                    logprobs,
                    encoded_len,
                    texts,
                    transcript_lengths=tuple(transcript_lengths)
                )
        
        return loss.item()
    '''
    
    def training_step(self, batch, batch_idx):
        wav_batch, wav_lengths, targets, target_lengths, texts = batch
        
        wav_batch = wav_batch
        wav_lengths = wav_lengths
        targets = targets
        target_lengths = target_lengths
        
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
        
        # # Проверка валидности длин для CTC
        # valid_batch = True
        # for j in range(input_lengths.size(0)):
        #     if input_lengths[j] < target_lengths[j]:
        #         self.logger.warning(f"Пропуск примера {j} в батче: input_lengths {input_lengths[j]} < target_lengths {target_lengths[j]}")
        #         valid_batch = False
        #         break
            
        # CTC Loss
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        #! we don't need to call loss.backward() because pytorch lightning will do it by itself
        return loss

    def validation_step(self, batch, batch_idx):
        wav_batch, wav_lengths, targets, target_lengths, texts = batch
        
        wav_batch = wav_batch
        wav_lengths = wav_lengths
        targets = targets
        target_lengths = target_lengths
        
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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):    
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        if self.scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
        else:
            return optimizer
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def _compute_ctc_loss(self, logprobs, encoded_len, transcripts, transcript_lengths):
        # Проверяем и выравниваем длины
        encoded_len = tuple(encoded_len.to('cpu').numpy())
        
        # Убеждаемся, что encoded_len не превышает длину logprobs по времени
        T = logprobs.size(1)  # временная размерность после transpose
        encoded_len = tuple(min(el, T) for el in encoded_len)

        # CTCLoss требует логиты в формате (T, N, C)
        logprobs = logprobs.transpose(0, 1)  # Теперь форма (T, N, C)

        BLANK_IDX = 33
        ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)

        # Вычисляем потерю
        loss = ctc_loss(
            logprobs,           # (T, N, C)
            transcripts,        # (N, S) -> целочисленные индексы
            encoded_len,        # (N,) -> длины выходных последовательностей
            transcript_lengths  # (N,) -> длины целевых последовательностей
        )

        return loss
