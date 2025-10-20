import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class CTCLightningModule(pl.LightningModule):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Пример архитектуры модели
        self.encoder = nn.LSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.output_layer = nn.Linear(config.model.hidden_size, vocab_size)
        
        # Сохраняем гиперпараметры
        self.save_hyperparameters()

    def forward(self, x):
        features, _ = self.encoder(x)
        return self.output_layer(features)

    def training_step(self, batch, batch_idx):
        # ЗАГЛУШКА - переопределите этот метод для вашей функции потерь
        x, y, input_lengths, target_lengths = batch
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Здесь должна быть ваша кастомная CTC loss функция
        loss = self._compute_ctc_loss(
            log_probs=log_probs,
            targets=y,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, input_lengths, target_lengths = batch
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = self._compute_ctc_loss(
            log_probs=log_probs,
            targets=y,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        return optimizer

    def _compute_ctc_loss(self, log_probs, targets, input_lengths, target_lengths):
        """
        ЗАГЛУШКА - переопределите этот метод для вашей реализации CTC loss
        """
        # Пример стандартной CTC loss
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            zero_infinity=True
        )
        return loss