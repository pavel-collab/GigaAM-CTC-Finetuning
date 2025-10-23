from src.models.utils import import_gigaam_model, get_model_vocab, get_texts_idxs, get_gigaam_logprobs

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class CTCLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Сохраняем гиперпараметры
        self.save_hyperparameters()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = import_gigaam_model(model_type=self.config.model.name, device=self.device)

    def training_step(self, batch, batch_idx):
        audios, audio_lengths, texts = batch

        model_vocab = get_model_vocab(self.model)

        #TODO: maybe move it to the dataloader?
        texts = get_texts_idxs(texts, model_vocab)

        transcript_lengths=(len(sample) for sample in texts)

        logprobs, encoded_len = get_gigaam_logprobs(self.model, audios.to(self.device), audio_lengths.to(self.device))

        loss =  self._compute_ctc_loss(
                    logprobs,
                    encoded_len,
                    texts,
                    transcript_lengths=tuple(transcript_lengths)
                )
        
        return loss.item()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        audios, audio_lengths, texts = batch
        
        model_vocab = get_model_vocab(self.model)
        texts = get_texts_idxs(texts, model_vocab)

        logprobs, encoded_len = get_gigaam_logprobs(self.model, audios.to(self.device), audio_lengths.to(self.device))

        transcript_lengths=(len(sample) for sample in texts)
        loss = self._compute_ctc_loss(
                    logprobs, 
                    encoded_len,
                    texts,
                    transcript_lengths=tuple(transcript_lengths)
                )
        
        self.model.train()
        return loss.item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        return optimizer

    def _compute_ctc_loss(self, logprobs, encoded_len, transcripts, transcript_lengths):
        # Проверяем и выравниваем длины
        encoded_len = tuple(encoded_len.to('cpu').numpy())
        
        # Убеждаемся, что encoded_len не превышает длину logprobs по времени
        T = logprobs.size(1)  # временная размерность после transpose
        encoded_len = tuple(min(el, T) for el in encoded_len)

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