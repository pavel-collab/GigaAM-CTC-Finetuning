from src.models.utils import get_gigaam_logprobs

import random
import torch
import os
import numpy as np
from tqdm import tqdm
import pywer
import jiwer

def fix_torch_seed(seed: int = 42) -> None:
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def calculate_wer_on_dataset(model, dataloader, return_transcriptions=False):
    references = []
    hypotheses = []

    for batch in tqdm(dataloader):
        wav_batch, wav_lengths, texts = batch
        logprobs, lengths, transcriptions = get_gigaam_logprobs(model, wav_batch, wav_lengths, return_transcriptions=True)
        references.extend(texts)
        hypotheses.extend(transcriptions)

    wer = pywer.wer(references, hypotheses)
    if return_transcriptions:
        return wer, references, hypotheses

    return wer

def greedy_decode(logprobs, idx_to_char, blank_id=33):
    """
    CTC greedy decoding согласно официальной реализации
    logprobs: (T, C) - логарифмы вероятностей, где C = 34 (33 символа + blank)
    """
    # Получаем наиболее вероятные символы для каждого временного шага
    max_indices = torch.argmax(logprobs, dim=-1).tolist()
    
    # Применяем CTC collapse как в официальном коде
    token_ids = []
    prev_token = blank_id
    
    for token in max_indices:
        if (token != prev_token or prev_token == blank_id) and token != blank_id:
            token_ids.append(token)
        prev_token = token
    
    return ''.join(idx_to_char.get(tok, '') for tok in token_ids)

def calculate_wer(model, dataloader, device, idx_to_char, blank_id=33):
    """Вычисление WER на датасете"""
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Вычисление WER"):
            wav_batch, wav_lengths, targets, target_lengths, texts = batch
            
            wav_batch = wav_batch.to(device)
            wav_lengths = wav_lengths.to(device)

            # Прямой проход
            features, feat_lengths = model.preprocessor(wav_batch, wav_lengths)
            encoded, encoded_len = model.encoder(features, feat_lengths)
            logprobs = model.head(encoded)
            logprobs = logprobs.log_softmax(dim=2)  # Важно: log_softmax

            # Декодирование каждого примера
            for i in range(logprobs.shape[0]):
                length = min(encoded_len[i].item(), logprobs.shape[1])
                example_logprobs = logprobs[i, :length, :]
                decoded_text = greedy_decode(example_logprobs, idx_to_char, blank_id)
                hypotheses.append(decoded_text)

            references.extend(texts)
    
    model.train()

    # Вычисление WER
    wer = jiwer.wer(references, hypotheses)
    
    return wer, references, hypotheses