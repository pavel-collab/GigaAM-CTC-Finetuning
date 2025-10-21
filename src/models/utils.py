from src.data.preprocess import preprocess_text

from gigaam import GigaAMASR
import gigaam
import os
import torch
import re
from typing import List, Dict

def get_model_vocab(model):
    model_vocab = {sym: idx for idx, sym in enumerate(model.decoding.tokenizer.vocab)}
    return model_vocab

def import_gigaam_model(model_type: str="ctc", checkpoint_path: str=None, device: str="cpu"):
    CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
    model_name, model_path = gigaam._download_model(model_type, CACHE_DIR)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    ckpt["cfg"].encoder.flash_attn = False
    model = GigaAMASR(ckpt['cfg'])

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.eval()

    model.to(device)

    #if device.type != "cpu":
    #    model.encoder = model.encoder.half()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    return model

def get_gigaam_logprobs(model, wav_batch, wav_lengths, return_transcriptions=False):
    wav_batch = wav_batch.to(model._device)
    wav_lengths = wav_lengths.to(model._device)

    encoded, encoded_len = model.forward(wav_batch, wav_lengths)

    logprobs = model.head(encoded)

    if return_transcriptions:
        transcriptions = model.decoding.decode(model.head, encoded, encoded_len)
        return logprobs, encoded_len, transcriptions
    else:
        return logprobs, encoded_len

def get_texts_idxs(texts: List[str], model_vocab: Dict[str, str]) -> torch.Tensor:
  texts_idxs = []
  for text in texts:
    #print(f"[DEBUG] {text}")

    text = preprocess_text(text)

    #print(f"[DEBUG] preprocessed text: {text}")

    text_idxs = [model_vocab[sym] for sym in text]
    texts_idxs.append(text_idxs)

  return torch.tensor(texts_idxs, dtype=torch.int)

# def load_checkpoint(model, checkpoint_path: str, device):
#     """
#     Загрузка чекпоинта
    
#     Args:
#         checkpoint_path: путь к чекпоинту
#     """
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model
