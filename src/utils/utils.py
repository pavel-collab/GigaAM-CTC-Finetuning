import random
import numpy as np
import torch
import os
from tqdm import tqdm
import pywer
import gigaam
from gigaam import GigaAMASR

def fix_torch_seed(seed: int = 42):
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

    print(f"✅ Random seed fixed to {seed}")

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
    
def calculate_wer_on_dataset(model, dataloader, batch_size=8, num_workers=2, return_transcriptions=False):
  # dataloader = torch.utils.data.DataLoader(
  #     dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
  # )
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

def decode_indices(labels, model):
    return "".join(model.decoding.tokenizer.decode(labels.cpu().tolist()))

def get_gigaam_model(device):
    CACHE_DIR = os.path.expanduser("~/.cache/gigaam")

    model_name, model_path = gigaam._download_model('ctc', CACHE_DIR)

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    ckpt["cfg"].encoder.flash_attn = False
    model = GigaAMASR(ckpt['cfg'])

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.eval()

    if device.type != "cpu":
        model.encoder = model.encoder.half()

    return model
