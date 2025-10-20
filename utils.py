from src.models.utils import get_gigaam_logprobs

import random
import torch
import os
import numpy as np
from tqdm import tqdm
import pywer

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