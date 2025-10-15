from src.data.dataset import AudioDataset, collate_fn
from src.utils.utils import get_gigaam_model, fix_torch_seed, calculate_wer_on_dataset

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

# import target dataset
fleurs = load_dataset("google/fleurs", "ru_ru")
fleurs = fleurs['train']

# make a dataset and daataloader from train part of data
dataset = AudioDataset(fleurs)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=1
)

# set a model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_gigaam_model(device=device)
model = model.to(device)

# fix seed 
fix_torch_seed()

wer, references, hypotheses = calculate_wer_on_dataset(model, dataloader, return_transcriptions=True)
print('\nWER on farfield: ', wer)