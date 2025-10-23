from src.data.utils import collate_fn

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from typing import Dict
import torch

class AudioDataset(Dataset):   
    def __init__(self, dataset_part: str="train"):
       self.dataset = load_dataset("google/fleurs", "ru_ru")

       if dataset_part == "train":
        self.dataset = self.dataset['train']
       elif dataset_part == "validation":
        self.dataset = self.dataset['validation']
       else: 
          self.dataset = self.dataset['test']
   
    def __len__(self) -> int:
        return len(self.dataset)
   
    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        return {
            'audio': torch.tensor(sample['audio']['array'], dtype=torch.float32),
            'num_samples': torch.tensor(sample['num_samples'], dtype=torch.int32),
            'transcription': sample['transcription'],
        }

class CTCDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(dataset_part="train")
        self.val_dataset = AudioDataset(dataset_part="validation")

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_fn
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

        return self.val_loader