from src.data.utils import FunctionFactory

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

        # return {
        #     'audio': torch.tensor(sample['audio']['array'], dtype=torch.float32),
        #     'num_samples': torch.tensor(sample['num_samples'], dtype=torch.int32),
        #     'transcription': sample['transcription'],
        # }

        # Аудио
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']
        wav = torch.from_numpy(audio_array).float()
        
        # Нормализация к моно
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        
        # Текст
        text = self.normalize_fn(sample['transcription'])
        
        return wav, text, sampling_rate

class CTCDataModule(pl.LightningDataModule):
    def __init__(self, config, model_vocab=None):
        super().__init__()
        self.config = config

        self.collate_fn = None
        fn_factory = FunctionFactory(model_vocabular=model_vocab)
        if model_vocab is None:
           self.collate_fn = fn_factory.create_default_collate_function()
        else:
           self.collate_fn = fn_factory.create_advanced_collate_function()        
        assert(self.collate_fn is not None)


    def setup(self, stage=None):
        self.train_dataset = AudioDataset(dataset_part="train")
        self.val_dataset = AudioDataset(dataset_part="validation")

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            collate_fn=self.collate_fn
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return self.val_loader