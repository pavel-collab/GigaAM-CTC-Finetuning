from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CTCDataset(Dataset):
    def __init__(self, data_path, vocab_path):
        # Ваша реализация датасета
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # Возвращает (features, labels, input_length, target_length)
        pass

class CTCDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = CTCDataset(
            self.config.data.dataset.train_path,
            self.config.data.dataset.vocab_path
        )
        self.val_dataset = CTCDataset(
            self.config.data.dataset.val_path,
            self.config.data.dataset.vocab_path
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4
        )