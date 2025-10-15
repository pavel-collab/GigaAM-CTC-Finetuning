import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'id': sample['id'],
            'num_samples': torch.tensor(sample['num_samples'], dtype=torch.int32),
            'path': sample['path'],
            'audio': torch.tensor(sample['audio']['array'], dtype=torch.float32),
            'transcription': sample['transcription'],
            'raw_transcription': sample['raw_transcription'],
            'gender': sample['gender'],
            'lang_id': torch.tensor(sample['lang_id'], dtype=torch.int32),
            'language': sample['language'],
            'lang_group_id': torch.tensor(sample['lang_group_id'], dtype=torch.int32)
        }
    
def collate_fn(batch):
    # Для полей с разной длиной (например, audio) нужно добавить паддинг
    audio = [item['audio'] for item in batch]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
    
    return audio_padded, torch.stack([item['num_samples'] for item in batch]), [item['transcription'] for item in batch]