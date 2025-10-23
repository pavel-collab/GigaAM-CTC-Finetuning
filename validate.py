from src.models.utils import import_gigaam_model
from utils import calculate_wer_on_dataset, calculate_wer
from src.data.dataset import AudioDataset
from src.data.utils import collate_fn

from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    #TODO: позже здесь нужно будет написать импорт модели с определенного чекпоинта
    model = import_gigaam_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    val_dataset = AudioDataset(preprocessor=None, dataset_part="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        # pin_memory=True if torch.cuda.is_available() else False
    )

    # wer = calculate_wer_on_dataset(model=model, dataloader=val_loader)
    wer, _, _ = calculate_wer(
        model=model,
        dataloader=val_loader,
        device=device
    )
    print('WER on validation dataset is: ', wer)