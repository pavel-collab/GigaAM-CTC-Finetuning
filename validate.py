from src.models.utils import import_gigaam_model, get_model_vocab_idx2char, get_model_vocab_char2idx
from src.data.preprocess import normalize_text
from src.utils.utils import calculate_wer
from src.data.dataset import AudioDataset
from src.data.utils import collate_fn_wrapper

from torch.utils.data import DataLoader
import argparse
import torch

BLANK_IDX = 33

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='set a path to a saved checkpoint')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint_path
    model = import_gigaam_model(
                model_type='ctc',
                checkpoint_path=checkpoint_path,
                device=device
            )

    char2idx = get_model_vocab_char2idx(model)

    val_dataset = AudioDataset(dataset_part="validation", normalize_fn=normalize_text)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: collate_fn_wrapper(x, char2idx),
        # pin_memory=True if torch.cuda.is_available() else False
    )

    idx2char = get_model_vocab_idx2char(model)

    wer, refs, hyps = calculate_wer(model, val_loader, device, idx2char, BLANK_IDX)
    print('WER on validation dataset is: ', wer)
