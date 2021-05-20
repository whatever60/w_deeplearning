import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


class EsperantoDataset(Dataset):
    def __init__(self, files, vocab, merges):
        tokenizer = ByteLevelBPETokenizer(vocab, merges)
        tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        ids = []
        for file in files:
            with open(file) as f:
                # do this only when your files are not too large
                lines = f.readlines()
            ids.extend([x.ids for x in tokenizer.encode_batch(lines)])
        self.ids = torch.tensor(ids)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.ids[index])


class 
