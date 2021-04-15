import torch
from torch import nn
from torch import optim
from torch.utils import data
import pytorch_lightning as pl

import nlp
import transformers


class IMDBSentimentDataset(data.Dataset):
    def __init__(self, tokenizer, dataset, seq_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        encoding, = self.tokenizer(
            data['text'],
            truncation=True,
            max_length=self.seq_length,
            padding='max_length',
            return_tensors='pt',
        ),
        return dict(
            input_ids=encoding['input_ids'][0],
            attention_mask=encoding['attention_mask'][0],
            label=data['label']
        )



class SentimentClassifier(pl.LightningModule):
    def __init__(self, model, seq_length, debug, batch_size, lr, momentum):
        super().__init__()
        self.save_hyperparameters()
        self.model = transformers.BertForSequenceClassification.from_pretrained(model)
        self.criterion = nn.CrossEntropyLoss()

    def prepare_data(self):
        tokenizer = transformers.BertTokenizerFast.from_pretrained(self.hparams.model)
        # seq_length = self.hparams.seq_length
        # def tokenize(x):
        #     x['input_ids'] = tokenizer(
        #         x['text'],
        #         max_length=seq_length,
        #         truncation=True,
        #         # pad_to_max_length=True,
        #         padding='max_length',
        #         return_tensors='pt'
        #     )['input_ids']
        #     # x['label'] = x['label'].copy()
        #     return x

        # def prepare_dataset(split):
        #     split_pct = self.hparams.batch_size if self.hparams.debug else '5%'
        #     dataset = nlp.load_dataset('imdb', split=f'{split}[:{split_pct}]')
        #     # dataset = dataset.map(tokenize, batched=True)
        #     dataset = tokenize(dataset)
        #     # dataset.set_format(type='torch', columns=['input_ids', 'label'])
        #     return dataset
        split_pct = self.hparams.batch_size if self.hparams.debug else '100%'
        train_dataset = nlp.load_dataset('imdb', split=f'train[:{split_pct}]')
        val_dataset = nlp.load_dataset('imdb', split=f'test[:{split_pct}]')
        self.train_dataset = IMDBSentimentDataset(tokenizer, train_dataset, self.hparams.seq_length)
        self.val_dataset = IMDBSentimentDataset(tokenizer, val_dataset, self.hparams.seq_length)
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    
    def configure_optimizers(self):
        # return optim.SGD(
        #     self.model.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=self.hparams.momentum
        # )
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask).logits
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['label'])
        self.log('loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['label'])
        acc = (logits.argmax(dim=1) == batch['label']).float()
        self.log_dict(dict(val_loss=loss, acc=acc))
    

if __name__ == '__main__':
    pl.seed_everything(2021)
    DEBUG = False
    MODEL = 'bert-base-uncased'
    SEQ_LENGTH = 200
    BATCH_SIZE = 32
    LR = 5e-5
    MOMENTUM = 0.9
    net = SentimentClassifier(
        MODEL,
        SEQ_LENGTH,
        DEBUG,
        BATCH_SIZE,
        LR,
        MOMENTUM,
    )

    trainer = pl.Trainer(gpus=[9], deterministic=True, fast_dev_run=DEBUG)
    trainer.fit(net)
