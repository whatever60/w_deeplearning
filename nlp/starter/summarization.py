import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer

from rich import print
from rich.traceback import install

install()


class NewsSummary(Dataset):
    def __init__(
        self,
        source_encoding,
        source_attention_mask,
        target_encoding,
        target_attention_mask,
    ) -> None:
        super().__init__()
        self.source_encoding = source_encoding
        self.source_attention_mask = source_attention_mask
        self.target_encoding = target_encoding
        self.target_attention_mask = target_attention_mask

    def __len__(self):
        return len(self.source_encoding)

    def __getitem__(self, index):
        return (
            self.source_encoding[index],
            self.source_attention_mask[index],
            self.target_encoding[index],
            self.target_attention_mask[index],
        )


def get_T5_data(df, max_token_len_text, max_token_len_summary):
    text_raw, summary_raw = df["text"].tolist(), df["summary"].tolist()

    text = tokenizer(
        text_raw,
        max_length=max_token_len_text,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    summary = tokenizer(
        summary_raw,
        max_length=max_token_len_summary,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    text_encoding = text["input_ids"]
    text_attention_mask = text["attention_mask"]
    summary_encoding = summary["input_ids"].where(
        torch.tensor(0).byte(), torch.tensor(-100)
    )  # padding id is -100 for T5.
    summary_attention_mask = summary["attention_mask"]

    return (
        text_raw,
        text_encoding,
        text_attention_mask,
        summary_raw,
        summary_encoding,
        summary_attention_mask,
    )


class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer,
        batch_size,
        max_token_len_text,
        max_token_len_summary,
        cache_dir="./cached/data.pt",
    ):
        """
        Args:
            data_dir: The path to raw .csv file. Set to None if you want to directly use cached data.
            cache_dir: If data_dir is not None, tokenzied data will be stored in `cache_dir`. Otherwise, data is loaded from `cache_dir`.
        """
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len_text = max_token_len_text
        self.max_token_len_summary = max_token_len_summary
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        if self.data_dir is not None:
            df = pd.read_csv(self.data_dir)
            train_df, val_df = train_test_split(df, test_size=0.1)
            cache_data = {}
            (
                _,
                cache_data["text_encoding_train"],
                cache_data["text_attention_mask_train"],
                _,
                cache_data["summary_encoding_train"],
                cache_data["summary_attention_mask_train"],
            ) = get_T5_data(
                train_df, self.max_token_len_text, self.max_token_len_summary
            )

            (
                _,
                cache_data["text_encoding_val"],
                cache_data["text_attention_mask_val"],
                _,
                cache_data["summary_encoding_val"],
                cache_data["summary_attention_mask_val"],
            ) = get_T5_data(val_df, self.max_token_len_text, self.max_token_len_summary)
            torch.save(cache_data, self.cache_dir)

    def setup(self, stage=None) -> None:
        if stage == "fit":
            data = torch.load(self.cache_dir)
            self.dataset_train = NewsSummary(
                data["text_encoding_train"],
                data["text_attention_mask_train"],
                data["summary_encoding_train"],
                data["summary_attention_mask_train"],
            )
            self.dataset_val = NewsSummary(
                data["text_encoding_val"],
                data["text_attention_mask_val"],
                data["summary_encoding_val"],
                data["summary_attention_mask_val"],
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=8,
        )


class T5(pl.LightningModule):
    def __init__(self, huggingface_model, model_name, lr) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = huggingface_model.from_pretrained(model_name, return_dict=True)

    def forward(
        self,
        text_encoding,
        text_attention_mask,
        summary_encoding,
        summary_attention_mask,
    ):
        output = self.model(
            text_encoding,
            attention_mask=text_attention_mask,
            labels=summary_encoding,
            decoder_attention_mask=summary_attention_mask,
        )
        return output.loss, output.logits

    def shared_step(self, batch):
        (
            text_encoding,
            text_attention_mask,
            summary_encoding,
            summary_attention_mask,
        ) = batch
        loss, _ = self(
            text_encoding, text_attention_mask, summary_encoding, summary_attention_mask
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("loss_val", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)


def summarize(text, tokenizer, model):
    text_encoding = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    return ''.join(preds)


if __name__ == "__main__":
    MODEL = "t5-base"
    DATA_DIR = "~/wusuowei/data/kaggle/news_summary/news_summary_processed.csv"
    CACHE_DIR = "./cached/data.pt"
    LR = 1e-4
    BATCH_SIZE = 4
    MAX_TOKEN_LEN_TEXT = 512
    MAX_TOKEN_LEN_SUMMARY = 128
    TRAIN = False

    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    
    # datamodule.prepare_data()
    net = T5(T5ForConditionalGeneration, MODEL, LR)

    if TRAIN:
        datamodule = NewsSummaryDataModule(
            None, tokenizer, BATCH_SIZE, MAX_TOKEN_LEN_TEXT, MAX_TOKEN_LEN_SUMMARY
        )
        trainer = pl.Trainer(gpus=[1], deterministic=True)
        trainer.fit(net, datamodule)
    else:
        df = pd.read_csv(DATA_DIR)
        sample_row = df.iloc[1000]
        print('---- [yellow]text[/] ----')
        print(sample_row['text'])
        print('---- [yellow]summary gt[/] ----')
        print(sample_row['summary'])
        print('---- [yellow]summary pred[/] ----')
        print(summarize(sample_row['text'], tokenizer, net.model))

