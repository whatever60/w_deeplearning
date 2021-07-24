import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


tokenizer = None


def process_data(data_path, cache_path, cached=False):
    df = pd.read_csv(data_path)
    df["Sentence #"].fillna(method="ffill", inplace=True)

    if cached:
        meta_data = joblib.load("meta.bin")
        enc_pos = meta_data["enc_pos"]
        enc_tag = meta_data["enc_tag"]
    else:
        enc_pos = preprocessing.LabelEncoder().fit(df.POS)
        enc_tag = preprocessing.LabelEncoder().fit(df.Tag)
        meta_data = dict(enc_pos=enc_pos, enc_tag=enc_tag)
        joblib.dump(meta_data, cache_path)

    df["POS"] = enc_pos.transform(df.POS)
    df["Tag"] = enc_tag.transform(df.Tag)

    group = df.groupby("Sentence #")
    sentences = group["Word"].apply(list).values
    poses = group["POS"].apply(list).values
    tags = group["Tag"].apply(list).values

    return sentences, poses, tags


class EntityDataset(Dataset):
    def __init__(self, texts, poses, tags, max_length):
        # texts: [["hello", ",", "world", "!"], ["Attention", "is", "all", "you", "need", "."], ...]
        # pos / tags: [[4, 2, 3, 4], [4, 2, 2, 1, 5, 3], ...]
        self.texts = texts
        self.poses = poses
        self.tags = tags
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        pos = self.poses[index]
        tag = self.tags[index]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            # why do we encode one word at a time?
            inputs = tokenizer.encode(s, add_special_tokens=False)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * len(inputs))
            target_tag.extend([tag[i]] * len(inputs))

        ids = ids[: self.max_length - 2]
        target_pos = target_pos[: self.max_length - 2]
        target_tag = target_tag[: self.max_length - 2]

        # 101, 102 respectively for BERT.
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        padding_len = self.max_length - len(ids)

        ids.extend([0] * padding_len)
        mask.extend([0] * padding_len)
        token_type_ids.extend([0] * padding_len)
        target_pos.extend([-100] * padding_len)
        target_tag.extend([-100] * padding_len)
        assert all(
            len(i) == self.max_length
            for i in [ids, mask, token_type_ids, target_pos, target_tag]
        ), [
            len(i) == self.max_length
            for i in [ids, mask, token_type_ids, target_pos, target_tag]
        ]

        data = dict(
            ids=torch.tensor(ids).long(),
            mask=torch.tensor(mask).long(),
            token_type_ids=torch.tensor(token_type_ids).long(),
            target_pos=torch.tensor(target_pos).long(),
            target_tag=torch.tensor(target_tag).long(),
        )
        return data


class EntityDataModule(pl.LightningDataModule):
    name = "Kaggle - Annotated Corpus for Named Entity Recognition"
    num_classes_pos = 42
    num_classes_tag = 17

    length_train = int(47_959 / 10 * 9)
    length_val = 47_959 - length_train

    def __init__(self, texts, poses, tags, max_length, batch_size):
        super().__init__()
        self.sentences = texts
        self.poses = poses
        self.tags = tags
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            (
                train_sentences,
                val_sentences,
                train_pos,
                val_pos,
                train_tag,
                val_tag,
            ) = train_test_split(self.sentences, self.poses, self.tags, test_size=0.1)
            self.dataset_train = EntityDataset(
                train_sentences, train_pos, train_tag, self.max_length
            )
            self.dataset_val = EntityDataset(
                val_sentences, val_pos, val_tag, self.max_length
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            # pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            # pin_memory=True,
            drop_last=False,
        )
