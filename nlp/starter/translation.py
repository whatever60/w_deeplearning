import random

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from datasets import load_dataset, load_metric
    
from rich import print as rprint
from rich.traceback import install

install()

def train_tokenizer(model, normalizer, pre_tokenizer, trianer, files, post_processor):
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train()

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    # tokenizer for source
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def try_dataset(dataset):
    rprint('---- [yellow]Try dataset[/] ----')
    rprint(dataset)
    rprint(dataset["train"][0])


def try_metric(metric):
    rprint('---- [yellow]Try metric[/] ----')
    rprint(metric)
    fake_preds = ["hello there", "general kenobi"]
    fake_labels = [["hello there"], ["general kenobi"]]
    rprint(metric.compute(predictions=fake_preds, references=fake_labels))


def try_tokenizer(tokenizer):
    rprint('---- [yellow]Try tokenizer[/] ----')
    rprint(tokenizer.tokenize("Hello, this one sentence!"))
    rprint(tokenizer.encode("Hello, this one sentence!"))
    rprint(tokenizer("Hello, this one sentence!"))
    with tokenizer.as_target_tokenizer():
        # This context manager will make sure the tokenizer uses the special tokens corresponding to the targets.
        rprint(tokenizer.tokenize("Hello, this one sentence!"))
        rprint(tokenizer("Hello, this one sentence!"))
    with tokenizer.as_target_tokenizer():
        # This context manager will make sure the tokenizer uses the special tokens corresponding to the targets.
        rprint(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, p):
        super().__init__()
        self.dropout = nn.Identity()  # nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout=p)
    
    def forward(self, x):
        # x: [seq_length, batch]. The entire sentence.
        embedding = self.dropout(self.embedding(x))  # [seq_length, batch, emb_dim]
        outputs, (hn, cn) = self.rnn(embedding)  # hn: [batch, hid_dim], cn: [batch, hid_dim]
        return hn, cn


class Decoder(nn.Module):
    def __init__(self, vocab_size_source, emb_dim, hid_dim, num_layers, vocab_size_target, p):
        super().__init__()
        self.dropout = nn.Identity()  # nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size_source, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout=p)  # hid_dim of the encoder and decoder is the same.
        self.fc = nn.Linear(hid_dim, vocab_size_target)

    def forward(self, x, h0, c0):
        # x: [batch], Previous predicted word.
        x = x.unsqueeze(0)  # [1, batch]
        embedding = self.dropout(self.embedding(x))  # [1, batch, embedding]
        outputs, (hn, cn) = self.rnn(embedding, (h0, c0))  # outputs: [1, batch, hid_dim]
        logit = self.fc(outputs).squeeze(0)  # [batch, vocab_size_target]
        return logit, (hn, cn)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_source, vocab_size_target, emb_dim_encoder, emb_dim_decoder, hid_dim, num_layers_encoder, num_layers_decoder, p_encoder, p_decoder):
        super().__init__()
        self.encoder = Encoder(vocab_size_source, emb_dim_encoder, hid_dim, num_layers_encoder, p_encoder)
        self.decoder = Decoder(vocab_size_source, emb_dim_decoder, hid_dim, num_layers_decoder, vocab_size_target, p_decoder)
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        hn, cn = self.encoder(source)
        outputs = []
        x = target[0]  # start token
        for idx, target_token in enumerate(target):
            output, (hn, cn) = self.decoder(target_token, hn, cn)
            outputs.append(output)
            best_guess = output.argmax(1)
            x = target[idx] if random.random() < teacher_force_ratio else best_guess
        return torch.stack(outputs)


def test_lstm():
    model = Seq2Seq(1000, 1000, 200, 200, 500, 2, 2, 0.2, 0.2)
    source = torch.randint(0, 1000, (20, 10))
    target = torch.randint(0, 1000, (20, 10))
    rprint(model(source, target).shape)


class Net(pl.LightningModule):
    def __init__(
        self,
        vocab_size_source,
        vocab_size_target,
        emb_dim_encoder,
        emb_dim_decoder,
        hid_dim,
        num_layers_encoder,
        num_layers_decoders,
        p_encoder,
        p_decoder,
        lr,
        pad_idx,
    ):  
        self.save_hyperparameters()
        self.model = Seq2Seq(vocab_size_source, vocab_size_target, emb_dim_encoder, emb_dim_decoder, hid_dim, num_layers_encoder, num_layers_decoders, p_encoder, p_decoder)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, source):
        return self.model(source)
    
    def training_step(self, batch, batch_idx):
        source, target = batch
        preds = self(source)  # [target_length, batch, target_vocab_size]
        loss = self.criterion(preds[1:].view(-1, preds.shape[2]))
        # clip the gradient
        self.log('loss_train', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


def train_huggingface(model, batch_size, tokenizer, tokenized_datasets, compute_metrics):
    args = Seq2SeqTrainingArguments(
        "huggingface_logs",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    test_lstm()
    # model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
    # batch_size = 16
    # max_input_length = 32
    # max_target_length = 32
    # source_lang = "en"
    # target_lang = "ro"

    # raw_datasets = load_dataset("wmt16", "ro-en")
    # metric = load_metric("sacrebleu")
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # if "mbart" in model_checkpoint:
    #     tokenizer.src_lang = "en-XX"
    #     tokenizer.tgt_lang = "ro-RO"

    # if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    #     prefix = "translate English to Romanian: "
    # else:
    #     prefix = ""


    # BATCH_SIZE = 64
    # LR = 1e-3
    # EMB_DIM = 300  # embedding dimension of encoder and decoder can be different
    # HID_DIM = 1024  # hidden dimension of encoder and decoder must be the same
    # NUM_LAYERS = 1024
    # P = 0.5



    # tokenizer_source = ''
    # tokenizer_target = ''



    # tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=4)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # train_huggingface(model, batch_size, tokenizer, tokenized_datasets, compute_metrics)