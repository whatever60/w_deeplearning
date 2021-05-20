from transformers import AutoTokenizer
from datasets import load_dataset

block_size = 1024
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file, use_fast=True)
DATA_DIR = './data/wikitext-103-raw/'
data_files = dict(
    train=DATA_DIR + 'wiki.train.raw',
    validation=DATA_DIR + 'wiki.valid.raw',
    test=DATA_DIR + 'wiki.test.raw'
)
raw_datasets = load_dataset('text', data_files=data_files)

column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
)


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8,
)