import pickle

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing
from torchtext.datasets.wikitext2 import WikiText2
from transformers import AutoTokenizer


def prepare_tokenizer(data_dir_train, tokenizer_path) -> None:
    """
    Args:
        data_dir_train: str. Path to raw text file.
    """
    tokenizer_path = "./tokenizer/tokenizer.json"
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.train(
        files=[data_dir_train],
        vocab_size=20_000,
        min_frequency=1,
        special_tokens=["<|endoftext|>", "<unk>"],
    )
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )
    tokenizer.save(tokenizer_path)


def tokenize(
    tokenizer_path,
    data_dir_train,
    data_dir_val,
    data_dir_test,
    save_data_dir,
    pretrained=False,
) -> None:
    with open(data_dir_train) as f:
        entries_train = f.read().split(" \n \n = ")
    with open(data_dir_val) as f:
        entries_val = f.read().split(" \n \n = ")
    with open(data_dir_test) as f:
        entries_test = f.read().split(" \n \n = ")

    if pretrained:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = tokenizer.vocab_size
        data_list_train = [encoding for encoding in tokenizer(entries_train).input_ids]
        data_list_val = [encoding for encoding in tokenizer(entries_val).input_ids]
        data_list_test = [encoding for encoding in tokenizer(entries_test).input_ids]
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        data_list_train = [
            encoding.ids for encoding in tokenizer.encode_batch(entries_train)
        ]
        data_list_val = [
            encoding.ids for encoding in tokenizer.encode_batch(entries_val)
        ]
        data_list_test = [
            encoding.ids for encoding in tokenizer.encode_batch(entries_test)
        ]
    with open(save_data_dir, "wb") as f:
        pickle.dump([vocab_size, data_list_train, data_list_val, data_list_test], f)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()

    DATA_DIR = "./data/"
    DATA_DIR_TRAIN = DATA_DIR + "wikitext-2/wiki.train.tokens"
    DATA_DIR_VAL = DATA_DIR + "wikitext-2/wiki.valid.tokens"
    DATA_DIR_TEST = DATA_DIR + "wikitext-2/wiki.test.tokens"
    TOKENIZER_PATH = "./tokenizer/tokenizer.json"
    SAVE_DATA_DIR = "./data/tokenized.pkl"
    WikiText2(DATA_DIR)
    prepare_tokenizer(DATA_DIR_TRAIN, TOKENIZER_PATH)
    tokenize(TOKENIZER_PATH, DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST, SAVE_DATA_DIR)
