import pickle

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing
from torchtext.datasets.wikitext2 import WikiText2
from transformers import AutoTokenizer


def prepare_tokenizer(data_dir_train, tokenizer_path) -> None:
    """
    This function trains a Byte Level BPE tokenizer (what GPT2 used) from scratch using 
    the `tokenizer` package, and save it to the specified `.json` file.

    Vocabulary size is set to 20k, because the maximum size is approximately 40k. Minimum 
    frequency is set to 1.

    An `<|endoftext|>` token is added to the end of each tokenized entry. This is 
    different from what GPT did, who insert it in the front. I did this because I thought 
    `<|endoftext|>` should appear at the end, as implied by its name. An `<unk>` token is 
    also added as special token after noticing its existence in the raw text.

    Args:
        data_dir_train: str. Path to raw text file. Like the model, the tokenizer is 
            also trained only on the training data.
        tokenizer_path: str. Save the trained tokenizer to this file. Should be a `.json` 
            file.
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
    """
    Given path to text files and trained tokenizer name or path, read in text file, split 
    it into a list of entries, load tokenizer, do tokenization, and save the tokenized 
    results using `pickle`. The tokenized object is a list of list, with each "sublist" 
    being a list of token ids of corresponding entries, so the sublists are not of the 
    same length.

    This function is provided as a script highly specific to my use case, where the 
    dataset is provided in plain text format, the files are small in size so they can be 
    loaded into memory without streaming, the dataset (Wikitext) is composed of multiple 
    standalone entries instead of being a coherent piece like a novel.

    The separater ` \n \n = ` is chosen after some eye-checking. This separater can split 
    the text into Wiki entries as we want.

    One can use pretrianed tokenizers from Huggingface 🤗, or use your own tokenizer. 
    Because Huggingface 🤗 tokenizers of the `transformers` package and tokenizers of the 
    `tokenizers` packages provide different APIs, so scripts are wriiten for them 
    respectively. By setting the `pretrained` parameter to True, your `tokenizer_path` 
    will be passed to `AutoTokenizer.from_pretrained` API of `transformers` package, 
    otherwise to `Tokenizer.from_file` API of `tokenizers` package.
    """
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
