import os
import argparse
import logging
import pickle as pkl

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

import kaggleutils as kutils

parser = argparse.ArgumentParser(description="Clean and tokenize the data.")
parser.add_argument("-d", "--data", default="../input/", help="Path to the Toxic data")
parser.add_argument(
    "-s", "--save", default="../processed_input/", help="Path to save the new data to."
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Activates additional logging."
)

PUNCT = "/-'?!.,#$%'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + "∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&"
CLEAN_TABLE = str.maketrans(PUNCT, " " * len(PUNCT))
IDENTITY = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]
LABELS = ["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
tqdm.pandas()


def load_data(path_to_data="../input", train_name="train.csv", test_name="test.csv"):
    path_to_train_csv = os.path.join(path_to_data, train_name)
    # Open the csv file
    logging.info(f"Reading train CSV from {path_to_train_csv}")
    train_csv = pd.read_csv(path_to_train_csv)
    train_ids = train_csv["id"]
    train_texts = train_csv["comment_text"].fillna("NA")
    train_labels = train_csv[LABELS].fillna(0)
    train_identities = train_csv[IDENTITY].fillna(0)
    logging.info("Loaded %s samples from %s" % (len(train_texts), path_to_train_csv))

    path_to_test_csv = os.path.join(path_to_data, test_name)
    logging.info("Reading test CSV from %s" % path_to_test_csv)
    # Open the csv file
    test_csv = pd.read_csv(path_to_test_csv)
    test_ids = test_csv["id"]
    test_texts = test_csv["comment_text"].fillna("NA")
    logging.info("Loaded %s samples from %s" % (len(test_texts), path_to_test_csv))

    return (
        (train_ids, train_texts, train_labels, train_identities),
        (test_ids, test_texts),
    )


def clean_special_chars(text, clean_table=CLEAN_TABLE):
    return text.translate(CLEAN_TABLE)


def preprocess(texts):
    logging.info("Cleaning out special characters from texts...")
    texts = texts.astype(str).progress_apply(clean_special_chars)
    return texts


def tokenize(*texts):
    tokenizer = Tokenizer()

    def text_gen(*all_texts):
        for text in all_texts:
            for comment in text.values:
                yield comment

    # Fit the tokenizer
    text_count = sum(len(text) for text in texts)
    logging.info(f"Fitting tokenizer on {text_count} comments...")
    tokenizer.fit_on_texts(text_gen(*texts))
    logging.info(f"Found {len(tokenizer.word_index)} words in texts")
    return tokenizer.word_index


def map_indicies(texts, word_index):
    logging.info("Mapping the words to their indicies...")
    texts = texts.progress_apply(
        lambda x: [word_index[w] for w in text_to_word_sequence(x)]
    )
    return texts


def process_texts(
    train_name="train.csv",
    test_name="test.csv",
    path_to_data="../input",
    save_dest="../processed_input/",
):
    # Load the dsata
    (train_ids, train_texts, train_labels, train_identities), (
        test_ids,
        test_texts,
    ) = load_data(path_to_data, train_name, test_name)

    # Preprocess the texts
    train_texts = preprocess(train_texts)
    test_texts = preprocess(test_texts)

    # Tokenize the texts
    word_index = tokenize(train_texts, test_texts)

    # Map the texts to their indicies and save
    train_texts = map_indicies(train_texts, word_index)
    save_text(
        "train",
        train_ids.values,
        train_texts.values,
        labels=train_labels.values,
        identities=train_identities.values,
        save_dest=save_dest,
    )
    del train_texts
    del train_ids
    del train_labels
    del train_identities

    test_texts = map_indicies(test_texts, word_index)
    save_text("test", test_ids.values, test_texts.values, save_dest=save_dest)
    del test_ids
    del test_texts

    # Save the word index
    save_word_index(word_index, save_dest=save_dest)


def save_text(
    name, ids, text, labels=None, identities=None, save_dest="../processed_input/"
):
    save_path = kutils.safe_create_file(os.path.join(save_dest, name + ".npz"))
    if labels is not None:
        assert identities is not None
        np.savez(save_path, id=ids, text=text, label=labels, identity=identities)
    else:
        np.savez(save_path, id=ids, text=text)
    logging.info(f"Saved {name} to {save_path}")


def save_word_index(word_index, save_dest="../processed_input/"):
    save_path = kutils.safe_create_file(os.path.join(save_dest, "word_index.pkl"))
    with open(save_path, "wb") as word_index_file:
        pkl.dump(word_index, word_index_file)
    logging.info(f"Saved Word Index to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    # Set up Logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=log_level
    )

    # Process the data
    process_texts(path_to_data=args.data, save_dest=args.save)

