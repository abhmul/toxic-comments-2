import os
import argparse
import pickle as pkl
import logging

import numpy as np
from gensim.models import KeyedVectors, FastText
from tqdm import tqdm

import kaggleutils as kutils

parser = argparse.ArgumentParser(
    description="Construct an embeddings matrix for the data."
)
parser.add_argument(
    "-e", "--embeddings", nargs="+", required=True, help="Path to embeddings"
)
parser.add_argument(
    "-w",
    "--word_index",
    default="../processed_input/word_index.pkl",
    help="Path to word_index",
)
parser.add_argument(
    "-s",
    "--save",
    default="../embeddings/embeddings.npz",
    help="File path to save embeddings and missing indicies to.",
)
parser.add_argument(
    "--embeddings_type",
    nargs="+",
    default=["fasttext"],
    help="Type of embeddings to load",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Activates additional logging."
)


def load_glove_embeddings(embeddings_path, word_index):
    logging.info(f"Reading in GloVe embeddings from {embeddings_path}")
    # Get the embedding dim first
    embedding_dim = None
    with open(embeddings_path) as f:
        for line in f:
            values = line.split()
            embedding_dim = len(np.asarray(values[-300:], dtype="float32"))
            break

    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    not_missing = set()
    with open(embeddings_path) as f:
        for line in tqdm(f):
            values = line.split()
            word = " ".join(values[:-300])
            coefs = np.asarray(values[-300:], dtype="float32")
            if word in word_index:
                embeddings[word_index[word]] = coefs
                not_missing.add(word_index[word])

    # Figure out which words are missing
    missing = set(range(1, len(word_index) + 1)) - not_missing
    logging.info(
        f"Loaded {len(embeddings)} Glove embeddings with {len(missing)} missing words."
    )
    return embeddings, missing


def load_w2v_embeddings(embeddings_path, word_index):
    # Check the extension
    extension = os.path.splitext(embeddings_path)[1]
    is_binary = extension == ".bin"
    logging.info(
        f"Reading in {'binary' if is_binary else 'text'} Word2Vec embeddings from {embeddings_path}"
    )
    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    embedding_dim = word_vectors.vector_size

    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    missing = set()
    for word, i in tqdm(word_index.items()):
        if word in word_vectors.vocab:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    logging.info(
        f"Loaded {len(embeddings)} Word2vec embeddings with {len(missing)} missing words."
    )
    return embeddings, missing


def load_fasttext_embeddings(embeddings_path, word_index):
    logging.info(f"Reading in FastText embeddings from {embeddings_path}")
    try:
        word_vectors = FastText.load_fasttext_format(embeddings_path)
    except NotImplementedError:
        word_vectors = FastText.load(embeddings_path)
    embedding_dim = word_vectors.vector_size
    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    missing = set()
    for word, i in tqdm(word_index.items()):
        if word in word_vectors:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    logging.info(
        f"Loaded {len(embeddings)} FastText embeddings with {len(missing)} missing words."
    )
    return embeddings, missing


def load_embeddings(embeddings_path, word_index, embeddings_type="fasttext"):
    if embeddings_type == "word2vec":
        return load_w2v_embeddings(embeddings_path, word_index)
    elif embeddings_type == "glove":
        return load_glove_embeddings(embeddings_path, word_index)
    elif embeddings_type == "fasttext":
        return load_fasttext_embeddings(embeddings_path, word_index)
    raise NotImplementedError("Embeddings type %s is not supported" % embeddings_type)


def load_many_embeddings(embeddings_paths, word_index, embeddings_types=("fasttext",)):
    # Keras starts indexing from 1
    assert len(word_index) == max(word_index.values())

    embeddings, missings = zip(
        *(
            load_embeddings(path, word_index, embedding_type)
            for path, embedding_type in zip(embeddings_paths, embeddings_types)
        )
    )

    logging.info(f"Loaded {len(embeddings)} embeddings.")
    embeddings = np.concatenate(embeddings, axis=1)
    missing = set.intersection(*missings)
    return embeddings, missing


def save_embeddings(embeddings, missing, save_path="../embeddings/embeddings.npz"):
    missing = np.array(list(missing))
    save_path = kutils.safe_create_file(save_path)
    np.savez(save_path, embeddings=embeddings, missing=missing)
    logging.info(f"Saved the embeddings and missing indicies to {save_path}")


def load_word_index(word_index_path):
    logging.info(f"Reading word index from {word_index_path}...")
    with open(word_index_path, "rb") as word_index_file:
        word_index = pkl.load(word_index_file)
    return word_index


if __name__ == "__main__":
    args = parser.parse_args()
    # Set up Logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=log_level
    )

    word_index = load_word_index(args.word_index)
    embeddings_matrix, missing_indicies = load_many_embeddings(
        args.embeddings, word_index, embeddings_types=args.embeddings_type
    )
    save_embeddings(embeddings_matrix, missing_indicies, save_path=args.save)

