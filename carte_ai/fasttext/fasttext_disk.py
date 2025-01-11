import lmdb
import numpy as np
from tqdm import tqdm
from typing import List
from gensim.models import fasttext
import re
import huggingface_hub
import os
import functools


def export_to_lmdb(model: fasttext.FastText, lmdb_path: str):
    """Export a fastText model loaded through gensim to disk for fast and RAM-efficient
        random access.
    Args:
        model: FastText model loaded through gensim. Necessary since it gives easy
            access to the raw embeddings in a way the standard library doesn't.
        path: Path where the LMDB file will be exported.
    """
    # We expect fasttext to take up ~8GB, setting the map size to 12GB for
    # some headroom.
    env = lmdb.open(lmdb_path, map_size=12e9)
    print("Starting export process...")
    with env.begin(write=True) as txn:
        print("Storing word embeddings...")
        skipped = 0
        for idx, word in tqdm(enumerate(model.wv.index_to_key), desc="Storing words"):
            key_bytes = word.encode('utf-8')
            # This is the LMDB key size limit. Without this, the export fails as some
            # words strangely exceed 511 chars...
            if len(key_bytes) <= 511:
                txn.put(key_bytes, model.wv[word].tobytes())
            else:
                skipped += 1

        print("Storing subword (n-gram) embeddings...")
        for idx, embedding in tqdm(enumerate(model.wv.vectors_ngrams), desc="Storing subwords"):
            # Subwords are stored with keys "subword_{idx}"
            txn.put(f'subword_{idx}'.encode('utf-8'), np.array(embedding, dtype=np.float32).tobytes())

        # Store metadata (vocabulary size, embedding dimension, ngram size)
        txn.put(b'vocab_size', np.array(len(model.wv.index_to_key), dtype=np.int32).tobytes())
        txn.put(b'embedding_dim', np.array(model.wv.vectors_vocab.shape[1], dtype=np.int32).tobytes())
        txn.put(b'ngram_size', np.array(len(model.wv.vectors_ngrams), dtype=np.int32).tobytes())

    print(f"Export completed: {lmdb_path}")
    env.close()


class FastTextOnDisk:
    """RAM-efficient wrapper for fast random access of pre-trained FastText embeddings stored on disk."""

    def __init__(self, lmdb_path: str, min_n: int = 5, max_n: int = 5, bucket: int = 2_000_000, vector_size: int = 300):
        """Initialize FastTextOnDisk object.

        Attributes:
            lmdb_path: Path to LMDB base dir. If None, the file will be
              downloaded from
              https://huggingface.co/datasets/akachkach/fasttext-lmdb/.
            min_n: Minimum length of character n-grams.
            max_n: Maximum length of character n-grams.
            bucket: Number of buckets.
        """
        self.lmdb_path = lmdb_path
        # No lock or buffer initialization (meminit) since we're readonly.
        # No readahead since we'll be doing a lot of random reads.
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.vector_size = vector_size

    @classmethod
    def from_hugging_face(cls):
        base_lmdb_dir = './fasttext_lmdb'
        os.makedirs(base_lmdb_dir, exist_ok=True)
        repo = {'repo_id': 'akachkach/fasttext-lmdb', 'repo_type': 'dataset'}
        for filename in huggingface_hub.list_repo_files(**repo):
            huggingface_hub.hf_hub_download(
                filename=filename,
                local_dir=base_lmdb_dir,
                **repo
            )
        return cls(lmdb_path=base_lmdb_dir)

    def get_embedding(self, word: str) -> np.ndarray:
        """ Retrieve embedding for a word.

        If the word is in the word_embeddings, return that.
        Otherwise, compute n-grams and fetch n-gram embeddings.

        Args:
            word: Word to retrieve embedding for.
        Returns:
            np.ndarray: Embedding for the word if possible, else 
            the origin vector.
        """
        env = lmdb.open(self.lmdb_path, readonly=True)

        with env.begin() as txn:
            word_embedding_bytes = txn.get(word.encode('utf-8'))
            # If the word is in the vocab, return immediately.
            if word_embedding_bytes:
                return np.frombuffer(word_embedding_bytes, dtype=np.float32)

            # Otherwise, compute its n-grams using gensim's ft_ngram_hashes.
            ngram_embeddings = []
            for ngram_hash in fasttext.ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket):
                ngram_key = f"subword_{ngram_hash}".encode('utf-8')
                ngram_embedding_bytes = txn.get(ngram_key)
                if ngram_embedding_bytes:
                    ngram_embeddings.append(np.frombuffer(ngram_embedding_bytes, dtype=np.float32))

            if not ngram_embeddings:
                return np.zeros(self.vector_size, dtype=np.float32)

            return np.mean(ngram_embeddings, axis=0)

    @property
    @functools.lru_cache(maxsize=1)
    def vocab_size(self) -> int:
        """Get the size of the vocabulary (words)."""
        return self._get_metadata(b'vocab_size')

    @property
    @functools.lru_cache(maxsize=1)
    def ngram_size(self) -> int:
        """Get the size of the n-gram (subword) embeddings."""
        return self._get_metadata(b'ngram_size')

    @property
    @functools.lru_cache(maxsize=1)
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._get_metadata(b'embedding_dim')

    def _get_metadata(self, key: bytes) -> int:
        """Helper function to retrieve metadata."""
        # Open LMDB environment (read-only)
        env = lmdb.open(self.lmdb_path, readonly=True)

        with env.begin() as txn:
            # Fetch metadata associated with the key
            metadata_bytes = txn.get(key)
            if metadata_bytes:
                return int.from_bytes(metadata_bytes, byteorder='little')
        return 0

    def tokenize(self, sentence: str) -> List[str]:
        """Tokenize a sentence into words by whitespace."""
        return re.findall(r'\b\w+\b', sentence.lower())

    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        """Retrieve the sentence vector by averaging the embeddings of the words in the sentence.
        If any word is out-of-vocabulary, fall back to n-gram embeddings.
        """
        if not sentence:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        words = self.tokenize(sentence)

        # Adapted from https://github.com/facebookresearch/fastText/blob/1142dc4c4ecbc19cc16eee5cdd28472e689267e6/src/fasttext.cc#L490.
        count = 0
        sentence_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        for word in words:
            word_vec = self.get_embedding(word)
            norm = np.linalg.norm(word_vec)
            if not norm:
                continue
            word_vec = word_vec / norm
            sentence_vector = sentence_vector + word_vec
            count += 1

        if count > 0:
            sentence_vector /= count  # Average the vectors

        return sentence_vector
