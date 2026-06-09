import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }

        # Collect all unique lowercase words
        words = set()

        for text in texts:
            words.update(text.lower().split())

        # Add words in sorted order
        for idx, word in enumerate(sorted(words), start=4):
            self.word_to_id[word] = idx

        # Reverse mapping
        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in text.lower().split()
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        )