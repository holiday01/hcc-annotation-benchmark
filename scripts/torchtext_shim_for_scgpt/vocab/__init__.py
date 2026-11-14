"""
Pure-Python stand-in for the part of torchtext.vocab that scGPT uses
(scgpt/tokenizer/gene_tokenizer.py: Vocab, vocab()).

torchtext 0.18 (its last release) is compiled against torch 2.3 and cannot be
loaded with the torch builds that support this GPU (sm_120, torch >= 2.7).
scGPT only uses torchtext for its token <-> index table, so the table is
re-implemented here with the same semantics as torchtext 0.18:
  * vocab[token] -> index; unknown token -> default index if set, else RuntimeError
  * vocab(tokens) -> list of indices
  * insert_token(token, index) shifts later tokens; append_token appends
  * get_stoi() / get_itos() / lookup_* / set_default_index / get_default_index
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import torch.nn as nn

__all__ = ["Vocab", "vocab", "build_vocab_from_iterator"]


class _VocabTable:
    def __init__(self, tokens: Optional[List[str]] = None, default_index: Optional[int] = None):
        self.itos_: List[str] = list(tokens or [])
        self.stoi_: Dict[str, int] = {}
        for i, t in enumerate(self.itos_):
            if t in self.stoi_:
                raise RuntimeError(f"Duplicate token {t!r} in vocab")
            self.stoi_[t] = i
        self.default_index_ = default_index

    def __len__(self) -> int:
        return len(self.itos_)

    def __contains__(self, token) -> bool:
        return token in self.stoi_

    def __getitem__(self, token: str) -> int:
        i = self.stoi_.get(token)
        if i is not None:
            return i
        if self.default_index_ is None:
            raise RuntimeError(f"Token {token} not found and default index is not set")
        return self.default_index_

    def insert_token(self, token: str, index: int) -> None:
        if token in self.stoi_:
            raise RuntimeError(f"Token {token} already exists in the Vocab with index: {self.stoi_[token]}")
        if index < 0 or index > len(self.itos_):
            raise RuntimeError(f"Specified index {index} is out of bounds for vocab of size {len(self.itos_)}")
        self.itos_.insert(index, token)
        if index == len(self.itos_) - 1:
            self.stoi_[token] = index
        else:
            self.stoi_ = {t: i for i, t in enumerate(self.itos_)}

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self.itos_))


class Vocab(nn.Module):
    def __init__(self, vocab) -> None:
        super().__init__()
        if isinstance(vocab, Vocab):
            vocab = vocab.vocab
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def __prepare_scriptable__(self):
        return self

    def set_default_index(self, index: Optional[int]) -> None:
        self.vocab.default_index_ = index

    def get_default_index(self) -> Optional[int]:
        return self.vocab.default_index_

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)

    def append_token(self, token: str) -> None:
        self.vocab.append_token(token)

    def lookup_token(self, index: int) -> str:
        if index < 0 or index >= len(self.vocab.itos_):
            raise RuntimeError(f"Specified index {index} is out of bounds of the size of itos dictionary: {len(self.vocab.itos_)}.")
        return self.vocab.itos_[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(i) for i in indices]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.vocab[t] for t in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return dict(self.vocab.stoi_)

    def get_itos(self) -> List[str]:
        return list(self.vocab.itos_)


def vocab(ordered_dict: Dict, min_freq: int = 1, specials: Optional[List[str]] = None,
          special_first: bool = True) -> Vocab:
    specials = specials or []
    for t in specials:
        ordered_dict.pop(t, None)
    tokens = [t for t, freq in ordered_dict.items() if freq >= min_freq]
    tokens = (specials + tokens) if special_first else (tokens + specials)
    return Vocab(_VocabTable(tokens))


def build_vocab_from_iterator(iterator: Iterable, min_freq: int = 1, specials: Optional[List[str]] = None,
                              special_first: bool = True, max_tokens: Optional[int] = None) -> Vocab:
    from collections import Counter
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    specials = specials or []
    ordered = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if max_tokens is not None:
        ordered = ordered[: max_tokens - len(specials)]
    return vocab(OrderedDict(ordered), min_freq=min_freq, specials=specials, special_first=special_first)
