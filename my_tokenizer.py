from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import pickle
import os




class BaseTokenizer(ABC):
    """
    Base abstract class for tokenizers.
    Students will implement their own tokenizers derived from this class.
    """
    
    def __init__(self):
        """Initialize the tokenizer with empty vocabulary"""
        self.token_to_id = {}  # Token to ID mapping
        self.id_to_token = {}  # ID to token mapping
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
        }
        
        # Initialize special tokens in the mappings
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        pass
            
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        
        """
        Convert a batch of text strings into token IDs
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            A list of lists containing token IDs
        """
        return [self.encode(text) for text in texts]
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass


    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        """
        Convert a batch of token ID lists back to text strings
        
        Args:
            batch_token_ids: List of lists containing token IDs
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(token_ids) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary
        
        Returns:
            The number of tokens in the vocabulary
        """
        return len(self.token_to_id)
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file
        
        Args:
            path: Path where the tokenizer will be saved
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """
        Load a tokenizer from a file
        
        Args:
            path: Path to the saved tokenizer
            
        Returns:
            The loaded tokenizer
        """
        with open(path, 'rb') as f:
            return pickle.load(f) 
        


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
    def train(self, texts: List[str]) -> None:
        _initialvocabsize = self.get_vocab_size()
        while self.get_vocab_size() - _initialvocabsize < self.vocab_size:
            _vocab = {}
            for line in texts:
                line = line.strip().lower()
                words = line.split()
                for word in words:
                    chars = tuple(list(word) + ["</w>"])
                    _vocab[chars] = _vocab.get(chars, 0) + 1

            _pairs = {}
            for example in _vocab:
                for (a, b) in zip(example, example[1:]):
                    _pairs[(a, b)] = _pairs.get((a, b), 0) + _vocab[example]

            _maxpair = max(_pairs, key=_pairs.get)
            _mergedvocab = {}
            a, b = _maxpair
            ab = a + b

            for _word, _frequency in _vocab.items():
                new_word = []
                i = 0
                while i < len(_word):
                    if i < len(_word) - 1 and _word[i] == a and _word[i + 1] == b:
                        new_word.append(ab)
                        i += 2
                    else:
                        new_word.append(_word[i])
                        i += 1
                _mergedvocab[tuple(new_word)] = _mergedvocab.get(tuple(new_word), 0) + _frequency

            _vocab = _mergedvocab
            if ab not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[ab] = new_id
                self.id_to_token[new_id] = ab

    def encode(self, text: str) -> List[int]:
        _lower = text.strip().lower().split()
        _tokens = []

        for _word in _lower:
            word_tokens = list(_word) + ["</w>"]

            while True:
                pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]
                merge_candidate = None

                for pair in pairs:
                    merged = pair[0] + pair[1]
                    if merged in self.token_to_id:
                        merge_candidate = pair
                        break

                if merge_candidate is None:
                    break

                _new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == merge_candidate[0] and word_tokens[i + 1] == merge_candidate[1]:
                        _new_tokens.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        _new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = _new_tokens

            for _token in word_tokens:
                _token_id = self.token_to_id.get(_token, self.token_to_id["[UNK]"])
                _tokens.append(_token_id)

        return [self.token_to_id["[BOS]"]] + _tokens + [self.token_to_id["[EOS]"]]

    def decode(self, token_ids: List[int]) -> str:
        _filtered_ids = [token_id for token_id in token_ids if token_id not in self.special_tokens.values()]
        _tokens = [self.id_to_token.get(token_id, "[UNK]") for token_id in _filtered_ids]

        _words = []
        _current_word = []

        for _token in _tokens:
            _current_word.append(_token)
            if _token.endswith("</w>"):
                word = "".join(_current_word).replace("</w>", "")
                _words.append(word)
                _current_word = []

        return " ".join(_words).strip()
