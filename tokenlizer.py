import json
from pathlib import Path
from typing import Dict, Optional, List, Union

class Tokenizer:
    """
    参考https://github.com/iankur/img2tex/blob/main/dataloader/utils.py
    """
    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<PAD>" # 填充标识符
        self.sos_token = "<SOS>" # 开始标识符
        self.eos_token = "<EOS>" # 结束标识符
        self.unk_token = "<UNK>" # 未知标识符

        assert token_to_index, "vocabulary with mapping from token to id?"
        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        self.token_to_index = token_to_index # token -> 索引的映射
        self.index_to_token = {index: token for token, index in self.token_to_index.items()} # 索引 -> token的映射
        self.pad_index = self.token_to_index[self.pad_token]
        self.sos_index = self.token_to_index[self.sos_token]
        self.eos_index = self.token_to_index[self.eos_token]
        self.unk_index = self.token_to_index[self.unk_token]

        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index} # 忽略特殊用途的token

    def __len__(self):
        return len(self.token_to_index)

    def encode(self, formula: List[str]) -> List[int]:
        """
        把公式转为索引序列，需要在开头加上sos标识符，结尾加上eos标识符。表示这条公式的开始和结束
        """
        indices = [self.sos_index]
        for token in formula:
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        """
        把索引序列重新转换为公式
        """
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    @classmethod
    def load(cls, filename: Union[Path, str]) -> "Tokenizer":
        """
        加载词典，格式为json，键为token，值为index
        """
        with open(filename) as f:
            data = f.readlines()
        token_to_index = {}
        for i in range(0,len(data)):
            token_to_index[data[i].strip()]=i
        return cls(token_to_index)