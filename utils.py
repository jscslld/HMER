from typing import Union
import torch
from torch import Tensor


def get_dataset(dirname, img_file, formula_file):
    image_names = []
    formulas = []
    with open(formula_file) as f:
        all_formulas = [line.strip().split() for line in f.readlines()]
    with open(img_file) as f:
        for line in f:
            img_name, formula_idx = line.strip("\n").split()
            image_names.append(dirname + "/" + img_name)
            formulas.append(all_formulas[int(formula_idx)])
    return image_names, formulas


def generate_square_subsequent_mask(size: int) -> Tensor:
    """
    参考代码：https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/36cab9d6dcdad84e3d1a69df5ab796cbf689c115/lab9/text_recognizer/models/transformer_util.py

    生成上三角掩码矩阵
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    参考代码：https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/lit_models/util.py
    Return indices of first occurence of element in x. If not found, return length of x along dim.
    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind
