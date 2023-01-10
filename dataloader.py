import random
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils import  get_dataset
from tokenlizer import Tokenizer
from dataset import BaseDataset


class CustomDataLoader(LightningDataModule):
    def __init__(
        self,
        batch_size= 8,
        num_workers= 0,
        pin_memory= False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.base_dir = Path(__file__).resolve().parent
        self.data_dirname = self.base_dir / "data"
        self.vocab_file = self.data_dirname / "formulas/vocab.txt"
        self.images_dirname = self.data_dirname / "images"
        self.transform = {
            "train": A.Compose(
                [
                    A.Affine(scale=(1.0, 1.0), rotate=(-1, 1), cval=255, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    ToTensorV2(),
                ]
            ),
            "val/test": ToTensorV2(),
        }

    def setup(self, stage = None):
        """
        构建训练、测试、验证数据集
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)

        if stage in ("fit", None):
            train_image_names, train_formulas = get_dataset(
                "images_train",
                self.data_dirname / "matching/train.matching.txt",
                self.data_dirname / "formulas/train.formulas.norm.txt",
            )
            self.train_dataset = BaseDataset(
                self.images_dirname,
                image_filenames=train_image_names,
                formulas=train_formulas,
                transform=self.transform["train"],
            )

            val_image_names, val_formulas = get_dataset(
                "images_val",
                self.data_dirname / "matching/val.matching.txt",
                self.data_dirname / "formulas/val.formulas.norm.txt",
            )
            self.val_dataset = BaseDataset(
                self.images_dirname,
                image_filenames=val_image_names,
                formulas=val_formulas,
                transform=self.transform["val/test"],
            )

        if stage in ("test", None):
            test_image_names, test_formulas = get_dataset(
                "images_test",
                self.data_dirname / "matching/test.matching.txt",
                self.data_dirname / "formulas/test.formulas.norm.txt",
            )
            self.test_dataset = BaseDataset(
                self.images_dirname,
                image_filenames=test_image_names,
                formulas=test_formulas,
                transform=self.transform["val/test"],
            )

    def collate_fn(self, batch):
        """
        用于整理数据，方便批量训练。

        batch :是一个列表，列表的长度是 batch_size，列表的每一个元素是 (image,formula) 这样的元组tuple。
        """
        images, formulas = zip(*batch) # 把输入的每个batch拆成图片列表和公式列表
        batch_size = len(images) # 当前这个batch的长度
        max_H = max(image.shape[1] for image in images) # 当前batch中图片的最大高度
        max_W = max(image.shape[2] for image in images) # 当前batch中图片的最大宽度
        max_length = max(len(formula) for formula in formulas) # 当前batch中公式的最大长度
        padded_images = torch.zeros((batch_size, 1, max_H, max_W))
        batched_indices = torch.zeros((batch_size, max_length + 2), dtype=torch.long)
        """
        循环遍历当前batch中的每个图片和对应的公式。
        把每张图片都补全到max_H*max_W的大小。补全方法为填充空白边缘，为了提高模型的鲁棒性，填充时边缘的宽度随机生成。
        把每张图片对应的公式的token序列填充到max_length + 2长度，填充方式为在末尾补零，因为零在字典中代表的是<PAD>，即占位符。
        """
        for i in range(batch_size):
            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y : y + H, x : x + W] = images[i]
            indices = self.tokenizer.encode(formulas[i])
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_images, batched_indices

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True, # 打乱训练集
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
