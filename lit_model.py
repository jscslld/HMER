from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from tokenlizer import Tokenizer
from resnet_transformer import ResNetTransformer
from metrics import CharacterErrorRate


class LitModel(LightningModule):
    def __init__(
        self,
        d_model: int, # Transformer输入向量的维度，也可以理解成特征提取层提取后输出向量的维度
        dim_feedforward: int, # Transformer解码器前馈神经网络(Linear层)的中间层输出维度，因为它的结构式linear1->linear2，即linear1的输出，linear2的输入
        nhead: int, # muti-head self attention中head的数量
        dropout: float, # 以dropout的比例丢弃神经元
        num_decoder_layers: int, # Transformer解码器层数
        max_output_len: int, # 最大输出长度
        lr: float = 0.001, # 学习率
        weight_decay: float = 0.0001, # 权重衰减，用于避免过拟合
        milestones=None, # 学习率调整节点
        gamma: float = 0.1, # 学习率调整倍率
    ):
        super().__init__()
        if milestones is None:
            milestones = [5]
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        vocab_file = Path(__file__).resolve().parent / "data/formulas/vocab.txt"
        self.tokenizer = Tokenizer.load(vocab_file)
        self.model = ResNetTransformer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.sos_index,
            eos_index=self.tokenizer.eos_index,
            pad_index=self.tokenizer.pad_index,
            num_classes=len(self.tokenizer),
        ) # 根据给定超参数，构建神经网络模型
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index) # 使用交叉熵损失函数
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices) # 计算验证集错误率
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices) # 计算测试集错误率

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1]) # 结合teacher forcing进行前向传播，计算每个位置的概率
        loss = self.loss_fn(logits, targets[:, 1:]) # 计算当前输出的交叉熵损失函数值
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1]) # 结合teacher forcing进行前向传播，计算每个位置的概率
        loss = self.loss_fn(logits, targets[:, 1:]) # 计算当前输出的交叉熵损失函数值
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(imgs) # 用当前epoch下训练出的模型，预测验证集的输出
        val_cer = self.val_cer(preds, targets) # 将验证集的预测输出与真实答案计算字符误差率
        self.log("val/cer", val_cer)

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.model.predict(imgs) # 用当前epoch下训练出的模型，预测验证集的输出
        test_cer = self.test_cer(preds, targets) # 将验证集的预测输出与真实答案计算字符误差率
        self.log("test/cer", test_cer)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay) # 使用AdamW优化器优化参数
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma) # 动态学习率调整
        return [optimizer], [scheduler]
