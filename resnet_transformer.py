import math
from typing import Union

import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor


from positional_encoding import PositionalEncodingImage, PositionalEncoding
from utils import first_element, generate_square_subsequent_mask


class ResNetTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len + 2 # 这里+2的目的是给开始和结束标识符留出位置
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index

        # 编码器结构
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        ) # 去掉resnet的分类部分，保留特征提取部分

        self.bottleneck = nn.Conv2d(256, self.d_model, 1) # 把ResNet提取的特征转为和Transformer的输入特征维度一致
        self.image_positional_encoder = PositionalEncodingImage(self.d_model) # 二维位置编码函数，用于对图片进行位置编码

        # 解码器结构
        self.embedding = nn.Embedding(num_classes, self.d_model) # 嵌入层
        self.y_mask = generate_square_subsequent_mask(self.max_output_len) # 上三角掩码
        self.word_positional_encoder = PositionalEncoding(self.d_model, max_len=self.max_output_len) # 对标签进行位置编码
        # 下面两步是利用标准组件构建了Transformer解码器结构
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)
        # 线性层，将解码器输出映射为概率
        self.fc = nn.Linear(self.d_model, num_classes)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        前向传播
        """
        encoded_x = self.encode(x)
        output = self.decode(y, encoded_x)
        output = output.permute(1, 2, 0)
        return output

    def encode(self, x: Tensor) -> Tensor:
        """
        编码模块

        x是批输入图像，形状是(batch_size,图像通道数channel_size,图像高度img_H,图像宽度img_W)
        """
        # 如果输入的图象是灰度图像（单通道图像），就把那一个通道复制三份扩充成三通道图像
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # 利用骨干网络进行特征编码，编码后的输出结果形状是(batch_size,ResNet输出通道数ResNet_out_plane,ResNet输出高度ResNet_H,ResNet输出宽度ResNet_W)
        x = self.backbone(x)
        # 用一个瓶颈层对ResNet的输出进行1*1卷积，进行数据降维，降维后的形状是(batch_size,BackBone输出通道数即Transformer输入通道数d_model,ResNet输出高度ResNet_H,ResNet输出宽度ResNet_W)
        x = self.bottleneck(x)
        # 对输入图像特征进行位置编码，编码后形状不变
        x = self.image_positional_encoder(x)
        # 把第3维和第4维压平成一维，输出形状为(batch_size,d_model,ResNet_H*ResNet_W)
        x = x.flatten(start_dim=2)
        # 根据Transformer解码器的输出要求，对形状进行变换，输出形状为(ResNet_H*ResNet_W,batch_size,d_model)，下面用Sx代替ResNet_H*ResNet_W
        x = x.permute(2, 0, 1)
        return x

    def decode(self, y: Tensor, encoded_x: Tensor) -> Tensor:
        """
        解码模块

        输入参数有两个：

        encoded_x：经编码器处理后的（批）编码序列，其形状为(Sx,batch_size,d_model)。在编码模块已经进行了详细解释，这里不再赘述。

        y: 这一个batch的真实标签，形状为(batch_size, Sy)，其中Sy表示的是该batch下的真实标签经过tokenlize后序列长度。
        """
        # 把标签转为适合进行embedding的形式
        y = y.permute(1, 0)
        # 对每一个标签的token_id转换为这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。乘以d_model的开方使得embedding matrix的方差是1。输出形状为(Sy, batch_size, d_model)
        y = self.embedding(y) * math.sqrt(self.d_model)
        # 嵌入位置编码
        y = self.word_positional_encoder(y)
        Sy = y.shape[0]
        # 生成一个Sy*Sy，左下角是0，右上角是-inf的上三角掩码，起主要作用是对未来信息进行遮掩。
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)
        # 对解码器模块进行前向传播，输出形状为(Sy,batch_size,d_model)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        # 经过一个线性层，得到每个token的概率预测。输出形状为(Sy, batch_size, num_classes)
        output = self.fc(output)
        return output

    def predict(self, x: Tensor) -> Tensor:
        """预测模块

        输入参数
            x: 输入图像

        输出参数:
            形状(batch_size, max_output_len) ，其中每个元素都在 (0, num_classes - 1) 之间，表示这个位置的token id.
        """
        batch_size = x.shape[0]
        max_output_len = self.max_output_len
        # 编码，输出尺寸(ResNet_H*ResNet_W,batch_size,d_model)
        encoded_x = self.encode(x)
        # 填充一个大小为batch_size*max_output_len的张量，每个元素都是占位符。其实就是初始化预测结果
        output_indices = torch.full((batch_size, max_output_len), self.pad_index).type_as(x).long()
        # 把每一行（就是batch里面每一张图片的预测结果）的第一个位置填成起始字符
        output_indices[:, 0] = self.sos_index
        # 用于标识这一行是否已经找到结尾
        has_ended = torch.full((batch_size,), False)

        for Sy in range(1, max_output_len):
            # 以下两步是输入这个batch的前Sy个字符进行预测，得到输出概率
            y = output_indices[:, :Sy]
            logits = self.decode(y, encoded_x)
            # 挑选概率最大的作为这个位置的token
            output = torch.argmax(logits, dim=-1)
            output_indices[:, Sy] = output[-1:]
            # 如果碰到EOS则把这个图片的状态设为END。
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            # 如果这个batch下所有图片都end了，就不再继续预测。
            if torch.all(has_ended):
                break

        # 找出这个batch中每个图片的输出token的结束标识符，在结束标识符后面填充<PAD>
        eos_positions = first_element(output_indices, self.eos_index)
        for i in range(batch_size):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index

        return output_indices
