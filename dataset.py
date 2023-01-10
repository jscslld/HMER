import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self,
        root_dir, # 图片所在目录
        image_filenames, # 图片文件名
        formulas, # 公式
        transform= None, # 图片变换算子
    ):
        super().__init__()
        assert len(image_filenames) == len(formulas),"图片长度和公式长度不一致"
        self.root_dir = root_dir
        self.image_filenames = image_filenames
        self.formulas = formulas
        self.transform = transform

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx: int):
        """
        对于给定的序号idx，返回对应的图片和公式
        """
        image_filename, formula = self.image_filenames[idx], self.formulas[idx]
        image_filepath = self.root_dir / image_filename
        if not image_filepath.is_file():
            image = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
            formula = []
        else:
            with open(image_filepath, "rb") as f:
                img = Image.open(f)
                image = img.convert("L") # 转换为灰度图像
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"] # 对图像进行变换
        return image, formula