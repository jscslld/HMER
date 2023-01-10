import cv2
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2

from lit_model import LitModel
import numpy as np
buckets = ([240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
           [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
           [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
           [1000, 400], [1200, 200], [1600, 200],
           )


def get_new_size(old_size, buckets=buckets, ratio=2):
    if buckets is None:
        return old_size
    else:
        w, h = old_size[0] / ratio, old_size[1] / ratio
        for (idx, (w_b, h_b)) in enumerate(buckets):
            if w_b >= w and h_b >= h:
                return w_b, h_b, idx

    return old_size


def data_turn(img_data, pad_size=(8, 8, 8, 8), resize=False):
    # 找到字符区域边界
    nnz_inds = np.where(img_data != 255)
    y_min = np.min(nnz_inds[1])
    y_max = np.max(nnz_inds[1])
    x_min = np.min(nnz_inds[0])
    x_max = np.max(nnz_inds[0])
    old_im = img_data[x_min:x_max + 1, y_min:y_max + 1]

    # 添加padding
    top, left, bottom, right = pad_size
    old_size = (old_im.shape[0] + left + right, old_im.shape[1] + top + bottom)
    new_im = np.ones(old_size, dtype=np.uint8) * 255
    new_im[top:top + old_im.shape[0], left:left + old_im.shape[1]] = old_im
    if resize:
        new_size = get_new_size(old_size, buckets)[:2]
        new_im = cv2.resize(new_im, new_size, cv2.INTER_LANCZOS4)
    return new_im

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = LitModel.load_from_checkpoint("model/cer=0.10.ckpt").cuda(device) # 请修改权重目录
lit_model.freeze()
transform = ToTensorV2()
image = Image.open("data/images/images_test/15.png").convert("L")
image_tensor = transform(image=data_turn(np.array(image)))["image"]  # type: ignore
image_tensor = torch.as_tensor(image_tensor).to(device)
pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]  # type: ignore
decoded = lit_model.tokenizer.decode(pred.tolist())  # type: ignore
decoded_str = " ".join(decoded)
print(decoded_str)
