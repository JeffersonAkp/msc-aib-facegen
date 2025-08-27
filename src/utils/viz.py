import os
from torchvision.utils import save_image

def save_tensor_images(tensor, out_path="outputs/sample.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # remap [-1,1] -> [0,1]
    img = (tensor + 1) / 2
    save_image(img, out_path)
    return out_path
