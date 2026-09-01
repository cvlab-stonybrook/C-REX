from PIL import Image, ImageOps
import groundingdino.datasets.transforms as T
from typing import Tuple
import numpy as np
import torch
import random
from groundingdino.util.misc import interpolate
import torchvision.transforms.functional as F
import os
import cv2
# import io

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None
    
    rescaled_densities = []
    for t in target:
        count = np.sum(t)
        rescaled_density = cv2.resize(t, dsize=[size[1], size[0]], interpolation=cv2.INTER_LINEAR)
        count_new = np.sum(rescaled_density)
        rescaled_density = (count / count_new) * rescaled_density
        rescaled_density = torch.from_numpy(rescaled_density)
        rescaled_densities.append(rescaled_density)
        # rescaled_density = F.resize(target, size)
    return rescaled_image, rescaled_densities

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)
    
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), torch.stack(target, dim=0)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def load_image(image_path: str, density_dir: str, density_names: list, client=None) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ## TODO: ceph 读取图像和density map
    # img_url = image_path
    # img_bytes = client.get(img_url)
    # image_source = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 本地读取直接修改这个就行
    image_source = Image.open(image_path).convert("RGB")
    
    image_source = ImageOps.exif_transpose(image_source)
    # image_source = Image.open(image_path).convert("RGB")
    density_maps = []
    for name in density_names:
        # density_path = os.path.join(density_dir, name.strip('.')+'.npy')
        # density_bytes = client.get(density_path)
        # density_source = np.load(io.BytesIO(density_bytes))
        
        density_source = np.load(density_path)
        density_maps.append(density_source)
    image = np.asarray(image_source)
    # cv2.imwrite('image.jpg', image[:,:,[2,1,0]])
    # cv2.imwrite('density.jpg', max_min(density_maps[1]) * 255)
    image_transformed, density_transformed = transform(image_source, density_maps)
    return image, image_transformed, density_transformed

def max_min(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

if __name__ == '__main__':
    img_pth = '/data2/wangzhicheng/Code/referring-expression-counting/datasets/rec-8k/0800-nwpu-3913.jpg'
    density_dir = '/data2/wangzhicheng/Code/referring-expression-counting/datasets/density_maps/0800-nwpu-3913'
    density_names = os.listdir(density_dir)
    image, image_transformed = load_image(img_pth, density_dir, density_names)
    print(1)