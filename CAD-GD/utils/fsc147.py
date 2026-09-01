"""
FSC-147 dataset
The exemplar boxes are sampled and resized to the same size
"""


from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import json
import torch
import numpy as np
from torchvision.transforms import transforms
import io
import torchvision.transforms as T
from random import choice
import random

def get_image_classes(class_file):
    class_dict = dict()
    with open(class_file, 'r') as f:
        classes = [line.split('\t') for line in f.readlines()]
    
    for entry in classes:
        class_dict[entry[0]] = entry[1]
    
    return class_dict 

def batch_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0], scale_embedding, batch[2] = batch_padding(batch[0], batch[2])
    patches = torch.stack(batch[1], dim=0)
    batch[1] = {'patches': patches, 'scale_embedding': scale_embedding.long()}
    b,c,h,w = batch[0].shape
    b3 = torch.stack(batch[3], dim=0)
    batch[3] = torch.cat([b3[:,:,0],b3[:,:,2]],dim=-1)
    batch[3][:,:,[0,2]] = batch[3][:,:,[0,2]] / w
    batch[3][:,:,[1,3]] = batch[3][:,:,[1,3]] / h
    batch[4] = list(batch[4])
    return tuple(batch)

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def batch_padding(tensor_list, target_dict):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        density_shape = [len(tensor_list)] + [1, max_size[1], max_size[2]]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        density_map = torch.zeros(density_shape, dtype=dtype, device=device)
        pt_map = torch.zeros(density_shape, dtype=dtype, device=device)
        gtcount = []
        scale_embedding = []
        points = []
        for idx, package  in enumerate(zip(tensor_list, tensor, density_map, pt_map)):
            img, pad_img, pad_density, pad_pt_map = package
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_density[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['density_map'])
            pad_pt_map[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['pt_map'])
            gtcount.append(target_dict[idx]['gtcount']) 
            scale_embedding.append(target_dict[idx]['scale_embedding'])
            points.append(target_dict[idx]['points'])
        target = {'density_map': density_map,
                  'pt_map': pt_map,
                  'gtcount': torch.tensor(gtcount),
                  'points': points}
    else:
        raise ValueError('not supported')
    return tensor, torch.stack(scale_embedding), target

class FSC147Dataset(Dataset):
    def __init__(self, data_dir, data_list, scaling, box_number=3, scale_number=20, min_size=384, max_size=1584, preload=True, main_transform=None, query_transform=None, split='val',
                 vertical_flip=False, vertical_flip_prob=0.5, horizon_flip=False, horizon_flip_prob=0.5):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.scaling = scaling
        self.box_number = box_number
        self.scale_number = scale_number
        self.preload = preload
        self.main_transform = main_transform
        self.query_transform = query_transform
        self.min_size = min_size
        self.max_size = max_size 
        # load annotations for the entire dataset
        annotation_file = "./datasets/FSC147/annotation_FSC147_384.json"
        image_classes_file = "./datasets/FSC147/ImageClasses_FSC147.txt"
        image_classes_file_d= '/mnt/workspace_new/guijiang/code/CAD-GD-LOCAL-main/datasets/fsc147/FSC-147-D.json'
        self.image_classes = get_image_classes(image_classes_file)
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        """with open(image_classes_file_d, 'r') as f:
            self.image_classes_d = eval(f.read())"""

        self.image_classes_d = {}
        with open("./datasets/FSC147/ImageClasses_FSC147.txt", "r") as f:
            for line in f:
                # strip newline and split on first tab (or any whitespace)
                parts = line.strip().split(None, 1)  # split into 2 parts max
                if len(parts) == 2:
                    key, value = parts
                    self.image_classes_d[key] = value
        
        self.random_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.vertical_flip = vertical_flip
        self.vertical_flip_prob = vertical_flip_prob
        self.horizon_flip = horizon_flip
        self.horizon_flip_prob = horizon_flip_prob

        self.images = {}
        self.targets = {}
        self.patches = {}
        self.split = split

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx][0]

        image_path = os.path.join(self.data_dir, f'images/{self.split}/' + file_name)
        density_path = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2/' + file_name.replace('jpg', 'npy'))
        
        image_source = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(image_source)
        
        img_info = self.annotations[file_name]
        target_class = self.image_classes_d[file_name].replace('the ','') + '.'

        w, h = img.size
        # resize the image
        r = 1.0
        if r != 1.0:
            print("?")
        if h > self.max_size or w > self.max_size:
            r = self.max_size / max(h, w)
        if r * h < self.min_size or w*r < self.min_size:
            r = self.min_size / min(h, w)
        nh, nw = int(r*h), int(r*w)
        img = img.resize((nw, nh), resample=Image.BICUBIC)
    
        density_map = np.load(density_path).astype(np.float32)
        pt_map = np.zeros((nh, nw), dtype=np.int32)
        points = (np.array(img_info['points']) * r).astype(np.int32)
        boxes = np.array(img_info['box_examples_coordinates']) * r   
        boxes = boxes[:self.box_number, :, :]
        gtcount = points.shape[0]
        
        # crop patches and data transformation
        target = dict()
        patches = []
        scale_embedding = []
        
        #print('boxes:', boxes.shape[0])
        raw_points = points.copy()
        if points.shape[0] > 0:     
            points[:,0] = np.clip(points[:,0], 0, nw-1)
            points[:,1] = np.clip(points[:,1], 0, nh-1)
            pt_map[points[:, 1], points[:, 0]] = 1 
            for box in boxes:
                x1, y1 = box[0].astype(np.int32)
                x2, y2 = box[2].astype(np.int32)
                patch = img.crop((x1, y1, x2, y2))
                patches.append(self.query_transform(patch))
                # calculate scale
                scale = (x2 - x1) / nw * 0.5 + (y2 -y1) / nh * 0.5 
                scale = scale // (0.5 / self.scale_number)
                scale = scale if scale < self.scale_number - 1 else self.scale_number - 1
                scale_embedding.append(scale)
        target['points'] = raw_points
        target['density_map'] = density_map * self.scaling
        target['pt_map'] = pt_map
        target['gtcount'] = gtcount
        target['scale_embedding'] = torch.tensor(scale_embedding)
        img, target = self.main_transform(img, target)
        patches = torch.stack(patches, dim=0)

        if self.split != 'train':
            # 定义 Resize 操作，最短边放大为 800
            resize_transform = T.Resize(800)
            # 执行放大操作
            resized_img = resize_transform(img)
            resized_density_map = resize_transform(target['density_map'])
            resized_density_map = resized_density_map * torch.sum(target['density_map']) / torch.sum(resized_density_map)
            target['density_map'] = resized_density_map
            target['pt_map'] = resize_transform(target['pt_map'])
            ratio = resized_img.shape[1] / img.shape[1]
            target['points'] = target['points'] * ratio
            
            resized_boxes = boxes * ratio
            return resized_img, patches, target, torch.tensor(resized_boxes), target_class, file_name

        if self.split == 'train':
            img, target, boxes = self.augumentation(img, target, boxes)

        return img, patches, target, torch.tensor(boxes), target_class, file_name

    def augumentation(self, img, target, boxes):
        random_size = choice(self.random_size)
        resize_transform = T.Resize(random_size)
        resized_img = resize_transform(img)
        resized_density_map = resize_transform(target['density_map'])
        resized_density_map = resized_density_map * torch.sum(target['density_map']) / torch.sum(resized_density_map)
        target['density_map'] = resized_density_map
        target['pt_map'] = resize_transform(target['pt_map'])
        ratio = resized_img.shape[1] / img.shape[1]
        target['points'] = target['points'] * ratio
        resized_boxes = boxes * ratio
        
        ## setting vertical flip
        if self.vertical_flip and random.random() < self.vertical_flip_prob:
            # resized_img = resized_img[:,::-1,:]
            H = resized_img.shape[1]
            resized_img = torch.flip(resized_img, [1])
            target['density_map'] = torch.flip(target['density_map'], [1])
            target['pt_map'] = torch.flip(target['pt_map'], [1])
            target['points'][:,1] = H - target['points'][:,1]
        
        ## setting horizontal flip
        if self.horizon_flip and random.random() < self.horizon_flip_prob:
            # resized_img = resized_img[:,::-1,:]
            W = resized_img.shape[2]
            resized_img = torch.flip(resized_img, [2])
            target['density_map'] = torch.flip(target['density_map'], [2])
            target['pt_map'] = torch.flip(target['pt_map'], [2])
            target['points'][:,0] = W - target['points'][:,0]
        
        return resized_img, target, resized_boxes
        


def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize-h%psize),(psize-w%psize)
    (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)   
    (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
    if (ph!=psize) or (pw!=psize):
        tmp_pad = [pl, pr, pt, pb]
        inputs = torch.nn.functional.pad(inputs, tmp_pad)
    
    return inputs

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img, target):
        img = self.img_trans(img)
        density_map = target['density_map']
        pt_map = target['pt_map']
        pt_map = torch.from_numpy(pt_map).unsqueeze(0)
        density_map = torch.from_numpy(density_map).unsqueeze(0)
        target['density_map'] = density_map.float()
        target['pt_map'] = pt_map.float()
        
        return img, target


def get_query_transforms(is_train, exemplar_size):
    if is_train:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def build_dataset(cfg, is_train):
    main_transform = MainTransform()
    query_transform = get_query_transforms(is_train, cfg.DATASET.exemplar_size)
    if is_train: 
        data_list = cfg.DATASET.list_train
    else:
        if not cfg.VAL.evaluate_only:
            data_list = cfg.DATASET.list_val 
        else:
            data_list = cfg.DATASET.list_test
    
    dataset = FSC147Dataset(data_dir=cfg.DIR.dataset,
                            data_list=data_list,
                            scaling=1.0,
                            box_number=cfg.DATASET.exemplar_number,
                            scale_number=cfg.MODEL.ep_scale_number,
                            main_transform=main_transform,
                            query_transform=query_transform)
    
    return dataset
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_transform = MainTransform()
    query_transform = get_query_transforms(is_train=True, exemplar_size=(128, 128))
    
    dataset = FSC147Dataset(data_dir='s3://fsc147',
                            data_list='/mnt/petrelfs/wangzhicheng/DATASETS/counting/fsc-147/train.txt',
                            scaling=1.0,
                            main_transform=main_transform,
                            query_transform=query_transform)
    
    data_loader = DataLoader(dataset, batch_size=5, collate_fn=batch_collate_fn)
    
    for idx, sample in enumerate(data_loader):
        img, patches, targets, boxes = sample
        print(img.shape)
        print(boxes.shape)
        print(targets['density_map'].shape)
        break

    
    

    
