import os

from torch.utils.data import DataLoader

from .fsc147 import FSC147Dataset, MainTransform, get_query_transforms, batch_collate_fn

def get_fsc_loader(split, batch_size, args=None):
    main_transform = MainTransform()
    query_transform = get_query_transforms(True, (128,128))
    if split=='train': 
        data_list ='./datasets/FSC147/train.txt'
    elif split=='val':
        data_list = './datasets/FSC147/val.txt'
    elif split=='test':    
        data_list = './datasets/FSC147/test.txt'
    
    if not args:
        dataset = FSC147Dataset(data_dir=os.environ.get("FSC147_IMAGE_ROOT", "data/FSC147_384_V2"),
                            data_list=data_list,
                            scaling=1.0,
                            box_number=3,
                            scale_number=1,
                            main_transform=main_transform,
                            query_transform=query_transform,
                            split=split)
    else:
        dataset = FSC147Dataset(data_dir=os.environ.get("FSC147_IMAGE_ROOT", "data/FSC147_384_V2"),
                    data_list=data_list,
                    scaling=1.0,
                    box_number=3,
                    scale_number=1,
                    main_transform=main_transform,
                    query_transform=query_transform,
                    split=split,
                    horizon_flip=args.horizon_flip,
                    horizon_flip_prob=args.horizon_flip_prob,
                    vertical_flip=args.vertical_flip,
                    vertical_flip_prob=args.vertical_flip_prob)
    shuffle = True if split == 'train' else False
    split_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=batch_collate_fn)
    return split_loader
    



