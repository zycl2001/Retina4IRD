import os

import pandas as pd
from torchvision import transforms
from utils import create_transform
from .dataset_folder_multi_cls import MultiTaskImageFolderFromCSV


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    csv_data = pd.read_excel(os.path.join(args.csv_path, f'{is_train}.xlsx'), dtype={args.label_column: int})
    return MultiTaskImageFolderFromCSV(args.data_path, args.in_domains, csv_data, transform=transform, nb_classes=args.nb_classes, label_column=args.label_column)



def build_dataset_only_testing(is_train, args):
    transform = build_transform('val', args)
    csv_data = pd.read_excel(os.path.join(args.csv_path, f'{is_train}.xlsx'), dtype={args.label_column: int})
    return MultiTaskImageFolderFromCSV(args.data_path, args.in_domains, csv_data, transform=transform, nb_classes=args.nb_classes, label_column=args.label_column)



def build_transform(is_train, args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train=='train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

