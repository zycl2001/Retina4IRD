import os
import os.path

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')


def pil_loader(path: str, convert_rgb=True) -> Image.Image:
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img



class MultiTaskDatasetFolderFromCSV(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            nb_classes: int,
            label_column: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str, str]] = None,
            max_images: Optional[int] = None
    ) -> None:
        super(MultiTaskDatasetFolderFromCSV, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        self.tasks = tasks
        classes = [f'{i}' for i in range(nb_classes)]
        class_to_idx = {f'{i}': i for i in range(nb_classes)}

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = dict()
        for task in self.tasks:

            if task == 'rgb':
                samples['rgb'] = csv_data.apply(lambda row: (os.path.join(self.root, row['cfp']), row[label_column]), axis=1).tolist()
            else:
                samples[task] = csv_data.apply(lambda row: (os.path.join(self.root, row[task]), row[label_column]), axis=1).tolist()


        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        image_name_dict = dict()

        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path, target = self.samples[task][index]
                sample = pil_loader(path, convert_rgb=True)

                sample_dict[task] = sample

                image_name_dict[task] = os.path.relpath(path, self.root)


        if self.transform is not None:

            for task in self.tasks:
                sample_dict[task] = self.transform(sample_dict[task])
        if self.target_transform is not None:
            target = self.target_transform(target)


        return sample_dict, target, image_name_dict

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


class MultiTaskImageFolderFromCSV(MultiTaskDatasetFolderFromCSV):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            nb_classes: int,
            label_column: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ):
        super(MultiTaskImageFolderFromCSV, self).__init__(root, tasks, csv_data, nb_classes, label_column, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          prefixes=prefixes,
                                          max_images=max_images)
        self.imgs = self.samples