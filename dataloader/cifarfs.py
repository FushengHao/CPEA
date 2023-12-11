import os
import os.path as osp
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_path = 'path to data'


class CIFARFS(Dataset):
    def __init__(self, setname, args):
        dataset_dir = os.path.join(data_path, 'cifar_fs/')
        if setname == 'train':
            path = osp.join(dataset_dir, 'meta-train')
            label_list = os.listdir(path)
        elif setname == 'test':
            path = osp.join(dataset_dir, 'meta-test')
            label_list = os.listdir(path)
        elif setname == 'val':
            path = osp.join(dataset_dir, 'meta-val')
            label_list = os.listdir(path)
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []

        folders = [osp.join(path, label) for label in label_list if os.path.isdir(osp.join(path, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        # Transformation
        image_size = 224
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.5071, 0.4866, 0.4409]),
                                 np.array([0.2009, 0.1984, 0.2023])),
            transforms.RandomErasing(value=[0.5071, 0.4866, 0.4409]),
        ])
        self.transform_val_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.5071, 0.4866, 0.4409]),
                                 np.array([0.2009, 0.1984, 0.2023]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.setname == 'train':
            image = self.transform_train(Image.open(path).convert('RGB'))
        else:
            image = self.transform_val_test(Image.open(path).convert('RGB'))
        return image, label
