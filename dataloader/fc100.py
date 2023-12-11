import os
import os.path as osp
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_path = 'path to data'


class FC100(Dataset):
    def __init__(self, setname, args):
        dataset_dir = os.path.join(data_path, 'FC100/')
        if setname == 'train':
            path = osp.join(dataset_dir, 'train')
            label_list = os.listdir(path)
        elif setname == 'test':
            path = osp.join(dataset_dir, 'test')
            label_list = os.listdir(path)
        elif setname == 'val':
            path = osp.join(dataset_dir, 'val')
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
            transforms.RandomResizedCrop(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225])),
        ])
        self.transform_val_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
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
