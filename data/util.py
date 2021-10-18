'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

class CMNISTDataset(Dataset):
    def __init__(self,root,split,transform=None, image_path_list=None):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*"))            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root

        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]

transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),]),
        },
    }


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "cifar10c": {
        "train": T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
}



transforms_preprcs_ae = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    },
    "cifar10c": {
        "train": T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
}
def get_dataset(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None, use_type0=None, use_type1=None):
    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

    elif 'cifar10c' in dataset:
        # if use_type0:
        #     root = data_dir + f"/cifar10c_0805_type0/{percent}"
        # elif use_type1:
        #     root = data_dir + f"/cifar10c_0805_type1/{percent}"
        # else:
        root = data_dir + f"/cifar10c/{percent}"
        dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=use_type0, use_type1=use_type1)

    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset

