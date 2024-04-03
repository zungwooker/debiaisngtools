from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from glob import glob
import os
import torch
    
    
class CMNIST_ZWDataset(Dataset):
    def __init__(self, split, conflict_ratio):
        super().__init__()
        root = '/mnt/sdc/zungwooker/workspace/data/dataset/cmnist-zw'
        self.transform = T.ToTensor()

        if conflict_ratio != 'unbiased': 
            conflict_ratio = str(conflict_ratio)+'pct'

        if split=='train':
            self.data = glob(os.path.join(root, f'{conflict_ratio}', 'train', '*', '*'))
        elif split=='valid':
            self.data = glob(os.path.join(root, f'{conflict_ratio}', 'valid', '*', '*'))
        elif split=='test':
            self.data = glob(os.path.join(root, 'cmnist-zw', 'test', '*', '*'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.LongTensor([int(self.data[index].split('_')[-2])])
        bias = torch.LongTensor([int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform(image)
        
        return image, label, bias, self.data[index]


# class CMNISTDataset(Dataset):
#     def __init__(self, root, split, conflict_ratio, transform=None, selected_label=None):
#         super().__init__()
#         self.transform = transform
#         self.root = root

#         if conflict_ratio != 'unbiased': conflict_ratio = str(conflict_ratio)+'pct'

#         if split=='train':
#             self.align = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'align', '*', '*'))
#             self.conflict = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'conflict', '*', '*'))
#             self.data = self.align + self.conflict
#         elif split=='valid':
#             if selected_label is None:
#                 self.data = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'valid', '*'))
#             else:
#                 self.data = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'valid', f'*_{selected_label}_*'))
#         elif split=='test':
#             if selected_label is None:
#                 self.data = glob(os.path.join(root, 'cmnist', 'test', '*', '*'))
#             else:
#                 self.data = glob(os.path.join(root, 'cmnist', 'test', f'{selected_label}', '*'))
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         label, bias = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)
        
#         return image, label, self.data[index], bias
    

# class bFFHQDataset(Dataset):
#     def __init__(self, root, split, transform=None):
#         super().__init__()
#         self.transform = transform
#         self.root = root

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align',"*","*"))
#             self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
#             self.data = self.align + self.conflict

#         elif split=='valid':
#             self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

#         elif split=='test':
#             self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
#             data_conflict = []
#             for path in self.data:
#                 target_label = path.split('/')[-1].split('.')[0].split('_')[1]
#                 bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
#                 if target_label != bias_label:
#                     data_conflict.append(path)
#             self.data = data_conflict

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, self.data[index]
    

# class BARDataset(Dataset):
#     def __init__(self, root, split, transform=None, percent=None):
#         super(BARDataset, self).__init__()
#         self.transform = transform
#         self.percent = percent # 1pct, 5pct
#         self.split = split

#         self.train_align = glob(os.path.join(root,'train/align',"*/*"))
#         self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
#         self.valid = glob(os.path.join(root,'valid',"*/*"))
#         self.test = glob(os.path.join(root,'test',"*/*"))

#         if self.split=='train':
#             self.data = self.train_align + self.train_conflict
#         elif self.split=='valid':
#             self.data = self.valid
#         elif self.split=='test':
#             self.data = self.test

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         attr = torch.LongTensor(
#             [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')
#         image_path = self.data[index]

        
#         if 'bar/train/conflict' in image_path:
#             attr[1] = (attr[0] + 1) % 6 # assign non-related attribute to bias attribute
#         elif 'bar/train/align' in image_path:
#             attr[1] = attr[0]

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, (image_path, index)
    

# class DogCatDataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None):
#         super(DogCatDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image_path_list = image_path_list

#         if split == "train":
#             self.align = glob(os.path.join(root, "align", "*", "*"))
#             self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
#             self.data = self.align + self.conflict
#         elif split == "valid":
#             self.data = glob(os.path.join(root, split, "*"))
#         elif split == "test":
#             self.data = glob(os.path.join(root, "../test", "*", "*"))
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor(
#             [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, self.data[index]
